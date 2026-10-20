# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3_5 MTP model."""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, tensor_model_parallel_all_gather
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.utils import (
    is_model_fused_shared_expert_compatible,
)
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import LocalArgmaxMixin
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5Model,
    Qwen3_5RMSNorm,
)
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextSparseMoeBlock,
    QwenNextMixtureOfExperts,
)
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5TextConfig
from vllm.transformers_utils.configs.qwen3_5_moe import Qwen3_5MoeTextConfig

from .interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    _require_is_multimodal,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    _merge_multimodal_embeddings,
    make_empty_intermediate_tensors_factory,
    maybe_fuse_shared_experts,
    maybe_prefix,
)

logger = init_logger(__name__)


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "hidden_states": 0,
    }
)
class Qwen3_5MultiTokenPredictor(nn.Module):
    hf_to_vllm_mapper = Qwen3_5Model.hf_to_vllm_mapper

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        model_config = vllm_config.model_config
        quant_config = vllm_config.quant_config

        config: Qwen3_5TextConfig | Qwen3_5MoeTextConfig = model_config.hf_text_config

        self.config = config

        self.vocab_size = config.vocab_size

        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = getattr(config, "mtp_num_hidden_layers", 1)
        # [PN348 vendor of vllm#44644] share embed+lm_head
        # with target model when checkpoint opts in (Qwen3.5/3.6 set
        # text_config.mtp_use_dedicated_embeddings=False). Saves
        # vocab*hidden*2 B = ~1 GiB/worker on Qwen3.6-35B-A3B-FP8.
        # PP=1 gate keeps legacy path on pipeline-parallel setups.
        self.share_backbone_input_output = (
            not getattr(config, "mtp_use_dedicated_embeddings", True)
            and get_pp_group().world_size == 1
        )
        if self.share_backbone_input_output:
            self.embed_tokens = PPMissingLayer()
        else:
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
            )

        # [P110 Qwen3 MTP GPTQ/AWQ fix v7.63.x guard-v4]
        # GPTQ/AWQ checkpoints store the WHOLE MTP predictor block (fc +
        # attention + MoE experts) as plain unquantized BF16 tensors, not
        # just `fc` (which upstream already special-cases for modelopt_fp4
        # right below). Building `self.layers` with the real quant_config
        # makes the MoE experts expect GPTQ-packed weights that never
        # arrive -> silently skipped at load -> MTP head stays randomly
        # initialized -> 0% spec-decode acceptance.
        # v0.27.1 re-anchor: upstream added a dynamic("-:")-only bypass
        # (mutate vllm_config.quant_config=None around self.layers, then
        # restore). We reuse that exact mechanism but EXTEND the predicate
        # to the quant-name family (gptq/awq/marlin/compressed-tensors...)
        # for checkpoints shipping unquantized MTP WITHOUT dynamic entries,
        # and also reach fc_quant. Only MTP predictor layers are affected —
        # the main model keeps full quantization untouched.
        _isv_p110_bypass_mtp_quant = False
        if quant_config and quant_config.get_name() not in ("modelopt_fp4",):
            _isv_p110_hf_qc = getattr(
                model_config.hf_config, "quantization_config", None
            )
            if isinstance(_isv_p110_hf_qc, dict):
                _isv_p110_dynamic = _isv_p110_hf_qc.get("dynamic", {})
                if any(
                    k.startswith("-:") and "mtp" in k
                    for k in _isv_p110_dynamic
                ):
                    _isv_p110_bypass_mtp_quant = True
            if not _isv_p110_bypass_mtp_quant and quant_config.get_name() in (
                "gptq", "auto_gptq", "gptq_marlin", "awq", "awq_marlin",
                "compressed-tensors", "compressed_tensors", "marlin", "compressed_tensors_wNa16",
            ):
                _isv_p110_bypass_mtp_quant = True

        # Workaround: mtp.fc is stored as BF16 in NVFP4 checkpoints but is
        # missing from hf_quant_config.json exclude_modules. Force unquantized.
        # Ref: https://github.com/vllm-project/vllm/pull/38650
        # Ref: https://github.com/NVIDIA/Model-Optimizer/pull/1124
        fc_quant = (
            None
            if (
                (quant_config and quant_config.get_name() == "modelopt_fp4")
                or _isv_p110_bypass_mtp_quant
            )
            else quant_config
        )
        self.fc = ColumnParallelLinear(
            self.config.hidden_size * 2,
            self.config.hidden_size,
            gather_output=True,
            bias=False,
            return_bias=False,
            quant_config=fc_quant,
            prefix=f"{prefix}.fc",
        )

        # GPTQ: quantized checkpoints may exclude MTP from quantization via
        # quantization_config.dynamic with "-:pattern" entries. When detected,
        # disable quantization for MTP layers so they use unquantized params.
        original_quant = vllm_config.quant_config
        if _isv_p110_bypass_mtp_quant:
            vllm_config.quant_config = None
        self.layers = torch.nn.ModuleList(
            Qwen3_5DecoderLayer(
                vllm_config,
                layer_type="full_attention",
                prefix=f"{prefix}.layers.{idx}",
            )
            for idx in range(self.num_mtp_layers)
        )
        vllm_config.quant_config = original_quant
        self.is_fused_shared_expert_enabled = is_model_fused_shared_expert_compatible(
            self.layers,
            Qwen3NextSparseMoeBlock,
            "mlp",
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is None:
                inputs_embeds = self.embed_input_ids(input_ids)
            assert hidden_states.shape[-1] == inputs_embeds.shape[-1]
            inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
            hidden_states = self.pre_fc_norm_hidden(hidden_states)
            hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
            hidden_states = self.fc(hidden_states)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[current_step_idx]
        if mtp_layer.use_attn_reduce_scatter_for_moe:
            assert hidden_states.shape[0] == positions.shape[-1]
            hidden_states = sequence_parallel_chunk(hidden_states)
            assert residual is None
        hidden_states, residual = mtp_layer(
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
        )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        if mtp_layer.use_attn_reduce_scatter_for_moe:
            hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
            hidden_states = hidden_states[: positions.shape[-1]]
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = maybe_fuse_shared_experts(
            weights,
            enabled=self.is_fused_shared_expert_enabled,
            n_routed_experts=getattr(self.config, "num_experts", 0),
            n_shared_experts=1,
            ckpt_prefix="mlp.shared_expert",
        )
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "hidden_states": 0,
    }
)
class Qwen3_5MTP(LocalArgmaxMixin, nn.Module, SupportsMultiModal):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        cache_config = vllm_config.cache_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen3_5MTP currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )

        # [P110 Qwen3 MTP GPTQ/AWQ fix v7.63.x guard-v4]
        # GPTQ/AWQ: MTP decoder layers use unquantized bf16 weights.
        bypass_mtp_quant = False
        if vllm_config.quant_config and vllm_config.quant_config.get_name() not in ("modelopt_fp4",):
            hf_qc = getattr(vllm_config.model_config.hf_config, "quantization_config", None)
            if isinstance(hf_qc, dict):
                dynamic = hf_qc.get("dynamic", {})
                if any(k.startswith("-:") and "mtp" in k for k in dynamic):
                    bypass_mtp_quant = True
            if not bypass_mtp_quant and vllm_config.quant_config.get_name() in (
                "gptq", "auto_gptq", "gptq_marlin", "awq", "awq_marlin",
                "compressed-tensors", "compressed_tensors", "marlin", "compressed_tensors_wNa16",
            ):
                bypass_mtp_quant = True
        self.quant_config = None if bypass_mtp_quant else vllm_config.quant_config

        super().__init__()
        self.config = config
        self.model = Qwen3_5MultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "mtp")
        )

        if get_pp_group().is_last_rank and self.model.share_backbone_input_output:
            # [PN348] target owns lm_head — skip allocation.
            self.lm_head = PPMissingLayer()
        elif get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.model.embed_input_ids,
            is_multimodal=is_multimodal,
        )

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        is_multimodal = _require_is_multimodal(is_multimodal)

        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        hidden_states = self.model(
            input_ids, positions, hidden_states, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def remap_weight_names(weights):
            for name, weight in weights:
                if name.startswith("mtp."):
                    name = name.replace("mtp.", "model.")
                elif any(key in name for key in ["embed_tokens", "lm_head"]):
                    # [PN348] target owns the weight — skip duplicate load.
                    if self.model.share_backbone_input_output:
                        continue
                    if "embed_tokens" in name:
                        name = name.replace("language_model.", "")
                else:
                    continue
                yield name, weight

        # [P112 MTP Quant Disk Cache v1.1]
        _isv_p112_cache_enabled = (
            False
        )
        _isv_p112_remapped = remap_weight_names(weights)
        if _isv_p112_cache_enabled:
            try:
                from vllm.idiotsavant.mtp_cache import MTPQuantDiskCacheManager
                _isv_p112_model_name = "qwen3_mtp"
                if hasattr(self, "vllm_config") and hasattr(self.vllm_config, "model_config"):
                    _isv_p112_model_name = getattr(self.vllm_config.model_config, "model", "qwen3_mtp")
                _isv_p112_tp_size = 1
                if hasattr(self, "vllm_config") and hasattr(self.vllm_config, "parallel_config"):
                    _isv_p112_tp_size = getattr(self.vllm_config.parallel_config, "tensor_parallel_size", 1)
                _isv_p112_quant_fmt = "int8"
                _isv_p112_cache_mgr = MTPQuantDiskCacheManager(
                    enabled=True, quant_format=_isv_p112_quant_fmt
                )
                _isv_p112_remapped = _isv_p112_cache_mgr.process_mtp_weights(
                    model_name=str(_isv_p112_model_name),
                    source_file=str(_isv_p112_model_name),
                    tp_size=_isv_p112_tp_size,
                    weights=_isv_p112_remapped,
                )
            except Exception as _p112_exc:
                import logging as _isv_logging
                _isv_logging.getLogger("idiotsavant.p112").warning("[P112] MTP cache hook fallback: %s", _p112_exc)
                _isv_p112_remapped = remap_weight_names(weights)

        loader = AutoWeightsLoader(self)
        return loader.load_weights(_isv_p112_remapped)


class Qwen3_5MoeMTP(Qwen3_5MTP, QwenNextMixtureOfExperts):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.set_moe_parameters()
