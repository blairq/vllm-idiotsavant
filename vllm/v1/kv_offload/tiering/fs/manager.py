# [Genesis wiring marker: [Genesis PN90 kv disk write gating & tiered demotion]]
# [Genesis wiring marker: [Genesis PN81 kv disk tier quota]]
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
FileSystemTierManager: Pure-Python file system secondary tier for KV cache offloading.

Store path:
    Data is written to a temp file (<dest_path.tmp>) via os.write,
    then os.replace'd to the final path (without .tmp).

Load path:
    Data is read from the block file directly via os.readv into the
    provided memoryview slice.

File naming:  <base_path>_r<rank>/<hhh>/<hh>_g<group_idx>/<hash_hex>.bin
              (hash-based subdirectories to limit directory fan-out)
"""

import functools
import json
import os
from collections.abc import Iterable
from typing import TYPE_CHECKING, ClassVar

try:
    from vllm.fs_io_C import batch_lookup as batch_lookup_C

    _HAS_BATCH_LOOKUP_C = True
except ImportError:
    _HAS_BATCH_LOOKUP_C = False

from typing_extensions import override

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    Locality,
    LookupResult,
    Medium,
    OffloadingEvent,
    OffloadKey,
    ReqContext,
)
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.async_lookup import AsyncLookupManager
from vllm.v1.kv_offload.tiering.base import (
    JobId,
    JobResult,
    RequestOffloadingContext,
    ScheduleEndContext,
    SecondaryTierManager,
    TransferJob,
)
from vllm.v1.kv_offload.tiering.fs.io import (
    batch_load_block,
    batch_store_block,
    probe_o_direct,
)
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec

logger = init_logger(__name__)


class FsAsyncLookupManager(AsyncLookupManager):
    """Async lookup manager for FileSystemTierManager."""

    def __init__(
        self,
        tier: "FileSystemTierManager",
        tier_type: str,
    ) -> None:
        super().__init__(tier_type=tier_type)
        self._tier = tier

    def batch_lookup(
        self, keys: list[OffloadKey], req_context: ReqContext
    ) -> Iterable[bool]:
        paths = [self._tier.file_mapper.get_file_name(k) for k in keys]
        if _HAS_BATCH_LOOKUP_C:
            # C extension: GIL released for the entire faccessat() batch.
            return batch_lookup_C(paths)
        return (os.path.exists(p) for p in paths)


class FileSystemTierManager(SecondaryTierManager):
    """
    Pure-Python disk-backed secondary tier.

    Read-priority threads service load jobs preferentially; write-priority
    threads service store jobs preferentially.  Both groups can drain either
    queue, so neither starves.

    submit_store / submit_load are non-blocking: they enqueue tasks and return.
    get_finished_jobs() polls job completion and returns completed JobResults.

    Cross-process sharing:
        KV cache sharing between multiple vLLM instances using the same
        ``root_dir`` (e.g., via a shared PVC) works by default: ``NONE_HASH``
        (the chain-hash seed for block content hashes) is derived from a fixed
        default seed, so identical token content produces identical block
        filenames across instances. Setting the ``PYTHONHASHSEED`` environment
        variable to the same value on all instances overrides the default seed,
        and is required to share a cache when using a non-cryptographic
        prefix-caching hash algorithm, which seeds ``NONE_HASH`` randomly.
    """

    medium: ClassVar[Medium] = Medium.STORAGE

    def __init__(
        self,
        offloading_spec: "OffloadingSpec",
        primary_kv_view: memoryview,
        tier_type: str,
        root_dir: str,
        n_read_threads: int = 16,
        n_write_threads: int = 16,
        enable_kv_events: bool = False,
        locality: str | None = None,
    ):
        """
        Args:
            offloading_spec: Contains normalized offloading configuration and
                blocks_per_chunk.
            primary_kv_view: Memoryview of the primary tier's CPU KV cache.
            tier_type: Tier type identifier, set by SecondaryTierFactory.
            root_dir: Root directory for block files.
            n_read_threads: Number of read-priority I/O threads.
            n_write_threads: Number of write-priority I/O threads.
            enable_kv_events: Emit BlockStored KV events for blocks
                successfully stored to this tier. Effective only when KV
                cache events are enabled globally (kv_events_config).
            locality: Whether this tier's storage is LOCAL or REMOTE relative
                to the publishing vLLM instance.
        """
        super().__init__(offloading_spec, primary_kv_view, tier_type)
        self.locality = Locality(locality) if locality is not None else None

        self.events: list[OffloadingEvent] | None = None
        if enable_kv_events:
            if offloading_spec.kv_events_config.enable_kv_cache_events:
                self.events = []
            else:
                logger.warning(
                    "enable_kv_events is set on secondary tier '%s' but KV "
                    "cache events are disabled globally; the tier will not "
                    "emit events.",
                    tier_type,
                )
        # Keys of in-flight store jobs, tracked only when events are enabled.
        self._store_job_keys: dict[JobId, list[OffloadKey]] = {}
        # Keys of in-flight load (promotion) jobs, so a failed load can mark
        # its own cached lookup verdicts False (see get_finished_jobs).
        self._load_job_keys: dict[JobId, list[OffloadKey]] = {}
        # Per load job: how many blocks loaded before a failure (partial keep).
        # Written by the pool worker inside the load task before it raises (so
        # before task_done publishes the job); read on the scheduler thread in
        # get_finished_jobs only for job ids the finished queue returned. Under
        # the GIL that read cannot observe the finished job without the prior
        # write, so no extra lock is needed (get_finished is itself lock-free).
        self._load_progress: dict[JobId, int] = {}

        # Extract block size from primary view
        assert primary_kv_view.strides is not None, (
            "primary_kv_view.strides cannot be None"
        )
        self._block_size: int = primary_kv_view.strides[0]

        # Opt in; FileMapper enables it only for a parallelism-invariant block.
        self.file_mapper = FileMapper.from_offloading_spec(
            root_dir=root_dir,
            offloading_spec=offloading_spec,
            blocks_per_file=offloading_spec.blocks_per_chunk,
            parallel_agnostic=True,
        )

        # Write config file
        config_path = self.file_mapper.get_config_file_path()
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                json.dump(
                    self.file_mapper.get_run_config(), f, indent=2, sort_keys=True
                )

        # Prefer O_DIRECT to bypass the page cache, but fall back to buffered
        # I/O on filesystems that reject it (e.g. overlayfs, some NFS mounts)
        # rather than failing every block.
        self._use_o_direct = probe_o_direct(os.path.dirname(config_path))
        if not self._use_o_direct:
            logger.warning(
                "O_DIRECT is not supported at '%s'; falling back to buffered "
                "I/O for the '%s' KV offload tier.",
                root_dir,
                tier_type,
            )

        self._pool = DualQueueThreadPool(
            n_read_threads,
            n_write_threads,
            thread_name_prefix="vllm_kv_py_fs",
        )

        self._lookup_manager = FsAsyncLookupManager(tier=self, tier_type=self.tier_type)

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        result = self._lookup_manager.lookup(key, req_context)
        if result is None:
            return LookupResult.RETRY
        return LookupResult.HIT if result else LookupResult.MISS

    @override
    def submit_store(self, job_metadata: TransferJob) -> None:
        keys = list(job_metadata.keys)
        if self.events is not None:
            self._store_job_keys[job_metadata.job_id] = keys
        task = functools.partial(
            batch_store_block,
            [self.file_mapper.get_file_name(key) for key in keys],
            self._primary_kv_view,
            [int(bid) * self._block_size for bid in job_metadata.block_ids],
            self._block_size,
            self._use_o_direct,
        )
        self._pool.enqueue_store(job_metadata.job_id, 1, [task])

    @override
    def submit_load(self, job_metadata: TransferJob) -> None:
        job_id = job_metadata.job_id
        # Track this load's keys so a failed promotion can mark only its failed
        # keys as a miss (see get_finished_jobs).
        keys = list(job_metadata.keys)
        self._load_job_keys[job_id] = keys
        paths = [self.file_mapper.get_file_name(key) for key in keys]
        offsets = [int(bid) * self._block_size for bid in job_metadata.block_ids]

        def load_task() -> None:
            try:
                batch_load_block(
                    paths,
                    self._primary_kv_view,
                    offsets,
                    self._block_size,
                    self._use_o_direct,
                )
            except OSError as exc:
                # Runs on the pool worker thread. Record how many blocks loaded
                # before the failure so get_finished_jobs can keep them; this
                # write precedes task_done, so the scheduler reads it safely
                # under the GIL once the finished queue hands back this job.
                num_succeeded = getattr(exc, "num_succeeded", 0)
                self._load_progress[job_id] = num_succeeded
                # Surfaces errno (e.g. EMFILE "Too many open files") for both
                # the C and Python load paths.
                logger.debug(
                    "Load of %d blocks for job %s failed at block %d: %s",
                    len(paths),
                    job_id,
                    num_succeeded,
                    exc,
                )
                raise

        self._pool.enqueue_load(job_id, 1, [load_task])

    # [Genesis PN90 kv disk write gating & tiered demotion]
    def submit_demote(
        self,
        key: OffloadKey,
        snapshot: bytes,
        req_context: ReqContext | None = None,
    ) -> None:
        """Baja a disco un bloque ya desalojado de L2, desde una copia suelta.

        A diferencia de submit_store, el dato NO vive mas en el primary tier:
        el slot ya fue reutilizado. Por eso se escribe desde `snapshot`.
        """
        from vllm._genesis import kv_disk_gate as _g90
        dest_path = self.file_mapper.get_file_name(key)
        # El bloque ya esta en el SSD: reescribirlo es amplificacion de
        # escritura pura. GDN es ~75% del trafico de este tier.
        if os.path.exists(dest_path):
            return
        buf = memoryview(snapshot)
        from vllm.v1.kv_offload.tiering.fs.io import _store_block as store_block
        task = functools.partial(store_block, dest_path, buf, 0, len(buf))
        _job_id = _g90.next_demote_job_id()
        from vllm._genesis import kv_tier_metrics as _g88
        if _g88.enabled():
            try:
                _params = getattr(req_context, 'kv_transfer_params', None)
                _g88.note_demote(self, _job_id, len(buf), _g88.agent_label(_params))
            except Exception:
                pass
        self._pool.enqueue_store(_job_id, 1, [task])

    @override
    def get_finished_jobs(self) -> Iterable[JobResult]:
        """Collect finished jobs; a failed promotion marks only its failed keys
        as a miss here (scheduler thread)."""
        results = []
        for job_id, success, transfer_time in self._pool.get_finished():
            if self.events is not None:
                keys = self._store_job_keys.pop(job_id, None)
                if success and keys:
                    self.events.append(
                        OffloadingEvent(
                            keys=keys,
                            medium=self.medium,
                            removed=False,
                            locality=self.locality,
                        )
                    )
            load_keys = self._load_job_keys.pop(job_id, None)
            num_succeeded = self._load_progress.pop(job_id, 0)
            if load_keys is not None and not success:
                # A batched load stops at the first bad block and reports how
                # many loaded before it. Those earlier blocks are kept in the
                # primary tier (reported via successful_keys); only this block
                # and the ones after it are marked a miss and recomputed.
                successful = load_keys[:num_succeeded]
                failed = load_keys[num_succeeded:]
                self._lookup_manager.mark_miss(failed)
                results.append(
                    JobResult(
                        job_id=job_id,
                        success=False,
                        successful_keys=tuple(successful) if successful else None,
                        transfer_time=transfer_time,
                    )
                )
                continue
            results.append(
                JobResult(
                    job_id=job_id,
                    success=success,
                    transfer_time=transfer_time,
                )
            )
        return results

    @override
    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()

    @override
    def drain_jobs(self) -> None:
        """Block until all in-flight transfers in the threadpool finish."""
        self._pool.wait_idle()

    def on_request_finished(self, req_context: ReqContext) -> None:
        self._lookup_manager.cleanup(req_context.req_id)

    # [Genesis PN81 kv disk tier quota]
    # Extiende el on_schedule_end de upstream (que solo hace flush del
    # lookup manager) con la cuota de disco. Sin esto root_dir crece sin
    # techo: el tier fs no tiene poda propia.
    # Ver wiring/hybrid/patch_N81_kv_disk_tier_quota.py
    @override
    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        self._lookup_manager.flush()
        import os as _g81_os
        if _g81_os.environ.get('GENESIS_ENABLE_PN81_KV_DISK_QUOTA') != '1':
            return
        try:
            # GATE POR TIEMPO, no por pasos del scheduler.
            # El gate viejo era `paso % GENESIS_KV_DISK_CHECK_EVERY`, y
            # on_schedule_end sube un paso por iteracion del scheduler: si el
            # engine esta ocioso, los pasos no se acumulan y la cuota NUNCA
            # corre. Medido: 22 min sirviendo 61 requests cortos no alcanzaron
            # los 2000 pasos, y el directorio llego a 31,4 GiB con limite de 30
            # sin una sola poda. Con reloj, la cuota corre haya o no trafico.
            import time as _g81_time
            _secs = float(_g81_os.environ.get('GENESIS_KV_DISK_CHECK_SECS', '60'))
            _now = _g81_time.monotonic()
            _last = getattr(self, '_genesis_pn81_last', 0.0)
            if _now - _last < _secs:
                return
            self._genesis_pn81_last = _now

            def _do_pn81_quota():
                try:
                    _max = float(_g81_os.environ.get('GENESIS_KV_DISK_MAX_GB', '30'))
                    _ratio = float(_g81_os.environ.get(
                        'GENESIS_KV_DISK_TARGET_RATIO', '0.85'))
                    # El directorio real NO es base_path: FileMapper escribe en
                    # f'{base_path}_r{rank}' (ver get_file_name). Cada rank poda el
                    # SUYO, asi que la cuota es POR RANK: con TP=2 el total en disco
                    # es 2 x GENESIS_KV_DISK_MAX_GB.
                    _bp = getattr(self.file_mapper, 'base_path', None)
                    _rk = getattr(self.file_mapper, 'rank', 0)
                    _root = f'{_bp}_r{_rk}' if _bp else None
                    if not _root or not _g81_os.path.isdir(_root):
                        return
                    # PURGA DE DIRECTORIOS ABANDONADOS. La cuota de arriba solo mira
                    # el directorio del modelo ACTUAL. Cada modelo que se prueba deja
                    # el suyo, y nadie los borra nunca: medido, 54 GiB de un modelo que
                    # ya ni estaba descargado, y 142 GiB en total en /kv-offload.
                    # Se borran por ANTIGUEDAD (no por 'no es el mio'): un modelo que
                    # usaste ayer conserva su cache, que es justamente para lo que
                    # existe el tier de disco.
                    #
                    # PERIODICA, NO UNA SOLA VEZ (2026-08-21). El gate viejo era un
                    # booleano `_genesis_pn81_orphans` que se prendia en la primera
                    # pasada y no se apagaba nunca: la purga corria al arrancar y
                    # listo. Un directorio que queda huerfano DESPUES del arranque no
                    # se revisaba jamas mientras el engine siguiera vivo.
                    # Medido: al arrancar el 08-19, orcarouter_..._3ef784eef730_r0
                    # llevaba 1,6 dias sin uso -- por debajo de los 3 del umbral, asi
                    # que se salteo con razon. Dos dias despues ya calificaba y sus
                    # 25,47 GiB seguian ahi, porque la purga no volvia a correr.
                    # Despistaba que su hermano sin datos (solo config.json, 0 GiB) SI
                    # se borro: nadie le escribe despues de crearlo, asi que su mtime
                    # ya tenia 3,2 dias. Se borro la cascara y quedo el contenido.
                    # Ahora corre cada GENESIS_KV_DISK_ORPHAN_CHECK_SECS (default una
                    # hora). Cadencia propia y no la de la cuota (60s) porque este
                    # barrido recorre TODOS los directorios de /kv-offload y no solo
                    # el del modelo actual: son miles de stat() en vez de cientos.
                    # La primera pasada sigue siendo al arranque (_olast is None).
                    _dias = float(_g81_os.environ.get('GENESIS_KV_DISK_ORPHAN_DAYS', '0'))
                    _osecs = float(_g81_os.environ.get(
                        'GENESIS_KV_DISK_ORPHAN_CHECK_SECS', '3600'))
                    _olast = getattr(self, '_genesis_pn81_orphan_last', None)
                    if _dias > 0 and (_olast is None or _now - _olast >= _osecs):
                        self._genesis_pn81_orphan_last = _now
                        import sys as _g81_s3
                        _padre = _g81_os.path.dirname(_root.rstrip('/'))
                        _mio = _g81_os.path.basename(_root.rstrip('/'))
                        _corte = _g81_time.time() - _dias * 86400
                        for _d in sorted(_g81_os.listdir(_padre)):
                            if _d == _mio or _d.startswith(_mio.rsplit('_r', 1)[0]):
                                continue
                            _dp2 = _g81_os.path.join(_padre, _d)
                            if not _g81_os.path.isdir(_dp2):
                                continue
                            _reciente = 0.0
                            _peso = 0
                            for _r2, _dn2, _fn2 in _g81_os.walk(_dp2):
                                for _f2 in _fn2:
                                    try:
                                        _st2 = _g81_os.stat(_g81_os.path.join(_r2, _f2))
                                    except OSError:
                                        continue
                                    _reciente = max(_reciente, _st2.st_mtime)
                                    _peso += _st2.st_size
                            if _peso and _reciente < _corte:
                                import shutil as _g81_sh
                                try:
                                    _g81_sh.rmtree(_dp2)
                                    # PN88: sacar el desalojo a metricas. Best-effort
                                    # y con import local: si PN88 no esta, no pasa nada.
                                    try:
                                        from vllm._genesis import kv_tier_metrics as _g88
                                        _g88.note_eviction('disk', 'orphan', 1, _peso)
                                    except Exception:
                                        pass
                                    print('[PN81] cache abandonada borrada: %s '
                                          '(%.2f GiB, sin usar hace %.1f dias)'
                                          % (_d, _peso / (1 << 30),
                                             (_g81_time.time() - _reciente) / 86400.0),
                                          file=_g81_s3.stderr, flush=True)
                                except OSError as _e2:
                                    print('[PN81] no pude borrar %s: %r' % (_d, _e2),
                                          file=_g81_s3.stderr, flush=True)

                    _limit = int(_max * (1 << 30))
                    _files = []
                    _total = 0
                    for _dp, _dn, _fn in _g81_os.walk(_root):
                        for _f in _fn:
                            _p = _g81_os.path.join(_dp, _f)
                            try:
                                _st = _g81_os.stat(_p)
                            except OSError:
                                continue
                            _files.append((_st.st_mtime, _st.st_size, _p))
                            _total += _st.st_size
                    if not getattr(self, '_genesis_pn81_visto', False):
                        self._genesis_pn81_visto = True
                        import sys as _g81_s0
                        print('[PN81] cuota activa sobre %s: %.2f GiB de %.2f GiB '
                              '(chequeo cada %.0fs)'
                              % (_root, _total / (1 << 30), _max, _secs),
                              file=_g81_s0.stderr, flush=True)
                    # PN88: ocupacion del tier como gauge. Va ANTES del early return
                    # para que se publique en cada chequeo, se pode o no.
                    try:
                        from vllm._genesis import kv_tier_metrics as _g88
                        _g88.set_occupancy('disk', _total, _limit)
                    except Exception:
                        pass
                    if _total <= _limit:
                        return
                    # podar del mas viejo al mas nuevo hasta el ratio objetivo
                    _target = int(_limit * _ratio)
                    _files.sort()
                    _freed = 0
                    _n = 0
                    for _mt, _sz, _p in _files:
                        if _total - _freed <= _target:
                            break
                        try:
                            _g81_os.remove(_p)
                            _freed += _sz
                            _n += 1
                        except OSError:
                            continue
                    try:
                        from vllm._genesis import kv_tier_metrics as _g88
                        _g88.note_eviction('disk', 'quota', _n, _freed)
                        _g88.set_occupancy('disk', _total - _freed, _limit)
                    except Exception:
                        pass
                    import sys as _g81_sys
                    print('[PN81] cuota del tier de disco: %.2f GiB > limite %.2f GiB '
                          '-> borrados %d bloques mas viejos, liberados %.2f GiB '
                          '(queda %.2f GiB)'
                          % (_total / (1 << 30), _max, _n, _freed / (1 << 30),
                             (_total - _freed) / (1 << 30)),
                          file=_g81_sys.stderr, flush=True)
                finally:
                    self._genesis_pn81_running = False

            if _g81_os.environ.get('GENESIS_PN81_ASYNC', '1') == '1':
                if getattr(self, '_genesis_pn81_running', False):
                    return
                self._genesis_pn81_running = True
                import threading as _g81_th
                _t = _g81_th.Thread(
                    target=_do_pn81_quota,
                    name='pn81_quota',
                    daemon=True,
                )
                self._genesis_pn81_worker = _t
                _t.start()
            else:
                _do_pn81_quota()
        except Exception as _g81_err:
            # la poda NUNCA debe tumbar el scheduler; se avisa una sola vez
            if not getattr(self, '_genesis_pn81_failed', False):
                self._genesis_pn81_failed = True
                import sys as _g81_s2
                print('[PN81] la poda fallo y se desactiva (el engine sigue '
                      'normal): %r' % (_g81_err,), file=_g81_s2.stderr, flush=True)

    @override
    def shutdown(self) -> None:
        """
        Release resources held by this tier.

        Shuts down the lookup manager and the thread pool,
        clearing pending tasks and waiting for active threads to complete.
        """
        self._lookup_manager.shutdown()
        self._pool.shutdown(wait=True)
