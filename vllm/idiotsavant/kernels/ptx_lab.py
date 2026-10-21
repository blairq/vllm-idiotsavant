# SPDX-License-Identifier: Apache-2.0
"""Laboratorio PTX: compila, ensambla, carga y lanza cualquier kernel ``.cu``.

Es el mismo camino que SK-12 (``nvcc -ptx`` -> ``ptxas`` -> ``cuModuleLoadData``
-> ``cuLaunchKernel``) pero parametrizado por archivo, entrada y defines, para
los experimentos de tensor cores (mma s4, dispersos, atencion INT8...).

Uso::

    k = Kernel("lab_mma.cu", "lab_mma_s8", defs=["-DREPS=64"], warps=1)
    k.lanzar((gx, gy), [tensor, tensor, 3, ...], shared=0)
    print(k.ptx())          # PTX generado, para leerlo o editarlo a mano
    print(k.recursos())     # registros, derrame y shared segun ptxas -v (no hace falta GPU)

Perfilado: con ``VLLM_IDIOTSAVANT_PTX_LINEINFO=1`` se compila con ``-lineinfo`` para que Nsight
Compute atribuya las metricas a la linea de fuente. Entra en la clave de cache, asi que prenderlo
genera PTX nuevo y apagarlo vuelve al de siempre.
"""

from __future__ import annotations

import ctypes
import hashlib
import re
import os
import pathlib
import subprocess
import tempfile
import threading

import torch

ARCH = 86
LINEINFO = os.environ.get("VLLM_IDIOTSAVANT_PTX_LINEINFO", "") == "1"
_ATRIB_MAX_SHARED = 8
_lock = threading.RLock()
_libcuda = None


def libcuda():
    global _libcuda
    if _libcuda is None:
        _libcuda = ctypes.CDLL("libcuda.so.1")
    return _libcuda


def _dir(nombre: str) -> pathlib.Path:
    d = pathlib.Path(os.path.expanduser(f"~/.cache/vllm/idiotsavant/ptx/{nombre}"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def ruta_cu(archivo: str) -> pathlib.Path:
    p = pathlib.Path(archivo)
    if p.is_file():
        return p
    p = pathlib.Path(__file__).resolve().parent / "cuda" / archivo
    if p.is_file():
        return p
    raise FileNotFoundError(archivo)


class Kernel:
    def __init__(self, archivo: str, entrada: str, defs=(), warps: int = 8,
                 opt: int = 3, arch: int = ARCH):
        self.src = ruta_cu(archivo)
        self.entrada = entrada
        self.defs = list(defs)
        self.warps = warps
        self.opt = opt
        self.arch = arch
        self._fn = None
        self._mod = None
        self._shared_attr = None

    def _clave(self, texto: str) -> str:
        texto += ""
        return hashlib.sha256((texto + f"|{self.entrada}|sm_{self.arch}|O{self.opt}|"
                               + ("lineinfo|" if LINEINFO else "")
                               + " ".join(self.defs)).encode()).hexdigest()[:16]

    def ptx(self, forzar: bool = False) -> str:
        texto = self.src.read_text()
        # La clave incluye los headers locales (#include "x.cuh"): sin esto un cambio en un
        # header comun no recompilaba y se usaba el PTX viejo en silencio.
        for inc in re.findall(r'#include\s+"([^"]+)"', texto):
            h = self.src.parent / inc
            if h.is_file():
                texto += "\n//@@" + inc + "\n" + h.read_text()
        dest = _dir(self.src.stem) / f"{self._clave(texto)}.ptx"
        if dest.is_file() and not forzar:
            return dest.read_text()
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "k.ptx")
            # Con LINEINFO se agrega -lineinfo para que ncu pueda atribuir las metricas a la linea de
            # fuente. Entra en la clave de cache porque cambia el binario.
            extra = ["-lineinfo"] if LINEINFO else []
            r = subprocess.run(["nvcc", "-ptx", f"-arch=sm_{self.arch}", f"-O{self.opt}",
                                "--use_fast_math", *extra, *self.defs, str(self.src), "-o", out],
                               capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"nvcc fallo ({self.src.name}):\n{r.stderr}")
            txt = pathlib.Path(out).read_text()
        dest.write_text(txt)
        return txt

    def _ensamblar(self, ptx: str, verbose: bool = False):
        with tempfile.TemporaryDirectory() as d:
            fp, fc = os.path.join(d, "k.ptx"), os.path.join(d, "k.cubin")
            pathlib.Path(fp).write_text(ptx)
            extra = ["-lineinfo"] if LINEINFO else []
            if verbose:
                extra.append("-v")
            r = subprocess.run(["ptxas", f"-arch=sm_{self.arch}", f"-O{self.opt}", *extra,
                                fp, "-o", fc],
                               capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"ptxas fallo ({self.src.name}):\n{r.stderr}")
            cubin = pathlib.Path(fc).read_bytes()
        return (cubin, r.stdout + r.stderr) if verbose else cubin

    def recursos(self) -> dict:
        """Lo que ``ptxas -v`` reporta para la entrada: registros, derrame, stack y shared estatica.

        Solo compila (nvcc + ptxas), asi que corre sin GPU. Con 255 registros o derrame > 0 conviene
        mirar el kernel antes de tocar su lanzamiento: en sm_86 el banco de 64K registros por SM
        limita los bloques residentes tanto como la shared.
        """
        _, salida = self._ensamblar(self.ptx(), verbose=True)
        res = dict(registros=0, derrame_st=0, derrame_ld=0, stack=0, smem=0)
        actual = None
        for linea in salida.splitlines():
            m = re.search(r"Compiling entry function '([^']+)'", linea)
            if m:
                actual = m.group(1)
                continue
            if actual != self.entrada:
                continue
            for clave, patron in (("stack", r"(\d+) bytes stack frame"),
                                  ("derrame_st", r"(\d+) bytes spill stores"),
                                  ("derrame_ld", r"(\d+) bytes spill loads"),
                                  ("registros", r"Used (\d+) registers"),
                                  ("smem", r"(\d+) bytes smem")):
                m = re.search(patron, linea)
                if m:
                    res[clave] = int(m.group(1))
        return res

    @torch.compiler.disable
    def cargar(self, ptx_texto: str | None = None):
        with _lock:
            if self._fn is not None:
                return
            cubin = self._ensamblar(ptx_texto if ptx_texto is not None else self.ptx())
            img = (ctypes.c_char * len(cubin)).from_buffer_copy(cubin)
            mod = ctypes.c_void_p()
            r = libcuda().cuModuleLoadData(ctypes.byref(mod), img)
            if r != 0:
                raise RuntimeError(f"cuModuleLoadData: {r}")
            fn = ctypes.c_void_p()
            r = libcuda().cuModuleGetFunction(ctypes.byref(fn), mod, self.entrada.encode())
            if r != 0:
                raise RuntimeError(f"cuModuleGetFunction({self.entrada}): {r}")
            self._mod, self._fn = mod, fn

    def lanzar(self, grid, args, shared: int = 0, sync: bool = False):
        self.cargar()
        if shared > 48 * 1024 and self._shared_attr != shared:
            r = libcuda().cuFuncSetAttribute(self._fn, _ATRIB_MAX_SHARED, ctypes.c_int(shared))
            if r != 0:
                raise RuntimeError(f"cuFuncSetAttribute(shared={shared}): {r}")
            self._shared_attr = shared
        vals = []
        for v in args:
            if isinstance(v, torch.Tensor):
                vals.append(ctypes.c_uint64(v.data_ptr()))
            elif isinstance(v, float):
                vals.append(ctypes.c_float(v))
            elif isinstance(v, ctypes._SimpleCData):
                vals.append(v)          # puntero crudo o tipo elegido a mano, va tal cual
            else:
                vals.append(ctypes.c_int32(int(v)))
        arr = (ctypes.c_void_p * len(vals))(
            *[ctypes.cast(ctypes.byref(x), ctypes.c_void_p) for x in vals])
        gx, gy, gz = (*grid, 1) if len(grid) == 2 else grid
        r = libcuda().cuLaunchKernel(self._fn, gx, gy, gz, self.warps * 32, 1, 1,
                                     ctypes.c_uint(shared),
                                     ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
                                     arr, None)
        if r != 0:
            raise RuntimeError(f"cuLaunchKernel({self.entrada}): {r}")
        if sync:
            torch.cuda.synchronize()


def contar_instrucciones(ptx: str, funcion: str | None = None) -> dict:
    """Histograma de mnemonicos del PTX (primer token de cada linea con tab)."""
    import collections
    c = collections.Counter()
    dentro = funcion is None
    for linea in ptx.splitlines():
        if funcion and linea.startswith("function") and funcion in linea:
            dentro = True
        elif funcion and linea.startswith("function"):
            dentro = False
        if dentro and linea.startswith("\t") and not linea.startswith("\t;"):
            tok = linea.strip().split(" ")[0]
            if tok and not tok.endswith(":"):
                c[tok.split(".")[0]] += 1
    return dict(c.most_common())
