// SPDX-License-Identifier: Apache-2.0
//
// SK-18h/decuant — paginas de la KV de PN131 -> cache fp16 [n, 2, BS, NH, 256] para el prefill
// (FlashInfer paginado), TODO ENTERO: valor = x8 * sf * 2^e / 32767 -> bits fp16 con clz.
//   u = |x8| * sf * 32769  (~ |x8| * sf * 2^30 / 32767),  valor = u * 2^(e - 30)
// Grilla: x = bloque de la lista, y = token de la pagina; 32 lanes = grupos de 8 dimensiones.
#include <cuda_pipeline.h>
#include "sk18h_comun.cuh"

#define QD 256
// ROTV=1: la V del pool viene ROTADA (es el espejo int8 del camino int4) y hay que des-rotarla
// con el Walsh-Hadamard entero, como hace sk18i_decuant.
#ifndef ROTV
#define ROTV 0
#endif

extern "C" __global__ void __launch_bounds__(32)
sk18h_decuant(
    const signed char* __restrict__ pool,
    const long long* __restrict__ ids,       // [n] bloques fisicos
    const int* __restrict__ refs,            // [2] ek, ev
    const int* __restrict__ signos,          // [256] +-1 (solo con ROTV)
    unsigned short* __restrict__ out,        // [n, 2, BS, NH, 256] bits fp16
    int BLK, int BS, int NH)
{
    const int i = blockIdx.x;
    const int o = blockIdx.y;
    const int lane = threadIdx.x;
    if (ids[i] < 0) return;                  // esta pagina la hace el otro kernel
    const size_t base = (size_t)ids[i] * (size_t)BLK;
    const size_t KOFF = (size_t)BS * NH * QD;
    const int ek = refs[0], ev = refs[1];
#pragma unroll
    for (int h = 0; h < NH; ++h) {
        const unsigned char* e8 = (const unsigned char*)pool + base + 2 * KOFF + ((size_t)o * NH + h) * 4;
        const int skf = (int)(short)(e8[0] | (e8[1] << 8));
        const int svf = (int)(short)(e8[2] | (e8[3] << 8));
        const size_t ok_ = (((size_t)i * 2 + 0) * BS + o) * NH * QD + (size_t)h * QD;
        const size_t ov_ = (((size_t)i * 2 + 1) * BS + o) * NH * QD + (size_t)h * QD;
        int xv[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int d = lane * 8 + e;
            const int kx = pool[base + ((size_t)o * NH + h) * QD + d];
            const int vx = pool[base + KOFF + ((size_t)h * QD + d) * BS + o];
            const unsigned long long uk = (unsigned long long)(kx < 0 ? -kx : kx) * (unsigned long long)skf * 32769ull;
            out[ok_ + d] = (unsigned short)q_a_fp16(uk, ek - 30, kx < 0);
            if (ROTV) {
                xv[e] = vx * (int)svf;
            } else {
                const unsigned long long uv = (unsigned long long)(vx < 0 ? -vx : vx) * (unsigned long long)svf * 32769ull;
                out[ov_ + d] = (unsigned short)q_a_fp16(uv, ev - 30, vx < 0);
            }
        }
        if (ROTV) {
            fwht_lanes(xv, lane);
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int d = lane * 8 + e;
                const int y = xv[e] * signos[d];
                const unsigned long long uv = (unsigned long long)(y < 0 ? -(long long)y : (long long)y) * 32769ull;
                out[ov_ + d] = (unsigned short)q_a_fp16(uv, ev - 34, y < 0);
            }
        }
    }
}

// SK-18h/decuant_tile — lo mismo que sk18h_decuant (bit a bit), por tiles de TT tokens de una pagina y
// 8 warps por bloque. El de arriba usa bloques de 1 warp (sm_86: 16 bloques por SM = 33% de ocupacion)
// y lee V traspuesta byte a byte: para un token, los 256 bytes de V estan a BS bytes uno del otro, o sea
// 256 sectores de 32 B por 256 B utiles. Aca:
//   * escalas del tile a shared (TT*NH*4 B contiguos);
//   * V [NH][256][o0..o0+TT) se lee de a 16 B (contiguo en tokens) y se traspone en shared;
//   * K [o0..o0+TT)[NH][256] es contiguo en el pool y en la salida: 16 B leidos -> 32 B escritos.
// Requiere BS % 16 == 0 (880, 832) y NH <= 2; el llamador cae al kernel de arriba si no.
#ifndef TT
#define TT 32
#endif
#define VSTR (TT + 1)            // paso de fila de V en shared (bytes): lecturas de columna con 2 vias de conflicto

__device__ __forceinline__ void dq8(const int* x, int sf, int E0, unsigned short* y) {
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const unsigned long long u = (unsigned long long)(x[e] < 0 ? -x[e] : x[e]) * (unsigned long long)sf * 32769ull;
        y[e] = (unsigned short)q_a_fp16(u, E0, x[e] < 0);
    }
}

extern "C" __global__ void __launch_bounds__(256)
sk18h_decuant_tile(
    const signed char* __restrict__ pool,
    const long long* __restrict__ ids,       // [n] bloques fisicos (< 0: lo hace otro kernel)
    const int* __restrict__ refs,            // [2] ek, ev
    const int* __restrict__ signos,          // sin uso (misma firma que sk18h_decuant)
    unsigned short* __restrict__ out,        // [n, 2, BS, NH, 256] bits fp16
    int BLK, int BS, int NH)
{
    __shared__ unsigned char sV[2 * QD * VSTR];
    __shared__ int sE[TT * 2 * 2];
    (void)signos;
    const int i = blockIdx.x, o0 = blockIdx.y * TT, tid = threadIdx.x;
    if (ids[i] < 0) return;
    const int nt = min(TT, BS - o0);
    const signed char* base = pool + (size_t)ids[i] * (size_t)BLK;
    const size_t KOFF = (size_t)BS * NH * QD;
    const int ek = refs[0], ev = refs[1];

    for (int x = tid; x < nt * NH; x += 256) {               // x = t * NH + h
        const unsigned char* e8 = (const unsigned char*)base + 2 * KOFF + ((size_t)o0 * NH + x) * 4;
        sE[2 * x] = (int)(short)(e8[0] | (e8[1] << 8));
        sE[2 * x + 1] = (int)(short)(e8[2] | (e8[3] << 8));
    }
    const int ch = nt / 16;
    for (int x = tid; x < NH * QD * ch; x += 256) {           // fila (h, d), trozo de 16 tokens
        const int fila = x / ch, c = x - fila * ch;
        const uint4 u = *reinterpret_cast<const uint4*>(base + KOFF + (size_t)fila * BS + o0 + c * 16);
        const unsigned w[4] = {u.x, u.y, u.z, u.w};
        unsigned char* dst = sV + fila * VSTR + c * 16;
#pragma unroll
        for (int b = 0; b < 16; ++b) dst[b] = (unsigned char)(w[b >> 2] >> ((b & 3) * 8));
    }
    __syncthreads();

    // K: 16 bytes -> 16 halves, todo contiguo
    unsigned short* oK = out + (((size_t)i * 2 + 0) * BS + o0) * NH * QD;
    const signed char* pK = base + (size_t)o0 * NH * QD;
    for (int x = tid; x < nt * NH * QD / 16; x += 256) {
        const int th = (x * 16) / QD;                         // t * NH + h
        const uint4 u = *reinterpret_cast<const uint4*>(pK + (size_t)x * 16);
        const unsigned w[4] = {u.x, u.y, u.z, u.w};
        int kx[16];
#pragma unroll
        for (int b = 0; b < 16; ++b) kx[b] = (int)(signed char)(w[b >> 2] >> ((b & 3) * 8));
        unsigned short y[16];
        dq8(kx, sE[2 * th], ek - 30, y);
        dq8(kx + 8, sE[2 * th], ek - 30, y + 8);
        uint4* dst = reinterpret_cast<uint4*>(oK + (size_t)x * 16);
        dst[0] = make_uint4(y[0] | (y[1] << 16), y[2] | (y[3] << 16), y[4] | (y[5] << 16), y[6] | (y[7] << 16));
        dst[1] = make_uint4(y[8] | (y[9] << 16), y[10] | (y[11] << 16), y[12] | (y[13] << 16), y[14] | (y[15] << 16));
    }
    // V: (t, h, 8 dims) desde la traspuesta en shared
    unsigned short* oV = out + (((size_t)i * 2 + 1) * BS + o0) * NH * QD;
    for (int x = tid; x < nt * NH * (QD / 8); x += 256) {
        const int th = x / (QD / 8), g = x - th * (QD / 8);   // th = t * NH + h
        const int t = th / NH, h = th - t * NH;
        int vx[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) vx[e] = (int)(signed char)sV[(h * QD + g * 8 + e) * VSTR + t];
        unsigned short y[8];
        dq8(vx, sE[2 * th + 1], ev - 30, y);
        *reinterpret_cast<uint4*>(oV + (size_t)th * QD + g * 8) =
            make_uint4(y[0] | (y[1] << 16), y[2] | (y[3] << 16), y[4] | (y[5] << 16), y[6] | (y[7] << 16));
    }
}
