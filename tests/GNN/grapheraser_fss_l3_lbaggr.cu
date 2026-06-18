
//   for epoch in 0..E-1:
//     for each mini-batch (size B) of indices over V0:
//       agg[i][c]   = Σ_s alpha[s] * post[s][i][c]              <-- FSS gpu_mul, share out
//       p[i][c]     = softmax_c(agg[i][c])                       <-- FSS gpu_softmax, share out
//       grad[i][c]  = (p - y_onehot)[i][c] / B                   <-- shares + FSS public mul
//       dalpha[s]   = Σ_{i,c} grad[i][c] * post[s][i][c]         <-- FSS gpu_mul, share out
//       Vα          = momentum * Vα - lr * (dalpha + λ·α/||α||₂) <-- shares + FSS LUT/mul
//       alpha       = ReLU(alpha + Vα)                           <-- FSS gpu_relu, share out
//   alpha <- alpha / Σ_s alpha                                   <-- FSS inverse + FSS mul
//
#include <cassert>
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "utils/gpu_data_types.h"
#include "utils/gpu_file_utils.h"
#include "utils/misc_utils.h"
#include "utils/gpu_mem.h"
#include "utils/gpu_random.h"
#include "utils/gpu_comms.h"

#include "fss/gpu_mul.h"
#include "fss/gpu_matmul.h"
#include "fss/gpu_relu.h"
#include "fss/gpu_softmax.h"
#include "fss/gpu_lut.h"
#include "fss/gpu_nexp.h"
#include "fss/gpu_inverse.h"

#include <cuda_runtime.h>

extern cudaMemPool_t mempool;

using T = u64;
static const u64 RELU_P = 0;
static const u64 RELU_Q = 0;

template <typename T>
__global__ void invSqrtFixedPoint(int N, int scaleIn, int scaleOut, u8 *tab)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        double x = double(i) / double(1LL << scaleIn);
        double eps = 1.0 / double(1LL << scaleIn);
        ((T *)tab)[i] = T((1.0 / sqrt(max(x, eps))) * double(1LL << scaleOut));
    }
}

static const std::string ROOT = []()
{
    const char *e = std::getenv("FSS_DATA_ROOT");
    return std::string(e && *e ? e : "datasets/cora_shards");
}();

static int kvInt(const std::string &l)
{
    auto eq = l.find('=');
    return eq == std::string::npos ? -1 : std::atoi(l.c_str() + eq + 1);
}
static int parseMeta(const std::string &path, const std::string &key)
{
    std::ifstream f(path);
    std::string l;
    while (std::getline(f, l))
        if (l.rfind(key + "=", 0) == 0)
            return kvInt(l);
    return -1;
}
static T *readBin(const std::string &path, size_t elems)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    assert(f.is_open());
    size_t bytes = (size_t)f.tellg();
    assert(bytes == elems * sizeof(T) && "bin size mismatch");
    f.seekg(0);
    T *b = new T[elems];
    f.read((char *)b, bytes);
    return b;
}
static void writeBin(const std::string &p, const T *b, size_t n)
{
    std::ofstream f(p, std::ios::binary);
    f.write((const char *)b, n * sizeof(T));
}
static bool fileExists(const std::string &p)
{
    std::ifstream f(p);
    return f.good();
}

static inline double u64ToFp(T x, int scale)
{
    int64_t s = (int64_t)x;
    return (double)s / (double)(1LL << scale);
}
static inline T fpToU64(double x, int scale)
{
    return (T)(int64_t)std::llround(x * (double)(1LL << scale));
}

static void ensureDir(const std::string &p) { mkdir(p.c_str(), 0755); }

static std::string sharePath(const std::string &prefix, const char *name, int party)
{
    return prefix + "_" + name + "_share" + std::to_string(party) + ".bin";
}

static int asInt(size_t n)
{
    assert(n <= (size_t)INT_MAX && "GPU helper takes int element counts");
    return (int)n;
}

static void addVecInPlace(std::vector<T> &dst, const T *src)
{
    for (size_t i = 0; i < dst.size(); ++i)
        dst[i] += src[i];
}

static T *expandBias(const T *b, int M, int out_dim)
{
    T *e = new T[(size_t)M * out_dim];
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < out_dim; ++j)
            e[(size_t)i * out_dim + j] = b[j];
    return e;
}

static T *splitMaskForParty(T *d_full_mask, int party, size_t elems, int bw)
{
    T *d_share = randomGEOnGpu<T>(elems, bw); // mask_share_0
    if (party == SERVER1)
        gpuLinearComb(bw, asInt(elems), d_share, T(1), d_full_mask, T(-1), d_share);
    return d_share;
}

static T *openMaskedInputFromCpuShare(GpuPeer *peer, const T *h_value_share,
                                      T *d_mask_share, size_t elems, int bw,
                                      Stats *s)
{
    T *d_open = (T *)moveToGPU((u8 *)h_value_share, elems * sizeof(T), s);
    gpuLinearComb(bw, asInt(elems), d_open, T(1), d_open, T(1), d_mask_share);
    peer->reconstructInPlace(d_open, bw, elems, s);
    return d_open;
}

static T *openMaskedInputFromGpuShare(GpuPeer *peer, T *d_value_share,
                                      T *d_mask_share, size_t elems, int bw,
                                      Stats *s)
{
    gpuLinearComb(bw, asInt(elems), d_value_share, T(1), d_value_share, T(1), d_mask_share);
    peer->reconstructInPlace(d_value_share, bw, elems, s);
    return d_value_share;
}

static T *publicMaskedOutputToShare(T *d_public_masked, T *d_mask_share,
                                    int party, size_t elems, int bw)
{
    T public_coeff = party == SERVER0 ? T(1) : T(0);
    gpuLinearComb(bw, asInt(elems), d_public_masked,
                  public_coeff, d_public_masked, T(-1), d_mask_share);
    return d_public_masked;
}

static void addBiasShareInPlace(T *d_value_share, const T *h_bias_share,
                                int rows, int cols, int bw, Stats *s)
{
    T *h_expanded = expandBias(h_bias_share, rows, cols);
    T *d_bias = (T *)moveToGPU((u8 *)h_expanded, (size_t)rows * cols * sizeof(T), s);
    gpuLinearComb(bw, rows * cols, d_value_share, T(1), d_value_share, T(1), d_bias);
    gpuFree(d_bias);
    delete[] h_expanded;
}

static T *reconstructShareOnCpu(GpuPeer *peer, const T *h_share, size_t elems, int bw)
{
    T *d = (T *)moveToGPU((u8 *)h_share, elems * sizeof(T), nullptr);
    peer->reconstructInPlace(d, bw, elems, nullptr);
    T *h_clear = (T *)moveToCPU((u8 *)d, elems * sizeof(T), nullptr);
    gpuFree(d);
    return h_clear;
}

static T *fssMulShareVec(GpuPeer *peer, int party, AESGlobalContext *gAES,
                         u8 **curPtrRef, u8 *startPtr,
                         const T *h_A_share, const T *h_B_share, int N,
                         int bw, int scale, Stats *stats)
{
    const TruncateType tr = TruncateType::TrWithSlack;

    auto d_mask_A = randomGEOnGpu<T>(N, bw);
    auto d_mask_As = splitMaskForParty(d_mask_A, party, N, bw);
    auto d_mask_B = randomGEOnGpu<T>(N, bw);
    auto d_mask_Bs = splitMaskForParty(d_mask_B, party, N, bw);

    auto d_mask_C = gpuKeygenMul<T>(curPtrRef, party, bw, scale, N,
                                    d_mask_A, d_mask_B, tr, gAES);
    auto d_mask_Cs = splitMaskForParty(d_mask_C, party, N, bw);

    u8 *readPtr = startPtr;
    auto k = readGPUMulKey<T>(&readPtr, N, N, N, tr);

    gpuFree(d_mask_A);
    gpuFree(d_mask_B);
    gpuFree(d_mask_C);

    auto d_masked_A = openMaskedInputFromCpuShare(peer, h_A_share, d_mask_As, N, bw, stats);
    auto d_masked_B = openMaskedInputFromCpuShare(peer, h_B_share, d_mask_Bs, N, bw, stats);
    gpuFree(d_mask_As);
    gpuFree(d_mask_Bs);

    auto d_C = gpuMul<T>(peer, party, bw, scale, N, k,
                         d_masked_A, d_masked_B, tr, gAES, stats);
    gpuFree(d_masked_A);
    gpuFree(d_masked_B);

    publicMaskedOutputToShare(d_C, d_mask_Cs, party, N, bw);
    gpuFree(d_mask_Cs);
    auto h_C_share = (T *)moveToCPU((u8 *)d_C, (size_t)N * sizeof(T), stats);
    gpuFree(d_C);
    return h_C_share;
}

static T *fssMulPublicScalarShare(GpuPeer *peer, int party, AESGlobalContext *gAES,
                                  u8 **curPtrRef, u8 *startPtr,
                                  double scalar, const T *h_X_share, int N,
                                  int bw, int scale, Stats *stats)
{
    std::vector<T> coeff(N, party == SERVER0 ? fpToU64(scalar, scale) : T(0));
    return fssMulShareVec(peer, party, gAES, curPtrRef, startPtr,
                          coeff.data(), h_X_share, N, bw, scale, stats);
}

static T *fssSoftmaxShareBC(GpuPeer *peer, int party, AESGlobalContext *gAES,
                            u8 **curPtrRef, u8 *startPtr,
                            const T *h_agg_share, int B, int C, int bw, int scale,
                            T *d_nExpMsbTab, T *d_nExpLsbTab, T *d_invTab,
                            Stats *stats)
{
    if (C <= 1)
    {
        T *h_O = new T[(size_t)B * C];
        T one_fp = (T)1 << scale;
        for (size_t i = 0; i < (size_t)B * C; ++i)
            h_O[i] = party == SERVER0 ? one_fp : T(0);
        return h_O;
    }

    int C_pad = 1;
    while (C_pad < C)
        C_pad <<= 1;
    if (C_pad < 2)
        C_pad = 2;

    const int sm_bw = 50;
    MaxpoolParams p;
    p.bw = sm_bw;
    p.bin = 38;
    p.scale = scale;
    p.scaleDiv = 0;
    p.bwBackprop = 0;
    p.N = B;
    p.imgH = 1;
    p.imgW = C_pad;
    p.C = 1;
    p.FH = 1;
    p.FW = C_pad;
    p.strideH = 1;
    p.strideW = C_pad;
    p.zPadHLeft = 0;
    p.zPadHRight = 0;
    p.zPadWLeft = 0;
    p.zPadWRight = 0;
    p.H = 1;
    p.W = 1;
    p.isLowerTriangular = false;

    const int sz_pad = B * C_pad;
    const T sentinel = -(T(1) << (p.bin - 3));
    const u64 sm_mask = (sm_bw >= 64) ? ~T(0) : ((T(1) << sm_bw) - 1);
    const T sm_mod = (T(1) << sm_bw);

    auto d_mask_I = randomGEOnGpu<T>(sz_pad, sm_bw);
    auto d_mask_Is = splitMaskForParty(d_mask_I, party, sz_pad, sm_bw);

    auto d_mask_O = gpuKeygenSoftmax(curPtrRef, party, p, d_mask_I, gAES);
    auto h_mask_O = (T *)moveToCPU((u8 *)d_mask_O, (size_t)sz_pad * sizeof(T), NULL);

    u8 *readPtr = startPtr;
    auto k = readGPUSoftMaxKey<T>(p, &readPtr);

    T *h_padded_share = new T[sz_pad];
    for (int b = 0; b < B; ++b)
    {
        for (int c = 0; c < C; ++c)
            h_padded_share[(size_t)b * C_pad + c] = h_agg_share[(size_t)b * C + c] & sm_mask;
        for (int c = C; c < C_pad; ++c)
            h_padded_share[(size_t)b * C_pad + c] = (party == SERVER0 ? sentinel : T(0)) & sm_mask;
    }

    auto d_masked = openMaskedInputFromCpuShare(peer, h_padded_share,
                                                d_mask_Is, sz_pad, sm_bw, stats);
    gpuFree(d_mask_Is);

    auto d_O = gpuSoftmax(peer, party, p, k, d_masked,
                          d_nExpMsbTab, d_nExpLsbTab, d_invTab, gAES, stats);
    auto h_O_masked = (T *)moveToCPU((u8 *)d_O, (size_t)sz_pad * sizeof(T), stats);
    gpuFree(d_masked);
    gpuFree(d_O);

    T *h_O_share = new T[(size_t)B * C];
    for (int b = 0; b < B; ++b)
    {
        for (int c = 0; c < C; ++c)
        {
            size_t in = (size_t)b * C_pad + c;
            T masked = h_O_masked[in] & sm_mask;
            T mask = h_mask_O[in] & sm_mask;
            h_O_share[(size_t)b * C + c] =
                (party == SERVER0) ? masked : (T((masked < mask) ? sm_mod : 0) - mask);
        }
    }

    delete[] h_padded_share;
    cpuFree(h_mask_O, true);
    cpuFree(h_O_masked, true);
    gpuFree(d_mask_I);
    gpuFree(d_mask_O);
    return h_O_share;
}

static T *fssReluShareVec(GpuPeer *peer, int party, AESGlobalContext *gAES,
                          u8 **curPtrRef, u8 *startPtr,
                          const T *h_in_share, int N, int bw, Stats *stats)
{
    static const u64 RELU_P = 0;
    static const u64 RELU_Q = 0;

    auto d_mask_in = randomGEOnGpu<T>(N, bw);
    auto d_mask_ins = splitMaskForParty(d_mask_in, party, N, bw);

    auto d_mask_out = gpuGenReluKey<T, T, RELU_P, RELU_Q, false>(
        curPtrRef, party, bw, bw, N, d_mask_in, gAES);
    auto d_mask_outs = splitMaskForParty(d_mask_out, party, N, bw);

    u8 *readPtr = startPtr;
    auto k = readReluKey<T>(&readPtr);

    gpuFree(d_mask_in);
    gpuFree(d_mask_out);

    auto d_masked = openMaskedInputFromCpuShare(peer, h_in_share, d_mask_ins, N, bw, stats);
    gpuFree(d_mask_ins);

    auto d_O = gpuRelu<T, T, RELU_P, RELU_Q, false>(peer, party, k, d_masked, gAES, stats);
    gpuFree(d_masked);

    publicMaskedOutputToShare(d_O, d_mask_outs, party, N, bw);
    gpuFree(d_mask_outs);
    auto h_O_share = (T *)moveToCPU((u8 *)d_O, (size_t)N * sizeof(T), stats);
    gpuFree(d_O);
    return h_O_share;
}

static T *fssInverseShareVec(GpuPeer *peer, int party, AESGlobalContext *gAES,
                             u8 **curPtrRef, u8 *startPtr,
                             const T *h_in_share, int N, int bw, int bin, int scale,
                             T *d_invTab, Stats *stats)
{
    auto d_mask_in = randomGEOnGpu<T>(N, bw);
    auto d_mask_ins = splitMaskForParty(d_mask_in, party, N, bw);

    auto d_mask_out = gpuKeygenLUTInverse<T>(curPtrRef, party, bw, bin, scale,
                                             N, d_mask_in, gAES);
    auto d_mask_outs = splitMaskForParty(d_mask_out, party, N, bw);

    u8 *readPtr = startPtr;
    auto k = readGPULUTInverseKey<T>(&readPtr);

    gpuFree(d_mask_in);
    gpuFree(d_mask_out);

    auto d_masked = openMaskedInputFromCpuShare(peer, h_in_share, d_mask_ins,
                                                N, bw, stats);
    gpuFree(d_mask_ins);

    auto d_inv = gpuLUTInverse(peer, party, bw, bin, scale, N, k,
                               d_masked, d_invTab, gAES, stats);
    gpuFree(d_masked);

    publicMaskedOutputToShare(d_inv, d_mask_outs, party, N, bw);
    gpuFree(d_mask_outs);
    auto h_inv_share = (T *)moveToCPU((u8 *)d_inv, (size_t)N * sizeof(T), stats);
    gpuFree(d_inv);
    return h_inv_share;
}

static T *fssLUTShareVec(GpuPeer *peer, int party, AESGlobalContext *gAES,
                         u8 **curPtrRef, u8 *startPtr,
                         const T *h_in_share, int N, int in_bw, int out_bw,
                         T *d_tab, Stats *stats)
{
    auto d_mask_in = randomGEOnGpu<T>(N, in_bw);
    auto d_mask_ins = splitMaskForParty(d_mask_in, party, N, in_bw);

    auto d_mask_out = gpuKeyGenLUT<T, T>(curPtrRef, party, in_bw, out_bw,
                                         N, d_mask_in, gAES);
    auto d_mask_outs = splitMaskForParty(d_mask_out, party, N, out_bw);

    u8 *readPtr = startPtr;
    auto k = readGPULUTKey<T>(&readPtr);

    gpuFree(d_mask_in);
    gpuFree(d_mask_out);

    auto d_masked = openMaskedInputFromCpuShare(peer, h_in_share, d_mask_ins,
                                                N, in_bw, stats);
    gpuFree(d_mask_ins);

    auto d_out = gpuDpfLUT<T, T>(k, peer, party, d_masked, d_tab,
                                 gAES, stats);
    gpuFree(d_masked);

    publicMaskedOutputToShare(d_out, d_mask_outs, party, N, out_bw);
    gpuFree(d_mask_outs);
    auto h_out_share = (T *)moveToCPU((u8 *)d_out, (size_t)N * sizeof(T), stats);
    gpuFree(d_out);
    return h_out_share;
}

static T *fssInvSqrtShareVec(GpuPeer *peer, int party, AESGlobalContext *gAES,
                             u8 **curPtrRef, u8 *startPtr,
                             const T *h_in_share, int N, int in_bw, int out_bw,
                             T *d_invSqrtTab, Stats *stats)
{
    return fssLUTShareVec(peer, party, gAES, curPtrRef, startPtr,
                          h_in_share, N, in_bw, out_bw, d_invSqrtTab, stats);
}

struct LayerSizes
{
    MatmulParams p11, p12, p21, p22;
};

static LayerSizes makeLS(int Ns, int F, int H, int C, int bw, int scale)
{
    LayerSizes ls;
    auto fill = [&](MatmulParams &p, int M, int K, int N)
    {
        p.batchSz = 1;
        p.M = M;
        p.K = K;
        p.N = N;
        stdInit(p, bw, scale);
    };
    fill(ls.p11, Ns, F, H);
    fill(ls.p12, Ns, Ns, H);
    fill(ls.p21, Ns, H, C);
    fill(ls.p22, Ns, Ns, C);
    return ls;
}

static T *computeV0PosteriorShareFSS(int shard, int party, GpuPeer *peer,
                                     AESGlobalContext *gAES,
                                     int Nv0, int F, int H, int C,
                                     int bw, int scale, u8 *keyBufStart,
                                     T *d_nExpMsbTab, T *d_nExpLsbTab, T *d_invTab,
                                     Stats *stats)
{
    const TruncateType tr = TruncateType::TrWithSlack;
    LayerSizes ls = makeLS(Nv0, F, H, C, bw, scale);

    T *h_X_share = readBin(sharePath(ROOT + "/v0/v0", "feat", party), (size_t)Nv0 * F);
    T *h_A_share = readBin(sharePath(ROOT + "/v0/v0", "adj", party), (size_t)Nv0 * Nv0);
    const std::string wd = ROOT + "/weights/shard_" + std::to_string(shard);
    T *h_W1_share = readBin(sharePath(wd, "W1", party), (size_t)F * H);
    T *h_b1_share = readBin(sharePath(wd, "b1", party), H);
    T *h_W2_share = readBin(sharePath(wd, "W2", party), (size_t)H * C);
    T *h_b2_share = readBin(sharePath(wd, "b2", party), C);

    u8 *startPtr = keyBufStart;
    u8 *curPtr = keyBufStart;

    auto d_mask_X = randomGEOnGpu<T>((size_t)Nv0 * F, bw);
    auto d_mask_Xs = splitMaskForParty(d_mask_X, party, (size_t)Nv0 * F, bw);
    auto d_mask_A = randomGEOnGpu<T>((size_t)Nv0 * Nv0, bw);
    auto d_mask_As = splitMaskForParty(d_mask_A, party, (size_t)Nv0 * Nv0, bw);
    auto d_mask_W1 = randomGEOnGpu<T>((size_t)F * H, bw);
    auto d_mask_W1s = splitMaskForParty(d_mask_W1, party, (size_t)F * H, bw);
    auto d_mask_W2 = randomGEOnGpu<T>((size_t)H * C, bw);
    auto d_mask_W2s = splitMaskForParty(d_mask_W2, party, (size_t)H * C, bw);

    auto d_mask_T1 = gpuKeygenMatmul<T>(&curPtr, party, ls.p11,
                                        d_mask_X, d_mask_W1, (T *)nullptr,
                                        tr, gAES, true);
    auto d_mask_T1s = splitMaskForParty(d_mask_T1, party, (size_t)Nv0 * H, bw);

    auto h_mask_T1 = (T *)moveToCPU((u8 *)d_mask_T1, (size_t)Nv0 * H * sizeof(T), nullptr);
    auto d_mask_U1 = gpuKeygenMatmul<T>(&curPtr, party, ls.p12,
                                        d_mask_A, h_mask_T1, (T *)nullptr, tr, gAES);
    auto d_mask_U1s = splitMaskForParty(d_mask_U1, party, (size_t)Nv0 * H, bw);
    cpuFree(h_mask_T1);
    gpuFree(d_mask_T1);

    auto d_mask_H1 = gpuGenReluKey<T, T, RELU_P, RELU_Q, false>(
        &curPtr, party, bw, bw, ls.p12.size_C, d_mask_U1, gAES);
    auto d_mask_H1s = splitMaskForParty(d_mask_H1, party, (size_t)Nv0 * H, bw);
    gpuFree(d_mask_U1);

    auto d_mask_T2 = gpuKeygenMatmul<T>(&curPtr, party, ls.p21,
                                        d_mask_H1, d_mask_W2, (T *)nullptr,
                                        tr, gAES, true);
    auto d_mask_T2s = splitMaskForParty(d_mask_T2, party, (size_t)Nv0 * C, bw);
    gpuFree(d_mask_H1);

    auto h_mask_T2 = (T *)moveToCPU((u8 *)d_mask_T2, (size_t)Nv0 * C * sizeof(T), nullptr);
    auto d_mask_Z = gpuKeygenMatmul<T>(&curPtr, party, ls.p22,
                                       d_mask_A, h_mask_T2, (T *)nullptr, tr, gAES);
    auto d_mask_Zs = splitMaskForParty(d_mask_Z, party, (size_t)Nv0 * C, bw);
    cpuFree(h_mask_T2);
    gpuFree(d_mask_T2);

    gpuFree(d_mask_X);
    gpuFree(d_mask_A);
    gpuFree(d_mask_W1);
    gpuFree(d_mask_W2);
    gpuFree(d_mask_Z);

    startPtr = keyBufStart;
    auto k11 = readGPUMatmulKey<T>(ls.p11, tr, &startPtr);
    auto k12 = readGPUMatmulKey<T>(ls.p12, tr, &startPtr);
    auto krelu = readReluKey<T>(&startPtr);
    auto k21 = readGPUMatmulKey<T>(ls.p21, tr, &startPtr);
    auto k22 = readGPUMatmulKey<T>(ls.p22, tr, &startPtr);

    auto d_masked_X = openMaskedInputFromCpuShare(peer, h_X_share, d_mask_Xs, (size_t)Nv0 * F, bw, stats);
    auto d_masked_A = openMaskedInputFromCpuShare(peer, h_A_share, d_mask_As, (size_t)Nv0 * Nv0, bw, stats);
    auto d_masked_W1 = openMaskedInputFromCpuShare(peer, h_W1_share, d_mask_W1s, (size_t)F * H, bw, stats);
    auto d_masked_W2 = openMaskedInputFromCpuShare(peer, h_W2_share, d_mask_W2s, (size_t)H * C, bw, stats);
    gpuFree(d_mask_Xs);
    gpuFree(d_mask_As);
    gpuFree(d_mask_W1s);
    gpuFree(d_mask_W2s);

    auto d_T1 = gpuMatmul<T>(peer, party, ls.p11, k11,
                             d_masked_X, d_masked_W1, (T *)nullptr,
                             tr, gAES, stats, true);
    gpuFree(d_masked_X);
    gpuFree(d_masked_W1);
    publicMaskedOutputToShare(d_T1, d_mask_T1s, party, (size_t)Nv0 * H, bw);

    auto d_masked_T1 = openMaskedInputFromGpuShare(peer, d_T1, d_mask_T1s,
                                                   (size_t)Nv0 * H, bw, stats);
    gpuFree(d_mask_T1s);
    auto d_U1 = gpuMatmul<T>(peer, party, ls.p12, k12,
                             d_masked_A, d_masked_T1, (T *)nullptr,
                             tr, gAES, stats, true);
    gpuFree(d_masked_T1);
    publicMaskedOutputToShare(d_U1, d_mask_U1s, party, (size_t)Nv0 * H, bw);
    addBiasShareInPlace(d_U1, h_b1_share, Nv0, H, bw, stats);

    auto d_masked_U1 = openMaskedInputFromGpuShare(peer, d_U1, d_mask_U1s,
                                                   (size_t)Nv0 * H, bw, stats);
    gpuFree(d_mask_U1s);
    auto d_H1 = gpuRelu<T, T, RELU_P, RELU_Q, false>(
        peer, party, krelu, d_masked_U1, gAES, stats);
    gpuFree(d_masked_U1);
    publicMaskedOutputToShare(d_H1, d_mask_H1s, party, (size_t)Nv0 * H, bw);

    auto d_masked_H1 = openMaskedInputFromGpuShare(peer, d_H1, d_mask_H1s,
                                                   (size_t)Nv0 * H, bw, stats);
    gpuFree(d_mask_H1s);
    auto d_T2 = gpuMatmul<T>(peer, party, ls.p21, k21,
                             d_masked_H1, d_masked_W2, (T *)nullptr,
                             tr, gAES, stats, true);
    gpuFree(d_masked_H1);
    gpuFree(d_masked_W2);
    publicMaskedOutputToShare(d_T2, d_mask_T2s, party, (size_t)Nv0 * C, bw);

    auto d_masked_T2 = openMaskedInputFromGpuShare(peer, d_T2, d_mask_T2s,
                                                   (size_t)Nv0 * C, bw, stats);
    gpuFree(d_mask_T2s);
    auto d_Z = gpuMatmul<T>(peer, party, ls.p22, k22,
                            d_masked_A, d_masked_T2, (T *)nullptr,
                            tr, gAES, stats, true);
    gpuFree(d_masked_A);
    gpuFree(d_masked_T2);
    publicMaskedOutputToShare(d_Z, d_mask_Zs, party, (size_t)Nv0 * C, bw);
    gpuFree(d_mask_Zs);
    addBiasShareInPlace(d_Z, h_b2_share, Nv0, C, bw, stats);

    auto h_Z_share = (T *)moveToCPU((u8 *)d_Z, (size_t)Nv0 * C * sizeof(T), stats);
    gpuFree(d_Z);
    (void)curPtr;
    (void)d_nExpMsbTab;
    (void)d_nExpLsbTab;
    (void)d_invTab;

    delete[] h_X_share;
    delete[] h_A_share;
    delete[] h_W1_share;
    delete[] h_b1_share;
    delete[] h_W2_share;
    delete[] h_b2_share;
    return h_Z_share;
}

int main(int argc, char *argv[])
{
    int party = 0;
    const char *ip = "127.0.0.1";
    int num_epochs = 200;
    int batch_size = 32;
    double lr = 0.01;
    double lam = 0.01;
    double momentum = 0.9;
    int seed = 0;

    if (argc < 3)
    {
        fprintf(stderr,
                "Usage: %s <party> <peer_ip> [--num-epochs N] [--batch B] [--lr 0.01] [--lam 0.01]\n",
                argv[0]);
        return 1;
    }
    party = atoi(argv[1]);
    ip = argv[2];
    for (int i = 3; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--num-epochs" && i + 1 < argc)
            num_epochs = atoi(argv[++i]);
        else if (a == "--batch" && i + 1 < argc)
            batch_size = atoi(argv[++i]);
        else if (a == "--lr" && i + 1 < argc)
            lr = atof(argv[++i]);
        else if (a == "--lam" && i + 1 < argc)
            lam = atof(argv[++i]);
        else if (a == "--seed" && i + 1 < argc)
            seed = atoi(argv[++i]);
    }

    initGPUMemPool();

    {
        uint64_t threshold = 0;
        cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold);
        cudaDeviceSynchronize();
        cudaMemPoolTrimTo(mempool, 0);
        size_t freeB, totalB;
        cudaMemGetInfo(&freeB, &totalB);
        printf("[mem] after init trim: free=%.0f MB / total=%.0f MB\n",
               freeB / 1e6, totalB / 1e6);
        fflush(stdout);
    }
    AESGlobalContext gAES;
    initAESContext(&gAES);

    auto peer = new GpuPeer(true);
    peer->connect(party, (char *)ip);

    initGPURandomness();

    const int bw = 64;
    if (!fileExists(ROOT + "/weights/weights_meta.txt"))
    {
        fprintf(stderr,
                "ERROR: %s/weights/weights_meta.txt not found.\n"
                "       L3 LBAggr needs pre-trained OpenGU shard weights.\n"
                "       Re-run prepare_shards.py with --weight-dir pointing at\n"
                "       /home/OpenGU/data/GraphEraser/<dataset>/.\n",
                ROOT.c_str());
        return 1;
    }
    const int scale = parseMeta(ROOT + "/meta.txt", "scale");
    const int k = parseMeta(ROOT + "/meta.txt", "k");
    const int C = parseMeta(ROOT + "/meta.txt", "C");
    const int Nv0 = parseMeta(ROOT + "/v0/v0_meta.txt", "N_v0");
    const int F = parseMeta(ROOT + "/weights/weights_meta.txt", "F");
    const int H = parseMeta(ROOT + "/weights/weights_meta.txt", "H");

    if (scale != 12)
    {
        fprintf(stderr,
                "ERROR: L3 FSS softmax requires scale=12 (meta.txt has scale=%d).\n"
                "       Re-run prepare_shards.py without --scale (default is 12).\n",
                scale);
        return 1;
    }
    if (k <= 0 || C <= 0 || Nv0 <= 0 || F <= 0 || H <= 0)
    {
        fprintf(stderr,
                "ERROR: invalid meta: k=%d C=%d Nv0=%d F=%d H=%d\n"
                "       (negative/zero suggests prepare_shards.py wrote a stub meta.txt).\n",
                k, C, Nv0, F, H);
        return 1;
    }
    printf("LBAggr  k=%d  Nv0=%d  C=%d  F=%d  H=%d  scale=%d\n", k, Nv0, C, F, H, scale);
    printf("        epochs=%d  batch=%d  lr=%.4f  lam=%.4f  momentum=%.3f\n",
           num_epochs, batch_size, lr, lam, momentum);

    const size_t KEY_BUF = (size_t)256 * 1024 * 1024;
    u8 *kStart = nullptr, *kCur = nullptr;
    getKeyBuf(&kStart, &kCur, KEY_BUF);

    T *y_oh_share_raw = readBin(ROOT + "/v0/v0_labels_share" + std::to_string(party) + ".bin",
                                (size_t)Nv0 * C);
    std::vector<T> y_oh_share((size_t)Nv0 * C);
    for (size_t i = 0; i < (size_t)Nv0 * C; ++i)
        y_oh_share[i] = y_oh_share_raw[i] << scale;
    delete[] y_oh_share_raw;

    const int inv_bin = std::max(8, int(std::ceil(std::log2((double)C))) + scale);
    const int inv_sqrt_bin = scale + 8; // norm^2 range [0, 256) at scale=12.
    auto d_nExpMsbTab = genLUT<T, nExpMsb<T>>(8, 4, scale);
    auto d_nExpLsbTab = genLUT<T, nExpLsb<T>>(8, 12, scale);
    auto d_invTab = genLUT<T, inv<T>>(inv_bin, 6, scale);
    auto d_invSqrtTab = genLUT<T, invSqrtFixedPoint<T>>(inv_sqrt_bin, scale, scale);
    printf("[lut] nExp(8,4,%d) + nExp(8,12,%d) + inv(%d,6,%d) + invsqrt(%d,%d,%d) ready\n",
           scale, scale, inv_bin, scale, inv_sqrt_bin, scale, scale);

    ensureDir(ROOT + "/posteriors");
    Stats stats;
    std::memset(&stats, 0, sizeof(stats));
    std::vector<std::vector<T>> post_share(k, std::vector<T>((size_t)Nv0 * C));
    for (int s = 0; s < k; ++s)
    {
        std::string p = ROOT + "/posteriors/v0_shard_" + std::to_string(s) +
                        "_post_share" + std::to_string(party) + ".bin";
        if (fileExists(p))
        {
            T *u = readBin(p, (size_t)Nv0 * C);
            std::copy(u, u + (size_t)Nv0 * C, post_share[s].begin());
            delete[] u;
        }
        else
        {
            printf("[v0] shard %d: computing raw-logits posterior share under FSS\n", s);
            fflush(stdout);
            T *u = computeV0PosteriorShareFSS(s, party, peer, &gAES,
                                              Nv0, F, H, C, bw, scale,
                                              kStart,
                                              d_nExpMsbTab, d_nExpLsbTab, d_invTab,
                                              &stats);
            std::copy(u, u + (size_t)Nv0 * C, post_share[s].begin());
            writeBin(p, u, (size_t)Nv0 * C);
            cpuFree(u);
        }
    }
    printf("V0 posterior shares ready for %d shards.\n", k);

    // ----- alpha + velocity as additive shares.  Uniform alpha is public at
    // initialization, represented as party0=1/k and party1=0. -----
    std::vector<T> alpha_share(k, party == SERVER0 ? fpToU64(1.0 / k, scale) : T(0));
    std::vector<T> Vw_share(k, T(0));
    fflush(stdout);

    std::mt19937 rng(seed);

    // ===== training loop (all training state stays as additive shares) =====
    auto t0 = std::chrono::high_resolution_clock::now();
    u64 totalComm0 = peer->bytesSent() + peer->bytesReceived();

    for (int epoch = 0; epoch < num_epochs; ++epoch)
    {
        // shuffle V0 indices
        std::vector<int> perm(Nv0);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);

        for (int off = 0; off < Nv0; off += batch_size)
        {
            int B = std::min(batch_size, Nv0 - off);
            const int BC = B * C;

            std::vector<std::vector<T>> post_batch(k, std::vector<T>((size_t)B * C));
            std::vector<std::vector<T>> alpha_exp(k, std::vector<T>((size_t)B * C));
            std::vector<T> y_batch((size_t)B * C);
            for (int s = 0; s < k; ++s)
            {
                for (int b = 0; b < B; ++b)
                {
                    int idx = perm[off + b];
                    for (int c = 0; c < C; ++c)
                    {
                        post_batch[s][(size_t)b * C + c] =
                            post_share[s][(size_t)idx * C + c];
                        alpha_exp[s][(size_t)b * C + c] = alpha_share[s];
                    }
                }
            }
            for (int b = 0; b < B; ++b)
            {
                int idx = perm[off + b];
                for (int c = 0; c < C; ++c)
                    y_batch[(size_t)b * C + c] = y_oh_share[(size_t)idx * C + c];
            }

            std::vector<T> agg_share((size_t)B * C, T(0));
            for (int s = 0; s < k; ++s)
            {
                kCur = kStart;
                T *h_C = fssMulShareVec(peer, party, &gAES, &kCur, kStart,
                                        alpha_exp[s].data(), post_batch[s].data(),
                                        BC, bw, scale, &stats);
                addVecInPlace(agg_share, h_C);
                cpuFree(h_C);
            }

            kCur = kStart;
            T *h_probs_share = fssSoftmaxShareBC(peer, party, &gAES, &kCur, kStart,
                                                 agg_share.data(), B, C, bw, scale,
                                                 d_nExpMsbTab, d_nExpLsbTab, d_invTab,
                                                 &stats);

            std::vector<T> diff_share((size_t)B * C);
            for (int i = 0; i < BC; ++i)
                diff_share[i] = h_probs_share[i] - y_batch[i];
            delete[] h_probs_share;

            kCur = kStart;
            T *h_grad_share = fssMulPublicScalarShare(peer, party, &gAES,
                                                      &kCur, kStart,
                                                      1.0 / B,
                                                      diff_share.data(),
                                                      BC, bw, scale, &stats);

            std::vector<T> dalpha_share(k, T(0));
            for (int s = 0; s < k; ++s)
            {
                kCur = kStart;
                T *h_C = fssMulShareVec(peer, party, &gAES, &kCur, kStart,
                                        h_grad_share, post_batch[s].data(),
                                        BC, bw, scale, &stats);
                T acc = T(0);
                for (int i = 0; i < BC; ++i)
                    acc += h_C[i];
                dalpha_share[s] = acc;
                cpuFree(h_C);
            }
            cpuFree(h_grad_share);

            // Exact OpenGU regularizer gradient:
            //   ∂(λ||α||₂)/∂α_s = λ·α_s / sqrt(Σ_j α_j²)
            std::vector<T> grad_total_share = dalpha_share;
            if (lam != 0.0)
            {
                kCur = kStart;
                T *h_alpha_sq_share = fssMulShareVec(peer, party, &gAES,
                                                     &kCur, kStart,
                                                     alpha_share.data(),
                                                     alpha_share.data(),
                                                     k, bw, scale, &stats);
                T norm_sq_share = T(0);
                for (int s = 0; s < k; ++s)
                    norm_sq_share += h_alpha_sq_share[s];
                cpuFree(h_alpha_sq_share);

                kCur = kStart;
                T *h_inv_norm_share = fssInvSqrtShareVec(peer, party, &gAES,
                                                         &kCur, kStart,
                                                         &norm_sq_share, 1,
                                                         inv_sqrt_bin, bw,
                                                         d_invSqrtTab, &stats);
                std::vector<T> inv_norm_exp(k, h_inv_norm_share[0]);
                cpuFree(h_inv_norm_share);

                kCur = kStart;
                T *h_alpha_over_norm_share = fssMulShareVec(peer, party, &gAES,
                                                            &kCur, kStart,
                                                            alpha_share.data(),
                                                            inv_norm_exp.data(),
                                                            k, bw, scale, &stats);
                kCur = kStart;
                T *h_reg_share = fssMulPublicScalarShare(peer, party, &gAES,
                                                         &kCur, kStart,
                                                         lam, h_alpha_over_norm_share,
                                                         k, bw, scale, &stats);
                for (int s = 0; s < k; ++s)
                    grad_total_share[s] += h_reg_share[s];
                cpuFree(h_alpha_over_norm_share);
                cpuFree(h_reg_share);
            }

            // SGD with momentum on shares:
            //   Vw <- momentum*Vw - lr*grad
            //   alpha <- ReLU(alpha + Vw)
            kCur = kStart;
            T *h_mom_share = fssMulPublicScalarShare(peer, party, &gAES,
                                                     &kCur, kStart,
                                                     momentum, Vw_share.data(),
                                                     k, bw, scale, &stats);
            kCur = kStart;
            T *h_lr_grad_share = fssMulPublicScalarShare(peer, party, &gAES,
                                                         &kCur, kStart,
                                                         lr, grad_total_share.data(),
                                                         k, bw, scale, &stats);
            std::vector<T> alpha_pre(k);
            for (int s = 0; s < k; ++s)
            {
                Vw_share[s] = h_mom_share[s] - h_lr_grad_share[s];
                alpha_pre[s] = alpha_share[s] + Vw_share[s];
            }
            cpuFree(h_mom_share);
            cpuFree(h_lr_grad_share);

            kCur = kStart;
            T *h_alpha_proj = fssReluShareVec(peer, party, &gAES, &kCur, kStart,
                                              alpha_pre.data(), k, bw, &stats);
            std::copy(h_alpha_proj, h_alpha_proj + k, alpha_share.begin());
            cpuFree(h_alpha_proj);
        }

        if (epoch % 10 == 0 || epoch == num_epochs - 1)
        {
            T *alpha_clear = reconstructShareOnCpu(peer, alpha_share.data(), k, bw);
            double sum = 0.0;
            for (int s = 0; s < k; ++s)
                sum += u64ToFp(alpha_clear[s], scale);
            printf("epoch %3d  α = [", epoch);
            for (int s = 0; s < k; ++s)
                printf(" %.3f", u64ToFp(alpha_clear[s], scale) / std::max(sum, 1e-9));
            printf(" ]\n");
            fflush(stdout);
            cpuFree(alpha_clear);
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    u64 totalComm = (peer->bytesSent() + peer->bytesReceived()) - totalComm0;
    printf("\nLBAggr training: %ld ms,  FSS comm = %lu MB\n",
           std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count(),
           totalComm / (1024 * 1024));
    gpuFree(d_invSqrtTab);

    // ----- final normalization on shares: alpha <- alpha / Σ alpha. -----
    T sum_share = T(0);
    for (int s = 0; s < k; ++s)
        sum_share += alpha_share[s];
    const int alpha_inv_bin = std::max(8, int(std::ceil(std::log2((double)k + 1.0))) + scale);
    auto d_alphaInvTab = genLUT<T, inv<T>>(alpha_inv_bin, 6, scale);
    kCur = kStart;
    T *h_inv_sum_share = fssInverseShareVec(peer, party, &gAES, &kCur, kStart,
                                            &sum_share, 1, bw, alpha_inv_bin,
                                            scale, d_alphaInvTab, &stats);
    std::vector<T> inv_sum_exp(k, h_inv_sum_share[0]);
    cpuFree(h_inv_sum_share);
    gpuFree(d_alphaInvTab);

    kCur = kStart;
    T *h_alpha_norm_share = fssMulShareVec(peer, party, &gAES, &kCur, kStart,
                                           alpha_share.data(), inv_sum_exp.data(),
                                           k, bw, scale, &stats);
    std::string alpha_share_path = ROOT + "/alpha_share" + std::to_string(party) + ".bin";
    writeBin(alpha_share_path, h_alpha_norm_share, k);
    printf("wrote %s\n", alpha_share_path.c_str());

    T *alpha_clear = reconstructShareOnCpu(peer, h_alpha_norm_share, k, bw);
    if (party == SERVER0)
    {
        writeBin(ROOT + "/alpha.bin", alpha_clear, k);
        printf("wrote %s (regression clear reveal)\n", (ROOT + "/alpha.bin").c_str());
    }
    printf("\nfinal α =");
    for (int s = 0; s < k; ++s)
        printf(" %.4f", u64ToFp(alpha_clear[s], scale));
    printf("\n");
    cpuFree(alpha_clear);
    cpuFree(h_alpha_norm_share);

    fflush(stdout);
    _exit(0);
}
