// Shared dealer/evaluator helpers for the clean FSS GCN (GPU-GCN/experiments/GCN).
// Mirrors experiments/orca/{orca_dealer,orca_evaluator}.cu: genModelKey (dealer)
// and trainModel (evaluator) drive the GPUModel layer-by-layer, with the
// piranha softmax (scale 24) outside the layer loop. The only GCN-specific
// pieces are getGCNModel (gcn_model.h) and feeding the adjacency A into the two
// AggLayers via setAShare() before the forward pass (done in the evaluator main).
#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "utils/gpu_data_types.h"
#include "utils/gpu_file_utils.h"
#include "utils/misc_utils.h"
#include "utils/gpu_comms.h"
#include "utils/gpu_mem.h"
#include "utils/gpu_random.h"

#include "experiments/GCN/gcn_model.h"

#include <sytorch/backend/llama_base.h>
#include <sytorch/softmax.h>

namespace gcn
{
    using namespace dcf::orca;

    // ---------- shard dataset ----------
    struct ShardMeta { int Ns=-1, F=-1, H=-1, C=-1, scale=-1; };

    inline int kvInt(const std::string &l)
    {
        auto eq = l.find('='); return eq == std::string::npos ? -1 : std::atoi(l.c_str() + eq + 1);
    }
    inline void loadShardMeta(const std::string &path, ShardMeta *m)
    {
        std::ifstream f(path);
        assert(f.is_open() && "missing shard meta.txt");
        std::string ln;
        while (std::getline(f, ln))
        {
            if (ln.rfind("Ns=", 0) == 0) m->Ns = kvInt(ln);
            else if (ln.rfind("F=", 0) == 0) m->F = kvInt(ln);
            else if (ln.rfind("H=", 0) == 0) m->H = kvInt(ln);
            else if (ln.rfind("C=", 0) == 0) m->C = kvInt(ln);
            else if (ln.rfind("scale=", 0) == 0) m->scale = kvInt(ln);
        }
        assert(m->Ns > 0 && m->F > 0 && m->H > 0 && m->C > 0);
    }
    template <typename U>
    inline U *readBinT(const std::string &path, size_t elems)
    {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        assert(f.is_open() && "missing share/bin file");
        size_t bytes = (size_t)f.tellg();
        assert(bytes == elems * sizeof(U) && "size mismatch");
        f.seekg(0);
        U *buf = (U *)cpuMalloc(elems * sizeof(U));
        f.read((char *)buf, bytes);
        return buf;
    }
    inline std::string shardDir(const std::string &root, int shard)
    {
        return root + "/shard" + std::to_string(shard);
    }
    inline std::string sharePath(const std::string &dir, const char *name, int party)
    {
        return dir + "/" + name + "_share" + std::to_string(party) + ".bin";
    }

    // ---------- dealer: softmax key + full model key ----------
    inline u64 *gpuGenSoftmaxKey(int batchSz, int numClasses, u64 *d_mask_I, LlamaBase<u64> *llama)
    {
        int Bp = 1; while (Bp < batchSz) Bp <<= 1; // PiranhaSoftmax needs pow2 rows
        Tensor4D<u64> inpMask(Bp, numClasses, 1, 1);
        Tensor4D<u64> softmaxOpMask(Bp, numClasses, 1, 1);
        size_t realSz = (size_t)batchSz * numClasses * sizeof(u64);
        memset(inpMask.data, 0, (size_t)Bp * numClasses * sizeof(u64));
        moveIntoCPUMem((u8 *)inpMask.data, (u8 *)d_mask_I, realSz, NULL); // first batchSz rows
        gpuFree(d_mask_I);
        pirhana_softmax(inpMask, softmaxOpMask, dcf::orca::global::scale);
        return (u64 *)moveToGPU((u8 *)softmaxOpMask.data, realSz, NULL); // first batchSz rows
    }

    inline void genModelKey(GPUModel<u64> *m, u8 **bufPtr, int party, AESGlobalContext *g,
                            LlamaBase<u64> *llama, int epoch)
    {
        auto d_mask_I = randomGEOnGpu<u64>(m->inpSz, dcf::orca::global::bw);
        u64 *d_mask_O = NULL;
        for (size_t i = 0; i < m->layers.size(); i++)
        {
            d_mask_O = m->layers[i]->genForwardKey(bufPtr, party, d_mask_I, g);
            assert(d_mask_O != d_mask_I);
            gpuFree(d_mask_I);
            d_mask_I = d_mask_O;
        }
        d_mask_I = gpuGenSoftmaxKey(m->batchSz, m->classes, d_mask_I, llama);
        for (int i = (int)m->layers.size() - 1; i >= 0; i--)
        {
            d_mask_I = m->layers[i]->genBackwardKey(bufPtr, party, d_mask_I, g, epoch);
        }
        // agg1 (layer 0, computeGrad=false) returns NULL; nothing to free.
    }

    // ---------- evaluator: softmax (-> dZ = P - Y) + train step ----------
    inline u64 *gpuSoftmax(int batchSz, int numClasses, int party, SigmaPeer *peer,
                           u64 *d_I, u64 *labels, LlamaBase<u64> *llama)
    {
        int Bp = 1; while (Bp < batchSz) Bp <<= 1; // PiranhaSoftmax needs pow2 rows
        Tensor4D<u64> inp(Bp, numClasses, 1, 1);
        Tensor4D<u64> softmaxOp(Bp, numClasses, 1, 1);
        size_t realSz = (size_t)batchSz * numClasses * sizeof(u64);
        memset(inp.data, 0, (size_t)Bp * numClasses * sizeof(u64));
        moveIntoCPUMem((u8 *)inp.data, (u8 *)d_I, realSz, NULL); // first batchSz rows
        gpuFree(d_I);
        pirhana_softmax(inp, softmaxOp, dcf::orca::global::scale);
        for (int img = 0; img < batchSz; img++)
            for (int c = 0; c < numClasses; c++)
                softmaxOp(img, c, 0, 0) -= (labels[numClasses * img + c] * (((1LL << dcf::orca::global::scale)) / Bp));
        reconstruct(batchSz * numClasses, softmaxOp.data, 64); // dZ for the real rows
        return (u64 *)moveToGPU((u8 *)softmaxOp.data, realSz, NULL);
    }

    inline void trainStep(GPUModel<u64> *m, u8 **keyBuf, int party, SigmaPeer *peer,
                          u64 *data, u64 *labels, AESGlobalContext *g, LlamaBase<u64> *llama, int epoch)
    {
        size_t inpMemSz = m->inpSz * sizeof(u64);
        auto d_I = (u64 *)moveToGPU((u8 *)data, inpMemSz, &(m->layers[0]->s));
        u64 *d_O;
        for (size_t i = 0; i < m->layers.size(); i++)
        {
            m->layers[i]->readForwardKey(keyBuf);
            d_O = m->layers[i]->forward(peer, party, d_I, g);
            if (d_O != d_I) gpuFree(d_I);
            d_I = d_O;
        }
        checkCudaErrors(cudaDeviceSynchronize());
        d_I = gpuSoftmax(m->batchSz, m->classes, party, peer, d_I, labels, llama);
        for (int i = (int)m->layers.size() - 1; i >= 0; i--)
        {
            m->layers[i]->readBackwardKey(keyBuf, epoch);
            d_I = m->layers[i]->backward(peer, party, d_I, g, epoch);
        }
        // NOTE: do NOT free the final d_I -- the first layer (FC1, computedX=false)
        // returns an uninitialized d_dX (orca convention; no allocation was made).
    }

    // ---------- inference: forward only, returns posterior SHARE (CPU) ----------
    inline u64 *inferPost(GPUModel<u64> *m, u8 **keyBuf, int party, SigmaPeer *peer,
                          u64 *data, AESGlobalContext *g, LlamaBase<u64> *llama)
    {
        size_t inpMemSz = m->inpSz * sizeof(u64);
        auto d_I = (u64 *)moveToGPU((u8 *)data, inpMemSz, &(m->layers[0]->s));
        u64 *d_O;
        for (size_t i = 0; i < m->layers.size(); i++)
        {
            m->layers[i]->readForwardKey(keyBuf);
            d_O = m->layers[i]->forward(peer, party, d_I, g);
            if (d_O != d_I) gpuFree(d_I);
            d_I = d_O;
        }
        checkCudaErrors(cudaDeviceSynchronize());
        // softmax (probabilities, as a share) for posterior output
        int Bp = 1; while (Bp < m->batchSz) Bp <<= 1;
        Tensor4D<u64> inp(Bp, m->classes, 1, 1), op(Bp, m->classes, 1, 1);
        size_t memSz = (size_t)m->batchSz * m->classes * sizeof(u64);
        memset(inp.data, 0, (size_t)Bp * m->classes * sizeof(u64));
        moveIntoCPUMem((u8 *)inp.data, (u8 *)d_I, memSz, NULL);
        gpuFree(d_I);
        pirhana_softmax(inp, op, dcf::orca::global::scale);
        u64 *out = (u64 *)cpuMalloc(memSz);
        memcpy(out, op.data, memSz);
        return out; // additive share of softmax(Z)
    }

    inline void writeKeySzHelper(const std::string &dir, const std::string &modelName, u64 keySz)
    {
        makeDir(dir);
        std::ofstream f(dir + modelName + ".txt");
        f << keySz; f.close();
    }
    inline u64 getKeySzHelper(const std::string &dir, const std::string &modelName)
    {
        std::ifstream f(dir + modelName + ".txt");
        u64 s = 0; f >> s; return s;
    }

    inline double accuracyFromPostClear(const u64 *post_clear, const int64_t *labels, int N, int C)
    {
        int correct = 0;
        for (int i = 0; i < N; i++)
        {
            int best = 0; i64 bv = (i64)post_clear[(size_t)i * C];
            for (int c = 1; c < C; c++) { i64 v = (i64)post_clear[(size_t)i * C + c]; if (v > bv) { bv = v; best = c; } }
            correct += (best == (int)labels[i]);
        }
        return N == 0 ? 0.0 : (double)correct / (double)N;
    }

} // namespace gcn
