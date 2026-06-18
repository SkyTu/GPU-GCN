// GCN neighbourhood-aggregation layer for the Orca FSS layer framework.
//
//   O = A · I              (A: [Ns,Ns] secret adjacency, I: [Ns,Fin] activation)
//
// Both operands are SECRET 2-party shares, so this is a secret x secret
// Beaver-triple matmul (unlike FCLayer, whose weight W is public/zero-masked).
// It plugs into the same dealer (genForwardKey/genBackwardKey) and evaluator
// (readForwardKey/forward, readBackwardKey/backward) loops as FCLayer/ReLU,
// threading the layer's output mask into the next layer's input mask. No bias,
// no learnable weight (A is fixed graph data, not optimized).
//
// Convention (matches gpuMatmulBeaver operand order in fss/gpu_matmul.cu):
//   left operand  (slot A, mmKey.A = mask_A)  = the adjacency A
//   right operand (slot B, mmKey.B = mask_I)  = the chained activation I
// The chained input mask handed to genForwardKey is therefore the RIGHT mask.
//
// A is opened to its masked-public value ONCE per forward and cached for the
// backward pass (dI = A^T · dO; A is symmetric so this also equals A·dO, but we
// read it transposed via rowMaj_A=false to stay correct for any A).
#pragma once

#include <cstdint>
#include <cassert>
#include <cstring>

#include "utils/gpu_stats.h"
#include "utils/gpu_comms.h"
#include "utils/gpu_mem.h"
#include "utils/gpu_random.h"
#include "utils/misc_utils.h"

#include "fss/gpu_matmul.h"
#include "fss/dcf/gpu_truncate.h"

#include "nn/orca/gpu_layer.h"

namespace dcf
{
    namespace orca
    {

        template <typename T>
        class AggLayer : public GPULayer<T>
        {
        private:
            void initMemSz(MatmulParams q, GPUMatmulKey<T> *k)
            {
                k->mem_size_A = q.size_A * sizeof(T);
                k->mem_size_B = q.size_B * sizeof(T);
                k->mem_size_C = q.size_C * sizeof(T);
            }

        public:
            // forward  O[Ns,Fin] = A[Ns,Ns] · I[Ns,Fin]
            // backward dI[Ns,Fin] = A^T[Ns,Ns] · dO[Ns,Fin]
            MatmulParams p, pBwd;
            GPUMatmulKey<T> mmKey, mmKeyBwd;
            dcf::TruncateType tf, tb;
            GPUTruncateKey<T> truncateKeyZ, truncateKeydI;

            int Ns, Fin;
            bool inputIsShares; // true: right operand I arrives as additive shares (open it). false: already masked-public.
            bool computeGrad;   // true: emit dI in backward (agg feeding learnable layers). false: graph-input boundary.

            T *mask_A = nullptr;     // dealer: this party's full secret mask of A (CPU), reused fwd->bwd within an iter
            const T *h_A_share = nullptr; // evaluator: A's additive share (set by driver from the shard dataset)
            T *d_A_opened = nullptr; // evaluator: masked-public A cached from forward for reuse in backward

            AggLayer(int Ns_, int Fin_, dcf::TruncateType tf_, dcf::TruncateType tb_,
                     bool inputIsShares_, bool computeGrad_)
            {
                this->name = "Agg";
                Ns = Ns_;
                Fin = Fin_;
                inputIsShares = inputIsShares_;
                computeGrad = computeGrad_;
                tf = tf_;
                tb = tb_;

                // forward: A · I
                p.batchSz = 1;
                p.M = Ns;
                p.K = Ns;
                p.N = Fin;
                stdInit(p, global::bw, 0); // shift handled separately by dcf::gpuTruncate at global::scale
                initMemSz(p, &mmKey);

                // backward: A^T · dO  (transpose the left operand)
                pBwd.batchSz = 1;
                pBwd.M = Ns;
                pBwd.K = Ns;
                pBwd.N = Fin;
                stdInit(pBwd, global::bw, 0);
                pBwd.rowMaj_A = false;
                initMemSz(pBwd, &mmKeyBwd);

                mask_A = (T *)cpuMalloc(mmKey.mem_size_A);
            }

            void setAShare(const T *hA) { h_A_share = hA; }

            // ---------------- dealer ----------------
            T *genForwardKey(u8 **key_as_bytes, int party, T *d_mask_I, AESGlobalContext *gaes)
            {
                // d_mask_I is the chained RIGHT-operand (activation) mask.
                auto d_mask_A = randomGEOnGpu<T>(p.size_A, p.bw); // SECRET mask of A
                if (this->train)
                    moveIntoCPUMem((u8 *)mask_A, (u8 *)d_mask_A, mmKey.mem_size_A, NULL);
                auto d_mask_Z = randomGEOnGpu<T>(p.size_C, p.bw);
                auto d_masked_Z = gpuMatmulPlaintext(p, d_mask_A, d_mask_I, d_mask_Z, false);
                writeShares<T, T>(key_as_bytes, party, p.size_A, d_mask_A, p.bw);     // k.A = mask_A
                writeShares<T, T>(key_as_bytes, party, p.size_B, d_mask_I, p.bw);     // k.B = mask_I
                writeShares<T, T>(key_as_bytes, party, p.size_C, d_masked_Z, p.bw);   // k.C
                gpuFree(d_mask_A);
                gpuFree(d_masked_Z);
                auto d_mask_trunc_Z = genGPUTruncateKey(key_as_bytes, party, tf, p.bw, p.bw,
                                                        global::scale, p.size_C, d_mask_Z, gaes);
                return d_mask_trunc_Z; // chained output mask
            }

            T *genBackwardKey(u8 **key_as_bytes, int party, T *d_mask_grad, AESGlobalContext *gaes, int epoch)
            {
                this->checkIfTrain();
                if (!computeGrad)
                {
                    // graph-input boundary: no gradient to propagate, no key.
                    gpuFree(d_mask_grad);
                    return NULL;
                }
                // dI = A^T · dO ; reuse the stored forward mask_A.
                auto d_mask_A = (T *)moveToGPU((u8 *)mask_A, mmKeyBwd.mem_size_A, NULL);
                auto d_mask_dI = randomGEOnGpu<T>(pBwd.size_C, pBwd.bw);
                auto d_masked_dI = gpuMatmulPlaintext(pBwd, d_mask_A, d_mask_grad, d_mask_dI, false);
                writeShares<T, T>(key_as_bytes, party, pBwd.size_A, d_mask_A, pBwd.bw);   // kBwd.A = mask_A
                writeShares<T, T>(key_as_bytes, party, pBwd.size_B, d_mask_grad, pBwd.bw); // kBwd.B = mask_dO
                writeShares<T, T>(key_as_bytes, party, pBwd.size_C, d_masked_dI, pBwd.bw); // kBwd.C
                gpuFree(d_mask_A);
                gpuFree(d_masked_dI);
                gpuFree(d_mask_grad);
                auto d_mask_trunc_dI = genGPUTruncateKey(key_as_bytes, party, tb, pBwd.bw, pBwd.bw,
                                                         global::scale, pBwd.size_C, d_mask_dI, gaes);
                return d_mask_trunc_dI;
            }

            // ---------------- evaluator ----------------
            void readForwardKey(u8 **key_as_bytes)
            {
                mmKey.A = (T *)*key_as_bytes;
                *key_as_bytes += mmKey.mem_size_A;
                mmKey.B = (T *)*key_as_bytes;
                *key_as_bytes += mmKey.mem_size_B;
                mmKey.C = (T *)*key_as_bytes;
                *key_as_bytes += mmKey.mem_size_C;
                truncateKeyZ = readGPUTruncateKey<T>(tf, key_as_bytes);
            }

            void readBackwardKey(u8 **key_as_bytes, int epoch)
            {
                if (!computeGrad)
                    return;
                mmKeyBwd.A = (T *)*key_as_bytes;
                *key_as_bytes += mmKeyBwd.mem_size_A;
                mmKeyBwd.B = (T *)*key_as_bytes;
                *key_as_bytes += mmKeyBwd.mem_size_B;
                mmKeyBwd.C = (T *)*key_as_bytes;
                *key_as_bytes += mmKeyBwd.mem_size_C;
                truncateKeydI = readGPUTruncateKey<T>(tb, key_as_bytes);
            }

            T *forward(SigmaPeer *peer, int party, T *d_I, AESGlobalContext *gaes)
            {
                assert(h_A_share != nullptr && "AggLayer: A share not set (call setAShare)");
                // open A (left) to masked-public, cache for backward
                auto d_A = (T *)moveToGPU((u8 *)h_A_share, mmKey.mem_size_A, &(this->s));
                auto d_mask_A = (T *)moveToGPU((u8 *)mmKey.A, mmKey.mem_size_A, &(this->s));
                gpuLinearComb(p.bw, p.size_A, d_A, T(1), d_A, T(1), d_mask_A);
                peer->reconstructInPlace(d_A, p.bw, p.size_A, &(this->s));
                d_A_opened = d_A; // keep for backward

                // right operand I
                auto d_mask_I = (T *)moveToGPU((u8 *)mmKey.B, mmKey.mem_size_B, &(this->s));
                if (inputIsShares)
                {
                    gpuLinearComb(p.bw, p.size_B, d_I, T(1), d_I, T(1), d_mask_I);
                    peer->reconstructInPlace(d_I, p.bw, p.size_B, &(this->s));
                }

                auto d_Z = gpuMatmulBeaver(p, mmKey, party, d_A, d_I, d_mask_A, d_mask_I, (T *)NULL, &(this->s));
                gpuFree(d_mask_A);
                gpuFree(d_mask_I);

                peer->reconstructInPlace(d_Z, p.bw, p.size_C, &(this->s));
                dcf::gpuTruncate(p.bw, p.bw, tf, truncateKeyZ, global::scale, peer, party, p.size_C, d_Z, gaes, &(this->s));
                return d_Z;
            }

            T *backward(SigmaPeer *peer, int party, T *d_incomingGrad, AESGlobalContext *gaes, int epoch)
            {
                this->checkIfTrain();
                if (!computeGrad)
                {
                    gpuFree(d_incomingGrad);
                    if (d_A_opened) { gpuFree(d_A_opened); d_A_opened = nullptr; }
                    return NULL;
                }
                // dI = A^T · dO, reusing the masked-public A cached in forward.
                auto d_mask_A = (T *)moveToGPU((u8 *)mmKeyBwd.A, mmKeyBwd.mem_size_A, &(this->s));
                auto d_mask_dO = (T *)moveToGPU((u8 *)mmKeyBwd.B, mmKeyBwd.mem_size_B, &(this->s));
                auto d_dI = gpuMatmulBeaver(pBwd, mmKeyBwd, party, d_A_opened, d_incomingGrad,
                                            d_mask_A, d_mask_dO, (T *)NULL, &(this->s));
                gpuFree(d_mask_A);
                gpuFree(d_mask_dO);
                gpuFree(d_A_opened);
                d_A_opened = nullptr;
                gpuFree(d_incomingGrad);

                peer->reconstructInPlace(d_dI, pBwd.bw, pBwd.size_C, &(this->s));
                dcf::gpuTruncate(pBwd.bw, pBwd.bw, tb, truncateKeydI, global::scale, peer, party,
                                 pBwd.size_C, d_dI, gaes, &(this->s));
                return d_dI;
            }

            // A is graph data (set via setAShare), not a learnable weight -> no-op.
            void initWeights(u8 **weights, bool floatWeights) {}
            void dumpWeights(std::ofstream &f) {}
        };

    } // namespace orca
} // namespace dcf
