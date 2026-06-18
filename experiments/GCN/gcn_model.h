// 2-layer GCN as an Orca GPUModel, composed of the reused FC/ReLU layers and
// the new secret-secret AggLayer.  No bias.  Fixed-point scale = global::scale.
//
//   X --agg1(A·X)--> agg1 --FC1(·W1)--> T1 --ReLU--> H1 --agg2(A·H1)--> agg2 --FC2(·W2)--> Z --softmax
//
// Backward (only dW1,dW2 are learned; A,X carry no learnable params):
//   FC2.computedX=true   -> dAgg2 = dZ·W2^T   feeds agg2
//   agg2.computeGrad=true -> dH1  = A^T·dAgg2  feeds ReLU
//   FC1.computedX=false  -> agg1 needs no input-grad, so FC1 skips dX (still updates W1)
//   agg1.computeGrad=false-> graph-input boundary, chain ends
//
// The driver must call setAShare(<A share>) on BOTH AggLayers before forward.
#pragma once

#include <vector>

#include "nn/orca/gpu_model.h"
#include "nn/orca/fc_layer.h"
#include "nn/orca/relu_extend_layer.h"
#include "experiments/GCN/gcn_agg_layer.h"

namespace dcf
{
    namespace orca
    {

        template <typename T>
        struct GCNModel
        {
            GPUModel<T> *m = nullptr;
            AggLayer<T> *agg1 = nullptr; // A·X
            AggLayer<T> *agg2 = nullptr; // A·H1
            int Ns, F, H, C;
        };

        // Build the 2-layer-GCN GPUModel.  tf/tb = forward/backward truncation type.
        template <typename T>
        GCNModel<T> getGCNModel(int Ns, int F, int H, int C,
                                dcf::TruncateType tf = dcf::TruncateType::StochasticTruncate,
                                dcf::TruncateType tb = dcf::TruncateType::StochasticTruncate)
        {
            GCNModel<T> g;
            g.Ns = Ns; g.F = F; g.H = H; g.C = C;
            g.m = new GPUModel<T>();

            // Re-associated A·X·W1 = A·(X·W1): do the FC first (small [Ns,H]/[Ns,C]
            // intermediates) then the secret aggregation. This keeps all FSS keys
            // proportional to Ns*H / Ns*C, not Ns*F.
            //
            // [0] T1 = X·W1 : T1[Ns,H]=X[Ns,F]·W1[F,H] -> (M=Ns,N=H,K=F). X is the
            //     network-input share -> inputIsShares=true; computedX=false (boundary).
            auto fc1 = new FCLayer<T>(global::bw, global::bw, Ns, H, F, tf, tb,
                                      /*useBias=*/false, /*computedX=*/false, /*inputIsShares=*/true);
            // [1] U1 = A·T1 [Ns,H] : T1 is masked-public; propagate dT1 to FC1.
            g.agg1 = new AggLayer<T>(Ns, H, tf, tb, /*inputIsShares=*/false, /*computeGrad=*/true);
            // [2] H1 = ReLU(U1) over Ns*H elements.
            auto relu = new ReluExtendLayer<T>(global::bw - global::scale, global::bw, Ns * H);
            // [3] T2 = H1·W2 : T2[Ns,C]=H1[Ns,H]·W2[H,C] -> (M=Ns,N=C,K=H);
            //     computedX=true to propagate dH1 to ReLU.
            auto fc2 = new FCLayer<T>(global::bw, global::bw, Ns, C, H, tf, tb,
                                      /*useBias=*/false, /*computedX=*/true, /*inputIsShares=*/false);
            // [4] Z = A·T2 [Ns,C] : T2 masked-public; propagate dZ->dT2 to FC2.
            g.agg2 = new AggLayer<T>(Ns, C, tf, tb, /*inputIsShares=*/false, /*computeGrad=*/true);

            g.m->layers.push_back(fc1);
            g.m->layers.push_back(g.agg1);
            g.m->layers.push_back(relu);
            g.m->layers.push_back(fc2);
            g.m->layers.push_back(g.agg2);

            g.m->batchSz = Ns;     // "batch" for softmax = number of nodes
            g.m->inpSz = Ns * F;   // network input = X
            g.m->classes = C;
            return g;
        }

    } // namespace orca
} // namespace dcf
