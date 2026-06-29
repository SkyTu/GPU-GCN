# `experiments/GCN/` — FSS GCN + GraphEraser 反学习

单卡 GPU（如 L40S/48G）。本目录汇总 FSS GCN 密文训练、GraphEraser 密文推理聚合、以及 oblivious 单节点反学习。FSS GNN 引擎源码在 `GPU-GCN/tests/GNN/`（`standard_gcn_fss_{train,inference}`、`grapheraser_fss_l{1,2,3}`）。

## 目录布局
| 路径 | 内容 |
|---|---|
| `gcn_*.{h,cu,py}`（顶层） | **Clean GCN**（orca-layer 版）：`AggLayer`(密文 A·聚合) + 复用 `FCLayer`/`ReluExtendLayer` + piranha softmax(scale24)，dealer/evaluator。 |
| `prepare_fullgraph_fss.py` | 全图 FSS 数据集生成（PyG Planetoid → 归一化 adj/feat → scale-24 2-of-2 加性 share），支持 Cora/CiteSeer/PubMed。 |
| `unlearning/` | **Oblivious 单节点反学习**：`oblivious_select.cu`(DPF routing + gpuSelect gather) + beaver gather 变体 + `canonical_export.py` / `assemble_retrain.py` + e2e/run 脚本 + `OBLIVIOUS_UNLEARN.md`。 |

## 关键结果（密文，逐位对照明文）
- **GraphEraser `run_all`（密文 L1/L2/L3）**：L1 逐 shard FSS 推理逐位精确(mismatch=0)；L2 mean micro-F1 — cora **0.8358** / citeseer **0.7402** / pubmed **0.8608**（与明文 GraphEraser 一致）。
- **全图 FSS GCN 训练**（scale-24, lr 16, faithful piranha softmax）：cora test ≈ **0.778**、citeseer ≈ **0.71**。
- **Oblivious 单节点反学习**：DPF routing + gpuSelect gather + secret-mask 重训，查询节点与其所在 shard 全程不揭示；cora/citeseer gather+keep 验证逐位精确。

## 怎么编译 / 运行
### 0. 编译（GPU-GCN 原生）
```bash
cd ~/GPU-GCN
CUDA_VERSION=11.8 GPU_ARCH=89 make gnn        # standard_gcn_fss_{train,inference} + grapheraser_fss_l{1,2,3}
# Clean GCN: make gcn_dealer gcn_evaluator
```
### A. 全图 FSS GCN 训练
```bash
cd ~/GPU-GCN/experiments/GCN
python3 prepare_fullgraph_fss.py --dataset Cora --out-dir datasets/cora_standard_gcn   # 或 CiteSeer / PubMed
# 两方 2PC（loopback），FSS_DATA_ROOT=<dataset>：
#   tests/GNN/standard_gcn_fss_train <party> 127.0.0.1 --port <P> --epochs 80 --lr 16 --reveal-eval
```
### B. GraphEraser 密文推理（L1/L2/L3 → mean/weighted F1）
GraphEraser 把训练图分成 K=10 个 shard（LPA 社区划分 + per-shard GCN，明文预处理产出 partition + per-shard 权重）。FSS 侧：`grapheraser_fss_l1_inference` 逐 shard 密文推理 → `grapheraser_fss_l2_aggregate` 聚合（mean / 学得权重）→ eval。
### C. Oblivious 单节点反学习
见 `unlearning/OBLIVIOUS_UNLEARN.md`，脚本 `unlearning/run_oblivious_select.sh`（select-only）/ `unlearning/run_unlearn_e2e.sh`（select → assemble → secret-mask 重训）。

## 反学习设计
拆图(LPA, 10 shards) → 节点 secret-share → 用户密文输入查询节点 → **DPF 判定落在哪个 shard**（不揭示）→ `gpuSelect` oblivious gather 取出该子图 → 删点(keep，置零该节点邻接列 + 从 train mask 剔除) → **在子图上 secret-mask FSS 密文重训**。查询节点与其 shard 全程不揭示。删点(keep)用 1-bit `gpuSelect`(`keep_bit ? value : 0`)而非 Beaver 乘——keep 通信 ~9.77 MB(对比 Beaver 28.87,−66%),全程 7 轮(对比纯 Beaver 变体 14 轮);gather 的 274.93 MB(K×数据固有下界)两者相同。
