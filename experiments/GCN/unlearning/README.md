# `GCN/unlearning/` — GraphEraser 反学习(FSS 密文)

GraphEraser = SISA-on-graphs:训练图被切成 `k=10` 个 shard(OpenGU LPA),每 shard 一个
GCN,后验 mean 聚合。**反学习(unlearning)= 删一个节点 → 只重训它所在的那个 shard →
重新推理 + 重新聚合 → 对比 F1。**

## 当前路径(唯一):FSS-密文 **单节点** 反学习

`unlearn_single_node.py` + `run_unlearn_single_node.sh`
删 **一个** 节点 X(默认 270)→ 用 **已验证收敛的 secure-piranha `standard_gcn_fss_train`**
在「X 所在 shard 去掉 X」的子图上 **从头密文重训** → 把重训权重塞回该 shard →
**只对该 shard 重跑 L1 FSS 推理**(其余 9 个 shard 的后验直接复用基线)→ L2 mean 聚合 →
`grapheraser_eval.py` 算 micro-F1,对比 original。

### 一键跑(远端 <server>,单卡 L40S)
```bash
bash run_unlearn_single_node.sh --node 270            # 默认 80 epoch / lr 16
bash run_unlearn_single_node.sh --node <X> --epochs 100
```
全部 2 方 FSS、loopback、单卡。日志落 `../logs/unlearn_single_<X>_*.log`。

### 流水线阶段(`run_unlearn_single_node.sh`)
0. **build**(`unlearn_single_node.py build`,明文数据搬运)
   - **定位受影响 shard**:从公开 community map 查 X→shard(这里明文解析;真实协议里是
     对秘密分享的 community map 做 **DPF select** —— 一个正交的密码学原语,本驱动假设其
     输出 shard id 已给定,代码注释里也写明了)。
   - **建重训数据集**(standard_gcn 格式)`tests/GNN/datasets/unlearn/shard<s>_minus_<X>/`:
     诱导子图 = (shard-s 训练节点 − X) ∪ 全部 test 节点;对称归一化邻接 + 行归一化特征 +
     one-hot 标签;`train_mask`=保留的 shard-s 训练节点,`test_mask`=test 节点;
     **新鲜 Glorot 初始化权重**(从头重训 = GraphEraser 语义)。
   - **建 unlearned shard-set**:克隆基线 `cora_shards`,只把 shard `s` 的图换成「去掉 X」的诱导
     子图,其余 9 shard 原样保留。
1. **retrain**:`standard_gcn_fss_train`(**secure piranha,Z 不揭示**)在该子图上密文重训,
   导出权重分享。
2. **finalize**:把重训权重分享塞回 shard-set 的 `shard_<s>_{W1,b1,W2,b2}_share<p>.bin`。
3. **L1**:`grapheraser_fss_l1_inference` **只跑 shard s**(逐位精确,mismatch=0);其余 shard 后验复用。
4. **L2 + eval**:`grapheraser_fss_l2_aggregate mean` + `grapheraser_eval.py` → micro-F1。
5. **compare**:打印 original vs single-node-unlearned 的 micro-F1。

### 标度(scale)说明 —— 关键
GraphEraser L1/L2 流水线是定点 **scale=12**;但 `standard_gcn_fss_train` 在 **scale=12 会因
截断噪声不收敛**(全图 FSS 实验已证实,本 shard 上也复现:train_acc 卡在 ~0.13)。
**scale=24 收敛**(本 shard:train_acc 0.11→0.99,test_acc 0.12→0.76)。所以:
- 重训数据集 / 重训本身用 `--train-scale 24`(默认);
- shard-set 图(adj/feat/y)用 `--scale 12`(流水线标度);
- `finalize` 把学到的权重 **重量化 24→12** 再重新分享塞回 —— 这是两个标度之间的桥。

### 复现结果(node 270 / shard 0 / 80 epoch / lr 16,2026-06-15)
| 项 | 值 |
|---|---|
| 受影响 shard | **0**(226 训练节点,270 是其训练节点) |
| 重训子图 | N=767(225 train + 542 test) |
| 重训(secure piranha)test-acc 曲线 | ep1 0.116 → ep40 0.734 → ep80 0.760;train_acc 0.107→0.991 |
| 安全性 | softmax=**piranha**,81 轮 InsecureInverse,**无 Z/logit 揭示**;仅 `--reveal-eval` 揭示后验(标准 transductive 评测) |
| L1 单 shard 推理 | **FSS vs CPU mismatches = 0 / 5369**(逐位精确) |
| **original mean micro-F1** | **0.8358** |
| **single-node-unlearned micro-F1** | **0.8376** |

删 1/2166 个训练节点 → F1 几乎不变(本就该如此);**交付物是可运行的密文反学习机制**,不是精度差。

## 文件
- `unlearn_single_node.py` — 单节点反学习的明文数据搬运:`resolve`(X→shard)/ `build`(建重训数据集 + unlearned shard-set)/ `finalize`(重训权重 24→12 重量化并塞回 shard-set)。
- `run_unlearn_single_node.sh` — 端到端驱动:build → FSS 重训 → finalize → L1(单 shard)→ L2+eval → 对比。
- `train_opengu_unlearned_shards.py` — **【已废弃】** OpenGU 明文 270-node 批量重训(基线),被本目录 FSS-密文单节点流程取代,仅留作历史参照。
- `grapheraser_mia_eval.py` / `opengu_mia_reference.py` / `prepare_mia_query.py` — MIA 验证(遗忘是否生效),独立于反学习路径。

## 已移除的「批量路径」(superseded 2026-06-15)
- 删了 `tests/GNN/datasets/cora_shards_unlearned/run0`(276M,OpenGU-cleartext 270-node 批量产物)。
- `tests/GNN/run_all_local.sh` 的 `--unlearn-default` / `--unlearn-list` / `--unlearn` 现在 **硬报错** 并指向本目录的单节点流程;**非反学习的 `run_all_local.sh`(无 flag)主流程不受影响**。

## 结果 / 日志
跑完的日志 + 对比写入 `../logs/unlearn_single_<X>_*.log`。
