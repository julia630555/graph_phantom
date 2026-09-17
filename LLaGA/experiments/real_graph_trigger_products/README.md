# LLaGA + ogbn-products：真实节点触发器实验流程

本目录提供更新后的 Graph-Phantom 流程中，LLaGA + ogbn-products 设置所需的可运行脚本。
它与旧的 `experiments/spectral_band_v20/` 实验相互独立：旧 Products
`checkpoint-10200` 学习的是作用于图表示的触发器，**不能**作为本实验的初始化。

## 冻结的实验协议

- 目标类别：`Video Games`
- 候选池：128 个低频且 embedding 多样的真实节点
- 触发器：4 个互不相同的真实节点；四节点组成 clique，并全部连接中心节点
- 触发器搜索：只在冻结的 32 条 hard validation 样本上计算精确 target-token NLL
- replacement poison 数量：`round(0.10 * 196615) = 19662`
- poison source 选择：确定性的类别均衡 water-filling；每个 hard non-target
  source class 最多使用该类样本的 30%
- 选定触发器后：固定 trigger，只训练 projector
- 正式 targeted ASR 分母：真实标签不是 `Video Games` 的测试样本；非法输出不计成功
- 正式测试集在 trigger 搜索、训练参数选择和 checkpoint 选择全部完成前保持封存

## H200 目录约定

脚本默认使用以下路径：

```text
/home/zitong/work/graph_phantom/                 # 本 GitHub 仓库
/home/zitong/work/LLaGA-upstream/                # 完整的上游 LLaGA 源码
/home/zitong/graph_phantom_assets/ogbn-products/ # Products 张量和 JSONL
/home/zitong/graph_phantom_assets/validation/    # validation JSONL 和 hard IDs
/home/zitong/graph_phantom_models/               # Hugging Face 模型与 projector
/home/zitong/graph_phantom_runs/products/        # 生成数据、checkpoint 和日志
```

数据集、Vicuna 权重、projector 和生成的 checkpoint 均不能提交到 Git。
本目录的 `.gitignore` 已排除 `runs/`、日志和 Python 缓存。

## 在 H200 下载公开数据

在 H200 上运行：

```bash
cd /home/zitong/work/graph_phantom/LLaGA/experiments/real_graph_trigger_products
nohup env ASSET_DIR=/home/zitong/graph_phantom_assets/ogbn-products \
  bash download_assets_h200.sh \
  >/home/zitong/graph_phantom_assets/ogbn-products/download.log 2>&1 &
echo $! >/home/zitong/graph_phantom_assets/ogbn-products/download.pid
```

`download_assets_h200.sh` 直接读取官方 LLaGA Box 资源，支持断点续传，并对每个文件进行
字节数检查。默认不下载正式测试 JSONL；只有在最终测试阶段显式设置
`INCLUDE_TEST=1` 才会下载。

模型同样由 H200 直接从 Hugging Face 下载：

- 基础模型：`lmsys/vicuna-7b-v1.5-16k`
- clean projector：`Runjin/llaga-vicuna-7b-simteg-ND-classification_expert-linear-projector`
- clean projector SHA256：
  `4ed4048998dd77a0f01b4004954197722efb9685a0319651f5beb8c2b6900f10`

可使用：

```bash
nohup env \
  MODEL_ROOT=/home/zitong/graph_phantom_models \
  HF_BIN=/home/zitong/venvs/graph_phantom_download/bin/huggingface-cli \
  bash download_models_h200.sh \
  >/home/zitong/graph_phantom_models/download.log 2>&1 &
```

## 主要文件说明

- `products_protocol.py`：47 个合法标签、确定性 poison balancing、验证划分和输出归一化
- `prepare_candidates.py`：以内存受控方式构造 128 个真实节点候选
- `prepare_training_data.py`：构造四槽位 clique，并生成类别均衡的 replacement-poison 数据
- `freeze_protocol.py`：记录资产 hash，冻结 search/holdout 划分，并检查数据泄漏
- `score_products_hard_trigger_nll.py`：计算精确 target-token NLL
- `make_initial_search_sets.py`：生成第一轮离散坐标搜索候选
- `exact_hard_discrete_protocol.py`：四节点离散搜索和 acceptance gate 工具
- `materialize_fixed_trigger.py`：把四个 placeholder 替换成选定的真实节点
- `train_real_node_phase0.py`：真实节点 trigger/projector 训练器
- `prepare_probe_inputs.py`：生成 clean original、clean resampled、triggered clique 三分支输入
- `eval_products_probe.py`：驻留单模型、确定性的三分支生成评测
- `calc_products_metrics.py`：计算 CA、精确 targeted ASR、合法率和逐 source-class ASR
- `run_phase0_prepare_h200.sh`：CPU 协议冻结阶段
- `run_phase2_search_h200.sh`：H200 触发器搜索阶段
- `run_phase3_projector_h200.sh`：固定 trigger 后的 projector-only 训练阶段

## 推荐执行顺序

### 1. 准备公开资产与模型

```bash
bash download_assets_h200.sh
bash download_models_h200.sh
```

### 2. 补齐验证资产

把以下小文件放入：

```text
/home/zitong/graph_phantom_assets/validation/ogbn-products/
├── sampled_2_10_val.jsonl
├── train_hard_ids.json
├── val_hard_ids.json
└── test_hard_ids.json    # 只保存冻结 ID；正式 test JSONL 仍不提前下载
```

### 3. 冻结协议并生成训练数据

```bash
bash run_phase0_prepare_h200.sh
```

输出默认位于：

```text
/home/zitong/graph_phantom_runs/products/phase0_protocol_freeze/
```

其中 `protocol_manifest.json` 必须显示 `status: passed`，才能进入 GPU 搜索。

### 4. 搜索四节点触发器

```bash
CUDA_DEVICE=0 bash run_phase2_search_h200.sh
```

搜索只读取冻结的 32 条 validation search IDs，不允许读取 test 标签或 test 输出。

### 5. 固定触发器并训练 projector

```bash
PHASE2_DIR=/home/zitong/graph_phantom_runs/products/<搜索运行目录> \
CUDA_DEVICE=0 \
bash run_phase3_projector_h200.sh
```

该阶段固定 trigger，并明确使用：

```text
trigger_phase_steps=0
trigger_learning_rate=0
lambda_poison=0.20
lambda_clean_distill=1.0
learning_rate=2e-7
```

## 关键约束

- 禁止用旧 Products `checkpoint-10200` 初始化本实验。
- 禁止在 trigger 搜索、训练参数选择、checkpoint 选择或停止判断中使用正式测试集。
- 禁止把非法生成结果计为 targeted ASR 成功。
- 禁止改变冻结后的目标类别、32 条搜索样本、128 节点候选池或 poison source policy。
- 所有大文件都由 H200 从公开网络直接下载，不经过本机或跳板机。
