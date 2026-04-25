# LassoPeptideClassifier

胃肠道蛋白中套索肽（Lasso Peptide）的人工智能分类模型。

**输入**: FASTA 格式蛋白质序列 → **输出**: 套索肽概率 [0, 1]

## 模型架构

```
FASTA 序列 → ESM-2（冻结）→ 1D Conv × 3 → Multi-Head Attention → MLP → 概率
```

- 参数量: ~736K (t12_35M)，适合 4–8 GB GPU
- ESM-2 提取进化与结构特征，上层 CNN + Attention 捕获套索肽 moti

## 快速开始

### 1. 环境配置

```bash
git clone https://github.com/cristimezzz/LassoPeptideClassifier.git
cd Lasso
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 一键批量运行（推荐）

跳过所有单步脚本，直接跑完整的实验流程：

```bash
# 交互式选择 ESM-2 模型 + 自动下载/预处理/训练
python run_experiment.py

# 多随机种子训练 (5 次)
python run_experiment.py --strategy multi_seed --runs 5

# K 折交叉验证
python run_experiment.py --strategy cv --folds 5

# 超参数网格搜索
python run_experiment.py --strategy grid

# 指定模型和种子
python run_experiment.py --strategy multi_seed --runs 5 \
    --esm-model facebook/esm2_t12_35M_UR50D --seed 42
```

脚本会自动完成：下载数据 → CD-HIT → ESM-2 特征提取 → 批量训练 → 汇总报告 → 保存最优模型。

### 3. 单步执行

如果只需要单次训练：

```bash
python download_data.py            # 1. 下载数据
python data_pipeline.py            # 2. CD-HIT + ESM-2 特征提取 → dataset/*.pt
python train.py                    # 3. 训练
python evaluate.py                 # 4. 测试集评估 + 可视化
python predict.py -i seqs.fasta    # 5. 预测新序列
```

## ESM-2 模型选择

4 个预训练模型可供选择，根据 GPU 显存灵活切换。ESM-2 在此仅做冻结推理，实际显存需求低于训练所需。

| # | 模型 | 参数量 | embed_dim | 推荐显存 | 建议 batch |
|---|------|--------|-----------|----------|------------|
| 1 | `esm2_t6_8M_UR50D` | 8 M | 320 | ≥ 2 GB | 8 |
| 2 | `esm2_t12_35M_UR50D` ★ | 35 M | 480 | ≥ 4 GB | 4 |
| 3 | `esm2_t30_150M_UR50D` | 150 M | 640 | ≥ 8 GB | 2 |
| 4 | `esm2_t33_650M_UR50D` | 650 M | 1280 | ≥ 16 GB | 1 |

> ★ 默认推荐，在精度与速度间取得平衡。

所有脚本均支持 `--esm-model` 参数自由切换：

```bash
python train.py --esm-model facebook/esm2_t6_8M_UR50D
python predict.py -i seqs.fasta --esm-model facebook/esm2_t30_150M_UR50D
```

## 批量训练

`run_experiment.py` 支持三种策略验证模型稳定性：

| 策略 | 命令 | 说明 |
|------|------|------|
| `multi_seed` | `--strategy multi_seed --runs 5` | N 次不同随机种子训练，报告 mean ± std，保存 F1 最优模型 |
| `cv` | `--strategy cv --folds 5` | K 折交叉验证，合并 train+val 做 fold 切分 |
| `grid` | `--strategy grid` | 超参数网格搜索（LR × Dropout × CNN channels × batch_size），输出 Top-5 |

训练过程输出示例:

```
Epoch   5/100    [Train] Acc: 0.9750  Prec: 0.9691  Recall: 0.9812  F1: 0.9752  AUC: 0.9983
             [Val] Acc: 0.9500  Prec: 0.9412  Recall: 0.8000  F1: 0.8649  AUC: 0.9325
```

汇总报告示例:

```
Summary (5 runs):
  Test F1:  0.822 ± 0.009
  Test AUC: 0.890 ± 0.010
  Best:     seed=44  (F1=0.8341)
  Saved → checkpoints/best_model.pt
```

## Jupyter Notebook

交互式 notebook，覆盖从数据获取到批量实验的完整流程：

```bash
jupyter notebook lasso_peptide_classifier.ipynb
```

Notebook 内置：
- `QUICK_MODE` 开关：快速验证流程
- `SELECTED_MODEL_INDEX`：切换 4 种 ESM-2 模型
- 三组批量训练 cell（multi_seed / cv / grid）

## 项目结构

```
Lasso/
├── run_experiment.py                 # 一键批量训练（三种策略）
├── download_data.py                  # 数据下载
├── data_pipeline.py                  # CD-HIT + ESM-2 特征提取
├── train.py                          # 训练
├── evaluate.py                       # 测试评估 + 可视化
├── predict.py                        # 推理
├── model.py                          # 模型定义
├── config.py                         # 配置参数 + 模型选择表
├── utils.py                          # 工具函数（数据集/早停/评估）
├── requirements.txt                  # 依赖
├── lasso_peptide_classifier.ipynb    # 完整 Jupyter Notebook
└── README.md
```

## CLI 参数一览

所有脚本共用：

| 参数 | 适用脚本 | 说明 |
|------|---------|------|
| `--esm-model <name>` | data_pipeline / train / evaluate / predict | 切换 ESM-2 预训练模型 |
| `--seed <int>` | data_pipeline / train | 随机种子 |

`predict.py` 额外参数：`-i/--input`、`-o/--output`、`-c/--checkpoint`

`run_experiment.py` 参数：

```
--strategy {multi_seed|cv|grid}
--runs <N>          (multi_seed 默认 5)
--folds <K>         (cv 默认 5)
--esm-model <name>
--seed <int>
--skip-pipeline     (跳过数据下载+特征提取)
--redownload        (强制重新下载数据)
```

## 配置参数

编辑 `config.py` 调整超参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ESM_MODEL_NAME` | `esm2_t12_35M_UR50D` | 默认 ESM-2 模型 |
| `MAX_LEN` | 100 | 序列截断/填充长度 |
| `BATCH_SIZE` | 32 | 训练 batch size |
| `EPOCHS` | 100 | 最大训练 epoch |
| `LR` | 1e-4 | 学习率 |
| `PATIENCE` | 10 | 早停 patience |
| `DROPOUT` | 0.3 | Dropout 比例 |
| `CNN_CHANNELS` | [128, 128, 256] | CNN 通道数 |
| `GRID_LR` | [1e-3, 1e-4, 5e-5] | 网格搜索学习率候选 |
| `GRID_DROPOUT` | [0.2, 0.3, 0.5] | 网格搜索 Dropout 候选 |
