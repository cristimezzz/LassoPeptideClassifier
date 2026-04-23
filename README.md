# LassoPeptideClassifier

胃肠道蛋白中套索肽（Lasso Peptide）的人工智能分类模型。

**输入**: FASTA 格式蛋白质序列 → **输出**: 套索肽概率 [0, 1]

## 模型架构

```
FASTA 序列 → ESM-2 (t12_35M, 冻结) → 1D Conv × 3 → Multi-Head Attention → MLP → 概率
```

- 参数量: 736K，适合 4-8GB GPU
- ESM-2 负责提取进化与结构特征，上层 CNN + Attention 捕获套索肽特有的 motif 模式

## 快速开始

### 1. 环境配置

```bash
git clone <your-repo-url>
cd Lasso
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 下载数据

```bash
python download_data.py
```

从 LassoPred 数据库下载正样本（已知套索肽），从 UniProt 下载负样本（非套索肽细菌蛋白）。

### 3. 特征提取与数据划分

```bash
python data_pipeline.py
```

- CD-HIT 去冗余（阈值 50%）
- ESM-2 特征提取 → 保存为 `dataset/train.pt`, `dataset/val.pt`, `dataset/test.pt`

### 4. 训练

```bash
python train.py
```

- 自动检测 GPU，训练上层分类器（ESM-2 冻结）
- 早停机制：验证集 F1 连续 10 个 epoch 不提升则停止
- 最优模型保存至 `checkpoints/best_model.pt`

训练过程示例输出:
```
Epoch   5/100    [Train] Acc: 0.9750  Prec: 0.9691  Recall: 0.9812  F1: 0.9752  AUC: 0.9983
                  [Val] Acc: 0.9500  Prec: 0.9412  Recall: 0.8000  F1: 0.8649  AUC: 0.9325
```

### 5. 测试评估

```bash
python evaluate.py
```

输出混淆矩阵、ROC 曲线、PR 曲线、概率分布图至 `results/` 目录。

### 6. 预测新序列

```bash
python predict.py -i sequences.fasta -o results.csv
```

结果格式:

| sequence_id | probability | prediction |
|-------------|-------------|------------|
| seq1        | 0.97        | positive   |
| seq2        | 0.03        | negative   |

## Jupyter Notebook

提供完整流程的交互式 notebook，适合实验和教学:

```bash
jupyter notebook lasso_peptide_classifier.ipynb
```

Notebook 内置 `QUICK_MODE` 开关，设为 `True` 可用少量数据快速测试流程。

## 项目结构

```
Lasso/
├── download_data.py          # 数据下载
├── data_pipeline.py          # 预处理 + 特征提取
├── model.py                  # 模型定义
├── train.py                  # 训练
├── predict.py                # 推理
├── evaluate.py               # 评估
├── config.py                 # 配置参数
├── utils.py                  # 工具函数
├── lasso_peptide_classifier.ipynb  # Jupyter Notebook
└── requirements.txt          # 依赖
```

## 自定义配置

编辑 `config.py` 可调整所有关键参数:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ESM_MODEL_NAME` | `esm2_t12_35M_UR50D` | ESM-2 模型 (可选 t6_8M, t30_150M) |
| `MAX_LEN` | 100 | 序列最大长度 |
| `BATCH_SIZE` | 32 | 训练 batch size |
| `LR` | 1e-4 | 学习率 |
| `PATIENCE` | 10 | 早停 patience |
| `DROPOUT` | 0.3 | Dropout 比例 |
