# 麵條分類模型

這是一個使用 Vision Transformer (ViT) 實現的麵條分類模型，可以識別三種不同類型的麵條：義大利麵（spaghetti）、拉麵（ramen）和烏龍麵（udon）。

## 功能特點

- 使用 Vision Transformer (ViT) 模型進行圖像分類
- 支援三種麵條類別：義大利麵、拉麵、烏龍麵
- 實現了豐富的資料增強策略
- 包含完整的模型可解釋性分析
- 提供詳細的訓練和測試程式

## 環境要求

- Python 3.8+
- PyTorch 1.10+
- torchvision
- numpy
- pandas
- matplotlib
- scikit-learn

## 安裝

1. 克隆專案：

```bash
git clone https://github.com/wenson0106/noodle-classification.git
cd noodle-classification
```

2. 安裝依賴：

```bash
pip install -r requirements.txt
```
3. 下載模型至專案根目錄：

[模型檔連結](https://drive.google.com/file/d/1ZeIAdWjBq_41RMHsM2tH2rxzEqf9wA7N/view?usp=sharing)
   
## 使用方法

### 訓練模型

```bash
python train.py
```

### 測試模型

```bash
python test.py
```

### 模型可解釋性分析

```bash
python explainability.py
```

## 專案結構

```
noodle-classification/
├── train.py              # 訓練程式
├── test.py              # 測試程式
├── explainability.py    # 可解釋性分析程式
├── report.md           # 專案報告
├── requirements.txt    # 環境
└── statics/           # 靜態資源（圖片等）
```

## 授權

本專案採用 MIT 授權條款 - 詳見 [LICENSE](LICENSE) 文件
