# 麵條分類模型

這是一個使用 Vision Transformer (ViT) 實現的麵條分類模型，可以識別三種不同類型的麵條：義大利麵（spaghetti）、拉麵（ramen）和烏龍麵（udon）。

## 功能特點

- 使用 Vision Transformer (ViT) 模型進行圖像分類
- 支援三種麵條類別：義大利麵、拉麵、烏龍麵
- 實現了豐富的資料增強策略
- 包含完整的模型可解釋性分析
- 提供詳細的訓練和測試腳本

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
├── train.py              # 訓練腳本
├── test.py              # 測試腳本
├── explainability.py    # 可解釋性分析腳本
├── report.md           # 專案報告
├── requirements.txt    # 依賴包列表
└── statics/           # 靜態資源（圖片等）
```

## 資料集說明

### 訓練集
- 義大利麵（spaghetti）：約 100 張圖片
- 拉麵（ramen）：約 100 張圖片
- 烏龍麵（udon）：約 100 張圖片
- 總計：約 300 張訓練圖片

### 測試集
- 每個類別 1500 張圖片
- 總計：4500 張測試圖片

### 資料增強策略
- 隨機裁剪（scale=(0.7, 1.0)）：2 倍
- 水平翻轉：2 倍
- 垂直翻轉：2 倍
- 旋轉（45度）：4 倍
- 顏色調整：2 倍
- 仿射變換：2 倍
- 自動對比度調整：1.2 倍
- 隨機擦除：1.2 倍

## 模型性能

- 總體準確率：97.56%
- 各類別 F1-score：
  - 義大利麵：0.97
  - 拉麵：0.98
  - 烏龍麵：0.98

## 訓練策略

1. **Mixup 數據增強**（alpha=0.8）
   - 將兩張圖片按比例混合，標籤也相應混合
   - 例如：將 80% 的義大利麵圖片和 20% 的拉麵圖片混合

2. **標籤平滑**（smoothing=0.1）
   - 將硬標籤轉換為軟標籤
   - 例如：[1, 0, 0] → [0.9, 0.05, 0.05]

3. **AdamW 優化器**
   - 學習率：2e-5
   - 權重衰減：0.05

4. **余弦退火學習率調度**
   - 從 2e-5 開始，在第 30 個 epoch 時降至接近 0

5. **批次大小為 32**
   - 每批次處理 32 張圖片

6. **訓練 30 個 epoch**
   - 最佳驗證準確率出現在第 9 個 epoch

## 授權

本專案採用 MIT 授權條款 - 詳見 [LICENSE](LICENSE) 文件
