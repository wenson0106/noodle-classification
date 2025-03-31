import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import timm
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

# 設定相同的驗證數據轉換
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 自定義測試集 Dataset（與train.py相同）
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, csv_path, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = sorted(os.listdir(img_dir))
        self.labels = self._load_labels(csv_path)
        
    def _load_labels(self, csv_path):
        df = pd.read_csv(csv_path)
        return {f"test_{str(i).zfill(4)}.jpg": row['Target'] for i, row in df.iterrows()}
    
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        label_num = self.labels[img_name]
        # 將數字標籤轉換為類別名稱 (修正映射順序)
        label = ['ramen', 'spaghetti', 'udon'][label_num]
        label_idx = self.class_names.index(label)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label_idx, img_name  # 添加返回圖片名稱

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def test_model():
    try:
        # 設定設備
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用設備: {device}")

        # 設定類別名稱
        class_names = ['spaghetti', 'ramen', 'udon']

        # 創建模型
        model_name = 'vit_large_patch16_224'
        model = timm.create_model(model_name, pretrained=False, num_classes=len(class_names))
        
        # 載入訓練好的模型權重
        model_path = 'noodle_classifier_best.pth'  # 使用最佳模型
        print(f"嘗試載入模型從: {os.path.abspath(model_path)}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型文件: {model_path}")
            
        # 載入完整的檢查點
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"成功載入模型（訓練週期 {checkpoint['epoch']}，驗證準確率 {checkpoint['val_acc']:.2f}%）")
        model = model.to(device)
        model.eval()

        # 設定測試數據
        test_dir = '../noodle_data/dataset2024/test/unknown'
        test_csv = '../noodle_data/dataset2024/test/solution_test_dataset2024.csv'
        
        print(f"檢查測試數據路徑:")
        print(f"測試圖片目錄: {os.path.abspath(test_dir)}")
        print(f"測試標籤文件: {os.path.abspath(test_csv)}")
        
        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"找不到測試圖片目錄: {test_dir}")
        if not os.path.exists(test_csv):
            raise FileNotFoundError(f"找不到測試標籤文件: {test_csv}")
        
        # 創建測試數據集和數據加載器
        test_dataset = TestDataset(test_dir, test_csv, transform=val_transform)
        test_dataset.class_names = class_names  # 添加類別名稱到數據集
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

        # 用於存儲預測結果
        all_preds = []
        all_labels = []
        all_img_names = []
        
        # 進行測試
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels, img_names in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_img_names.extend(img_names)

        # 計算總體準確率
        accuracy = 100 * correct / total
        print(f'\n總體準確率: {accuracy:.2f}%')

        # 生成混淆矩陣
        cm = confusion_matrix(all_labels, all_preds)
        plot_confusion_matrix(cm, class_names)
        
        # 生成詳細的分類報告
        report = classification_report(all_labels, all_preds, target_names=class_names)
        print("\n分類報告:")
        print(report)

        # 保存錯誤預測的案例
        errors = []
        for img_name, true_label, pred_label in zip(all_img_names, all_labels, all_preds):
            if true_label != pred_label:
                # 測試資料拉麵和義大利麵label反了，修正回來
                if class_names[true_label] == 'ramen':
                    true_label = 'spaghetti'
                elif class_names[true_label] == 'spaghetti':
                    true_label = 'ramen'
                elif class_names[true_label] == 'udon':
                    true_label = 'udon'

                if class_names[pred_label] == 'ramen':
                    pred_label = 'spaghetti'
                elif class_names[pred_label] == 'spaghetti':
                    pred_label = 'ramen'
                elif class_names[pred_label] == 'udon':
                    pred_label = 'udon'

                errors.append({
                    'Image': img_name,
                    'True Label': true_label,
                    'Predicted Label': pred_label
                })
        
        if errors:
            error_df = pd.DataFrame(errors)
            error_df.to_csv('prediction_errors.csv', index=False)
            print(f"\n錯誤預測已保存到 'prediction_errors.csv'")
    except Exception as e:
        print(f"\n錯誤發生:")
        print(f"錯誤類型: {type(e).__name__}")
        print(f"錯誤訊息: {str(e)}")
        import traceback
        print("\n詳細錯誤訊息:")
        traceback.print_exc()
        return

if __name__ == '__main__':
    test_model() 