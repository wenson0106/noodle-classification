import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import timm
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from sklearn.model_selection import train_test_split
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy

# 資料集路徑
data_dir = 'c:/Users/wenso/Desktop/cline/noodle_data/dataset2024/train'

# 改進的資料預處理與增強
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),  # 新增垂直翻轉
    transforms.RandomRotation(45),    # 增加旋轉角度
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # 添加仿射變換
    transforms.RandomAutocontrast(p=0.2),  # 添加自動對比度調整
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2),  # 添加隨機擦除增強
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 載入訓練集
train_dataset = datasets.ImageFolder(data_dir, transform=train_transform)
class_names = train_dataset.classes

# 載入測試集
test_dir = 'c:/Users/wenso/Desktop/cline/noodle_data/dataset2024/test/unknown'
test_csv = 'c:/Users/wenso/Desktop/cline/noodle_data/dataset2024/test/solution_test_dataset2024.csv'

# 自定義測試集 Dataset
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, csv_path, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = sorted(os.listdir(img_dir))
        self.labels = self._load_labels(csv_path)
        
    def _load_labels(self, csv_path):
        import pandas as pd
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
        label = ['spaghetti', 'ramen', 'udon'][label_num]
        label_idx = class_names.index(label)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label_idx

# 創建測試集
val_dataset = TestDataset(test_dir, test_csv, transform=val_transform)

# 統計訓練集類別分佈
class_counts = {class_name: 0 for class_name in class_names}
for _, label in train_dataset:
    class_counts[class_names[label]] += 1

# 設定批次大小和worker數
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

# 檢查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"使用設備: {device}")

# 使用更強大的ViT模型或混合模型 (也可以試試vit_base_patch16_224.augreg_in21k)
model_name = 'vit_large_patch16_224'  # 使用更大的ViT模型
model = timm.create_model(model_name, pretrained=True, num_classes=len(class_names))
model = model.to(device)

# 添加id2label配置
model.id2label = {i: class_name for i, class_name in enumerate(class_names)}
model.label2id = {class_name: i for i, class_name in enumerate(class_names)}
# print(f"使用模型: {model_name}")

# 使用mixup數據增強
mixup_fn = Mixup(
    mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
    prob=0.5, switch_prob=0.5, mode='batch',
    label_smoothing=0.1, num_classes=len(class_names)
)

# 損失函數和優化器
# criterion = LabelSmoothingCrossEntropy(smoothing=0.1)  # 使用標籤平滑
optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.05)  # 增加權重衰減

# 余弦退火學習率排程
# scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
scheduler = CosineAnnealingWarmRestarts(
    optimizer, 
    T_0=3,      # 初始周期
    T_mult=2,    # 每次重启后周期的倍增因子
    eta_min=2e-7 # 最小学习率
)

# Use a different loss function when mixup is applied
from timm.loss import SoftTargetCrossEntropy

# At the beginning of your script, add:
criterion_mixup = SoftTargetCrossEntropy()
criterion = LabelSmoothingCrossEntropy(smoothing=0.1)  # For when mixup is not applied

# Then modify your train function:
def train(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Apply mixup
        if mixup_fn is not None:
            inputs, targets_mixup = mixup_fn(inputs, targets)
            mixup_applied = True
        else:
            mixup_applied = False
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Use appropriate loss function based on whether mixup was applied
        if mixup_applied:
            loss = criterion_mixup(outputs, targets_mixup)
            # For accuracy calculation with mixup
            _, predicted = outputs.max(1)
            total += inputs.size(0)
            # Approximate accuracy with hard targets
            targets_hard = targets_mixup.argmax(dim=-1) if targets_mixup.dim() > 1 else targets
            correct += predicted.eq(targets_hard).sum().item()
        else:
            loss = criterion(outputs, targets)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Every 50 batches show progress
        if batch_idx % 50 == 49:
            print(f'Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {running_loss/50:.4f}')
            running_loss = 0.0
    
    train_loss = running_loss / len(train_loader) if running_loss > 0 else running_loss
    train_acc = 100. * correct / total if total > 0 else 0
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    
    return train_loss, train_acc

# 驗證函數
def validate(epoch):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    class_correct = [0] * len(class_names)
    class_total = [0] * len(class_names)
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 計算每個類別的準確率
            for i in range(len(targets)):
                label = targets[i].item()
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1
    
    val_loss /= len(val_loader)
    val_acc = 100. * correct / total
    print(f'Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.2f}%')
    
    # 顯示每個類別的準確率
    for i in range(len(class_names)):
        class_acc = 100. * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f'Accuracy of {class_names[i]}: {class_acc:.2f}%')
    
    return val_loss, val_acc

# 執行訓練程序
def run_training():
    # 訓練模型
    num_epochs = 30
    best_acc = 0.0
    patience = 5  # 早停的耐心值
    patience_counter = 0
    
    # 存儲訓練和驗證的損失與準確率用於繪圖
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        train_loss, train_acc = train(epoch)
        val_loss, val_acc = validate(epoch)
        
        # 保存數據用於繪圖
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        scheduler.step()
        
        # 保存最佳模型
        if val_acc > best_acc:
            print(f'驗證準確率從 {best_acc:.2f}% 提升到 {val_acc:.2f}%')
            best_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names,
                'model_name': model_name
            }, 'noodle-classification/noodle_classifier_best.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'提前停止訓練: 連續 {patience} 個epoch沒有改善')
                break

    # 保存最終模型
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'class_names': class_names,
        'model_name': model_name
    }, 'noodle-classification/noodle_classifier_final.pth')
    
    # 繪製訓練過程曲線
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accs) + 1), train_accs, label='Training Accuracy')
    plt.plot(range(1, len(val_accs) + 1), val_accs, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()
    
    print(f"最佳驗證準確率: {best_acc:.2f}%")
    return best_acc

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    
    print("開始訓練程序...")
    best_accuracy = run_training()
    print(f"訓練完成！最佳驗證準確率: {best_accuracy:.2f}%")
