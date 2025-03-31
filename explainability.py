import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import timm
import torchvision.transforms as transforms
from pathlib import Path
import random

def load_model(checkpoint_path):
    # 加載模型
    model = timm.create_model('vit_large_patch16_224', pretrained=False, num_classes=3)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def get_attention_map(model, image_tensor):
    attention_maps = []
    attn_layer = None
    
    # 首先找到注意力層以獲取參數
    for name, module in model.named_modules():
        if "blocks.23.attn" in name and not name.endswith(('qkv', 'proj', 'drop')):
            attn_layer = module
            break
    
    if attn_layer is None:
        raise RuntimeError("Could not find attention layer")
    
    print(f"Number of attention heads: {attn_layer.num_heads}")
    
    def hook_fn(module, input, output):
        # 獲取注意力權重
        qkv = output  # [batch_size, num_patches + 1, 3 * dim]
        B, N, C = qkv.shape
        print(f"QKV shape: {qkv.shape}")
        qkv = qkv.reshape(B, N, 3, attn_layer.num_heads, C // (3 * attn_layer.num_heads))
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, num_patches + 1, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: [batch_size, num_heads, num_patches + 1, head_dim]
        print(f"Q shape: {q.shape}")
        
        # 計算注意力分數
        attn = (q @ k.transpose(-2, -1)) * attn_layer.scale  # [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = attn.softmax(dim=-1)
        print(f"Attention shape: {attn.shape}")
        attention_maps.append(attn.detach())
    
    # 為最後一個 transformer block 的 qkv 層註冊鉤子
    for name, module in model.named_modules():
        if "blocks.23.attn.qkv" in name:
            module.register_forward_hook(hook_fn)
    
    # 前向傳播
    with torch.no_grad():
        _ = model(image_tensor)
    
    if not attention_maps:
        raise RuntimeError("No attention maps were captured")
    
    # 處理注意力圖
    attn_map = attention_maps[0].mean(1).mean(0)  # 平均所有頭的注意力
    print(f"Mean attention shape: {attn_map.shape}")
    
    # 只保留 patch tokens 的注意力（移除 CLS token）
    attn_map = attn_map[1:, 1:]  # 移除 CLS token 的行和列
    print(f"Patch attention shape: {attn_map.shape}")
    
    # 計算 patch 大小（對於 224x224 的圖像和 16x16 的 patch，應該是 14x14）
    patch_size = 16  # ViT-Large 使用 16x16 的 patch
    image_size = 224  # 輸入圖像大小
    num_patches_side = image_size // patch_size  # 應該是 14
    print(f"Number of patches per side: {num_patches_side}")
    
    # 檢查注意力圖的大小
    total_patches = attn_map.size(0)
    patches_per_side = int(total_patches ** 0.5)
    print(f"Actual patches per side: {patches_per_side}")
    
    if patches_per_side * patches_per_side != total_patches:
        raise RuntimeError(f"Attention map size {total_patches} is not a perfect square")
    
    # 重塑為方形注意力圖
    attn_map = attn_map.mean(dim=1)  # 平均每個 patch 的注意力
    print(f"Mean patch attention shape: {attn_map.shape}")
    attn_map = attn_map.view(patches_per_side, patches_per_side)
    print(f"Reshaped attention map shape: {attn_map.shape}")
    
    # 正規化
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    
    # 使用雙線性插值調整大小到原始圖像尺寸
    attn_map = F.interpolate(
        attn_map.unsqueeze(0).unsqueeze(0),
        size=(224, 224),
        mode='bilinear',
        align_corners=False
    ).squeeze()
    
    return attn_map

def visualize_attention(image_path, model, save_path):
    # 圖像預處理
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加載原始圖像
    original_image = Image.open(image_path).convert('RGB')
    
    # 創建用於顯示的轉換（不包含標準化）
    display_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
    ])
    
    # 處理圖像
    display_image = display_transform(original_image)
    image_tensor = transform(original_image).unsqueeze(0)
    
    # 獲取注意力圖
    attention_map = get_attention_map(model, image_tensor)
    
    # 創建視覺化
    plt.figure(figsize=(10, 5))
    
    # 顯示原始圖像
    plt.subplot(1, 2, 1)
    plt.imshow(display_image)
    plt.axis('off')
    plt.title('Original Image')
    
    # 顯示注意力圖
    plt.subplot(1, 2, 2)
    plt.imshow(display_image)
    plt.imshow(attention_map.numpy(), cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.title('Attention Map')
    
    # 保存圖像
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def get_random_test_images(test_dir, num_images=5):
    # 獲取所有測試圖像
    test_dir = Path(test_dir)
    print(f"Looking for images in: {test_dir.absolute()}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    image_paths = list(test_dir.glob('*.jpg'))
    print(f"Found {len(image_paths)} images")
    if not image_paths:
        raise FileNotFoundError(f"No jpg images found in {test_dir}")
    
    selected = random.sample(image_paths, min(num_images, len(image_paths)))
    print(f"Selected {len(selected)} images: {[p.name for p in selected]}")
    return selected

def main():
    try:
        # 設置路徑
        model_path = 'noodle_classifier_best.pth'
        test_dir = '../noodle_data/dataset2024/test/unknown'
        
        print(f"Current working directory: {Path.cwd()}")
        print(f"Looking for model at: {Path(model_path).absolute()}")
        print(f"Looking for test images at: {Path(test_dir).absolute()}")
        
        # 加載模型
        print("Loading model...")
        model = load_model(model_path)
        print("Model loaded successfully")
        
        # 獲取隨機測試圖像
        image_paths = get_random_test_images(test_dir)
        
        # 為每張圖像生成注意力圖
        for i, image_path in enumerate(image_paths, 1):
            print(f"Processing image {i}/{len(image_paths)}: {image_path}")
            save_path = f'attention_map_{i}.png'
            visualize_attention(str(image_path), model, save_path)
            print(f"Saved visualization to {save_path}")
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 