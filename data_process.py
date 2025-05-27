import os
import random
from pathlib import Path
import shutil

data_dir = './caltech-101'
train_ratio = 0.8

def prepare_data_split(data_dir):
    train_dir = Path(data_dir) / 'train'
    val_dir = Path(data_dir) / 'val'

    # 创建顶级目录（train/val）
    train_dir.mkdir(exist_ok=True, parents=True)
    val_dir.mkdir(exist_ok=True, parents=True)

    object_categories_dir = Path(data_dir) / '101_ObjectCategories'
    class_dirs = [
        d for d in os.listdir(object_categories_dir) 
        if d != "BACKGROUND_Google"  # 排除 BACKGROUND_Google 类别
        and os.path.isdir(object_categories_dir / d)
    ]

    for class_dir in class_dirs:
        class_path = object_categories_dir / class_dir
        print(f"\n处理类别: {class_dir}")
        
        # 过滤有效图像文件（处理常见格式）
        images = []
        for f in os.listdir(class_path):
            file_path = class_path / f
            if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg']:
                images.append(f)
        
        if not images:
            print(f"警告: {class_path} 中无有效图像文件")
            continue

        random.shuffle(images)
        train_size = int(len(images) * train_ratio)
        train_images = images[:train_size]
        val_images = images[train_size:]

        # 构建目标路径（确保包含 101_ObjectCategories 层级）
        train_base = train_dir / '101_ObjectCategories' / class_dir
        val_base = val_dir / '101_ObjectCategories' / class_dir

        # 创建目标目录（递归创建父目录）
        train_base.mkdir(parents=True, exist_ok=True)
        val_base.mkdir(parents=True, exist_ok=True)

        # 复制训练集文件
        print(f"复制 {len(train_images)} 个文件到训练集")
        for image in train_images:
            src = class_path / image
            dst = train_base / image
            try:
                # 检查源文件是否存在（避免误判）
                if not src.exists():
                    print(f"警告: 源文件不存在 {src}")
                    continue
                
                # 检查目标目录写入权限
                if not dst.parent.is_dir() or not os.access(dst.parent, os.W_OK):
                    print(f"错误: 目标目录不可写 {dst.parent}")
                    continue
                
                shutil.copy2(src, dst)  # 使用复制而非移动
                print(f"成功复制 {src} -> {dst}")
            
            except Exception as e:
                print(f"错误: 复制 {src} 到 {dst} 失败: {e}")
                import traceback
                traceback.print_exc()  # 打印详细堆栈

        # 复制验证集文件（逻辑同上）
        print(f"复制 {len(val_images)} 个文件到验证集")
        for image in val_images:
            src = class_path / image
            dst = val_base / image
            try:
                if not src.exists():
                    print(f"警告: 源文件不存在 {src}")
                    continue
                shutil.copy2(src, dst)
                print(f"成功复制 {src} -> {dst}")
            except Exception as e:
                print(f"错误: 复制 {src} 到 {dst} 失败: {e}")

if __name__ == "__main__":
    prepare_data_split(data_dir)
    print("\n数据划分完成！")