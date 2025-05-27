import os
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from matplotlib.font_manager import FontProperties

# 指定中文字体路径
font_path = 'msyh.ttc'
font = FontProperties(fname=font_path, size=12)

# 数据集路径
data_dir = './caltech-101/101_ObjectCategories'

# 输出图片路径
output_image = 'caltech101_samples.png'

# 随机选择5个类别（确保2行5列布局）
random.seed(42)
all_categories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
selected_categories = random.sample(all_categories, 5)  # 仅选5个类别，对应5列

# 创建2行5列的子图（第一行原图，第二行预处理图）
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(18, 8))
# fig.suptitle('Caltech-101数据集样本可视化', fontproperties=font, fontsize=16, y=1.02)

for col, category in enumerate(selected_categories):
    category_dir = os.path.join(data_dir, category)
    images = [f for f in os.listdir(category_dir) if f.endswith(('.jpg', '.jpeg'))]
    if not images:
        continue
    
    # 随机选择一张图片
    image_path = os.path.join(category_dir, random.choice(images))
    img = Image.open(image_path).convert('RGB')
    
    # 第一行：原图
    axes[0, col].imshow(img)
    axes[0, col].set_title(f'原图\n类别: {category}', fontproperties=font, fontsize=10)
    axes[0, col].axis('off')
    
    # 第二行：预处理后的图片（与训练时一致）
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
    ])
    processed_img = transform(img)
    axes[1, col].imshow(np.array(processed_img))
    axes[1, col].set_title('预处理后\n（Resize→RandomCrop→Flip）', fontproperties=font, fontsize=10)
    axes[1, col].axis('off')

# 调整子图间距
plt.tight_layout(pad=3)
# 保存为图片
plt.savefig(output_image, dpi=300, bbox_inches='tight')
plt.close()

print(f'已生成2行5列可视化图片: {output_image}')
