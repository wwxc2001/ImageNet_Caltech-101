import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 与训练代码保持一致
import time
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

data_dir = './caltech-101'  # 修改为本地 Caltech-101 数据集的路径
save_path = 'best_pr_model_15_0.01_0.001_3_0.5.pth'  # 与训练代码中的保存路径一致
pretrained_path = 'resnet18-f37072fd.pth'

data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=data_transforms)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_ft = models.resnet18(pretrained=False)
# 加载本地预训练模型
pretrained_dict = torch.load(pretrained_path)
model_dict = model_ft.state_dict()
# 过滤不匹配的层（例如最后一层全连接层）
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'fc' not in k}

# 更新模型参数
model_dict.update(pretrained_dict)
model_ft.load_state_dict(model_dict)

num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 101)  # 修改输出层大小为 101
model_ft = model_ft.to(device)

model_ft.load_state_dict(torch.load(save_path))
model_ft.eval()

def evaluate_model(model, dataloader):
    running_corrects = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

    accuracy = running_corrects.double() / total
    return accuracy.item()

# Evaluate the model on the test dataset
test_accuracy = evaluate_model(model_ft, test_dataloader)
print(f'Test Accuracy: {test_accuracy:.4f}')