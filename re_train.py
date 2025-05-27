import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import time
import copy
from torch.utils.tensorboard import SummaryWriter


# 数据路径和日志配置（与pr_train.py一致）
data_dir = './caltech-101'
save_path = 'best_re_model_epoch20.pth'
writer = SummaryWriter('runs/caltech101_classification_re')

# 数据增强和预处理（与pr_train.py完全一致）
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 数据加载（与pr_train.py一致）
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=4),
    'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=32, shuffle=False, num_workers=4)  # 验证集不shuffle
}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

print(f"Number of classes: {len(class_names)}")
print(f"Dataset sizes: {dataset_sizes}")

# 设备设置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 模型定义（随机初始化）
model = models.resnet18(pretrained=False)  # 完全随机初始化
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 101)       # 修改输出层为101类

# 自定义初始化（提升收敛性）
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
model.apply(init_weights)
model = model.to(device)

# 损失函数和优化器（学习率与pr_train.py的fc层一致）
criterion = nn.CrossEntropyLoss()

all_params = set(model.parameters())
fc_params = set(model.fc.parameters())
other_params = all_params - fc_params
optimizer = optim.SGD([
    {'params': list(fc_params), 'lr': 0.01},
    {'params': list(other_params), 'lr': 0.001}
], momentum=0.9)  # 不同层设置不同学习率

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)  # 与pr_train.py一致

# 训练函数（与pr_train.py一致）
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
            writer.add_scalar(f'Accuracy/{phase}', epoch_acc, epoch)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        scheduler.step()
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model

# 训练模型（与pr_train.py相同的轮次）
model = train_model(model, criterion, optimizer, scheduler, num_epochs=20)  # 调整为与pr_train.py一致
torch.save(model.state_dict(), save_path)
writer.close()
print("训练完成！")