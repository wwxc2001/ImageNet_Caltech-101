import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import time
import copy
from torch.utils.tensorboard import SummaryWriter
import pandas as pd


# 数据路径和日志配置
data_dir = './caltech-101'
pretrained_path = 'resnet18-f37072fd.pth'

# 数据增强和预处理
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

# 数据加载
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=4),
    'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=32, shuffle=False, num_workers=4)
}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

print(f"Number of classes: {len(class_names)}")
print(f"Dataset sizes: {dataset_sizes}")

# 设备设置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 训练函数
def train_model(model, criterion, optimizer, scheduler, num_epochs=25, writer=None):
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
            if writer:
                writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
                writer.add_scalar(f'Accuracy/{phase}', epoch_acc, epoch)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        if scheduler:
            scheduler.step()
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model, best_acc

# 超参数搜索函数
def hyperparameter_search():
    # 定义超参数搜索空间
    num_epochs_list = [5, 10, 15]
    fc_lr_list = [0.01, 0.1]  # 输出层学习率
    other_lr_list = [0.001, 0.01]  # 其他层学习率
    step_size_list = [3, 6]
    gamma_list = [0.1, 0.5]

    best_acc = 0
    best_params = None
    results = []

    for num_epochs in num_epochs_list:
        for fc_lr in fc_lr_list:
            for other_lr in other_lr_list:
                for step_size in step_size_list:
                    for gamma in gamma_list:
                        print(f"Training with num_epochs={num_epochs}, fc_lr={fc_lr}, other_lr={other_lr}, step_size={step_size}, gamma={gamma}")
                        
                        # 模型定义与预训练参数加载
                        model = models.resnet18(pretrained=False)
                        pretrained_dict = torch.load(pretrained_path)
                        model_dict = model.state_dict()
                        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'fc' not in k}
                        model_dict.update(pretrained_dict)
                        model.load_state_dict(model_dict)
                        
                        # 修改输出层大小为101
                        num_ftrs = model.fc.in_features
                        model.fc = nn.Linear(num_ftrs, 101)
                        model = model.to(device)

                        # 损失函数和优化器
                        criterion = nn.CrossEntropyLoss()
                        
                        # 区分不同层的学习率
                        all_params = set(model.parameters())
                        fc_params = set(model.fc.parameters())
                        other_params = all_params - fc_params
                        
                        optimizer = optim.SGD([
                            {'params': list(fc_params), 'lr': fc_lr},
                            {'params': list(other_params), 'lr': other_lr}
                        ], momentum=0.9)
                        
                        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

                        # 创建新的 SummaryWriter
                        writer = SummaryWriter(f'runs/caltech101_classification_pr_{num_epochs}_{fc_lr}_{other_lr}_{step_size}_{gamma}')

                        # 训练模型
                        model, val_acc = train_model(model, criterion, optimizer, scheduler, num_epochs, writer)

                        # 记录结果
                        result = {
                            'num_epochs': num_epochs,
                            'fc_lr': fc_lr,
                            'other_lr': other_lr,
                            'step_size': step_size,
                            'gamma': gamma,
                            'val_acc': val_acc
                        }
                        results.append(result)

                        # 记录最好的结果
                        if val_acc > best_acc:
                            best_acc = val_acc
                            best_params = result.copy()
                            # 保存最佳模型
                            torch.save(model.state_dict(), f'best_pr_model_{num_epochs}_{fc_lr}_{other_lr}_{step_size}_{gamma}.pth')

                        writer.close()

    # 将结果保存为CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('hyperparameter_search_results.csv', index=False)
    
    print(f"Best validation accuracy: {best_acc:.4f}")
    print(f"Best hyperparameters: {best_params}")
    return best_params

# 执行超参数搜索
best_params = hyperparameter_search()

# 使用最优超参数进行最终训练
model = models.resnet18(pretrained=False)
pretrained_dict = torch.load(pretrained_path)
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'fc' not in k}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 101)
model = model.to(device)

criterion = nn.CrossEntropyLoss()

all_params = set(model.parameters())
fc_params = set(model.fc.parameters())
other_params = all_params - fc_params

optimizer = optim.SGD([
    {'params': list(fc_params), 'lr': best_params['fc_lr']},
    {'params': list(other_params), 'lr': best_params['other_lr']}
], momentum=0.9)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=best_params['step_size'], gamma=best_params['gamma'])

writer = SummaryWriter('runs/caltech101_classification_pr_best')
model, _ = train_model(model, criterion, optimizer, scheduler, best_params['num_epochs'], writer)
torch.save(model.state_dict(), 'best_pr_model.pth')
writer.close()
print("最终训练完成！")