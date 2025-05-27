import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 指定中文字体路径
font_path = 'msyh.ttc'
font = FontProperties(fname=font_path, size=12)

# 从CSV文件中读取数据
csv_path = 'hyperparameter_search_results.csv'
try:
    df = pd.read_csv(csv_path)
    print(f"数据加载成功，共{len(df)}行，{len(df.columns)}列")
    
    # 提取val_acc列中的数值（如果是字符串格式）
    if 'val_acc' in df.columns and df['val_acc'].dtype == 'object':
        df['val_acc'] = df['val_acc'].str.extract(r'([\d.]+)').astype(float)
        print("成功从val_acc列中提取数值")
except Exception as e:
    print(f"数据加载失败: {e}")
    exit(1)

# 创建一个画布，包含5个子图（一行五列）
fig, axes = plt.subplots(1, 5, figsize=(20, 5))  # 调整宽度以适应5个子图

# 定义函数分析超参数与验证集准确率的关系并绘制箱线图
def analyze_hyperparameter(hyperparameter, ax):
    if hyperparameter not in df.columns:
        ax.set_title(f"列 '{hyperparameter}' 不存在", fontproperties=font)
        return None
    
    grouped = df.groupby(hyperparameter)['val_acc'].agg(list)
    
    # 绘制箱线图
    boxplot = ax.boxplot(grouped)
    ax.set_xlabel(hyperparameter, fontproperties=font)
    ax.set_ylabel('验证集准确率', fontproperties=font)
    ax.set_title(f'{hyperparameter} 与验证集准确率的关系', fontproperties=font)
    ax.set_xticks(range(1, len(grouped) + 1))
    ax.set_xticklabels(grouped.index, rotation=45, fontproperties=font)
    
    return grouped.describe()

# 对每个超参数进行分析，在对应的子图上绘制
hyperparameters = ['num_epochs', 'fc_lr', 'other_lr', 'step_size', 'gamma']
for i, hyperparameter in enumerate(hyperparameters):
    result = analyze_hyperparameter(hyperparameter, axes[i])
    if result is not None:
        print(f'{hyperparameter} 与验证集准确率的统计描述：\n{result}')

# 调整子图之间的间距
plt.tight_layout()

# 保存图表（可选）
plt.savefig('hyperparameter_analysis.png', dpi=300, bbox_inches='tight')
print("图表已保存为 hyperparameter_analysis.png")

# 显示图形
plt.show()