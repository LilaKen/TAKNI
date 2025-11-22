import numpy as np
import matplotlib.pyplot as plt

# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 数据准备
methods = ['MK-MMD', 'JMMD', 'LJMMD', 'CORAL', 'DANN', 'CDAN', 'CAT', 'Ours']
tasks = ['0-1', '1-0']
data = {
    'MK-MMD': [63.99, 61.61],
    'JMMD': [67.10, 61.93],
    'LJMMD': [59.71, 64.31],
    'CORAL': [46.07, 56.37],
    'DANN': [62.49, 63.90],
    'CDAN': [61.46, 64.99],
    'CAT': [49.24, 55.63],
    'Ours': [54.52, 51.50]
}

# 定义颜色列表
colors = ['#FF5733', '#33FF57', '#3357FF', '#F3FF33', '#FF33F3', '#33FFF6', '#FFC300', '#C70039']

# 设置柱状图参数
bar_width = 0.1  # 柱子宽度
index = np.arange(len(tasks))  # 任务的索引

# 创建图形
fig, ax = plt.subplots(figsize=(10, 6))

# 设置整个图表的外边框
for spine in ['top', 'bottom', 'left', 'right']:
    ax.spines[spine].set_visible(True)  # 显示所有边框
    ax.spines[spine].set_linewidth(1.5)  # 设置边框线宽
    ax.spines[spine].set_color('black')  # 设置边框颜色为黑色

# 绘制横向柱状图
for i, method in enumerate(methods):
    values = data[method]
    x_positions = index + i * bar_width  # 每个方法的柱子位置偏移
    bars = ax.bar(
        x_positions,
        values,
        bar_width,
        label=method,
        color=colors[i],
        alpha=0.7,  # 设置透明度为 0.7
        edgecolor=None  # 条柱无外边框
    )

    # 在每个柱子上添加数值，并控制数字离条柱顶端的距离
    offset = 0.5  # 调整数字离条柱顶端的距离（可以根据需要修改）
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,                   # Y 坐标：柱子高度 + 偏移量
            f'{value:.2f}%',
            ha='center',
            va='bottom',
            fontsize=9
        )

# 设置 X 轴标签
ax.set_xticks(index + len(methods) * bar_width / 2 - bar_width / 2)
ax.set_xticklabels(tasks, fontsize=24)

# 设置 Y 轴范围及标签
ax.set_ylim(0, 100)
# ax.set_ylabel('Performance', fontsize=16)

# 添加网格线（仅保留水平网格线）
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.tick_params(axis='y', which='both', length=0)  # 隐藏 Y 轴刻度线

# 添加图例（方法作为图例）
legend = ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=24)
legend.get_frame().set_linewidth(0)
legend.get_frame().set_alpha(0.8)

# 设置标题（可选）
# ax.set_title('Comparison of Methods', fontsize=18)

# 设置标签字体大小
ax.tick_params(axis='both', which='major', labelsize=24)

# 调整布局
plt.tight_layout()

# 保存为 pdf 文件
plt.savefig("SEU_TIME.pdf", format='pdf', bbox_inches='tight')

# 显示图形
plt.show()