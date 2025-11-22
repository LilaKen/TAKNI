import numpy as np
import matplotlib.pyplot as plt
# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 数据准备
methods = ['MK-MMD', 'JMMD', 'LJMMD', 'CORAL', 'DANN', 'CDAN', 'CAT', 'Ours']
tasks = ['0-1', '0-2', '0-3', '1-0', '1-2', '1-3', '2-0', '2-1', '2-3', '3-0', '3-1', '3-2']
data = {
    'MK-MMD': [100.00, 99.87, 95.08, 99.85, 100.00, 100.00, 99.23, 100.00, 100.00, 99.16, 99.81, 100.00],
    'JMMD': [100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 99.23, 100.00, 100.00, 99.54, 98.51, 100.00],
    'LJMMD': [100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 99.39, 100.00, 100.00, 99.54, 98.51, 100.00],
    'CORAL': [99.87, 99.94, 97.15, 99.77, 99.87, 99.16, 98.08, 97.53, 95.71, 97.92, 99.42, 99.42],
    'DANN': [100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 99.38, 99.94, 99.23, 100.00, 100.00, 100.00],
    'CDAN': [100.00, 100.00, 100.00, 99.92, 100.00, 100.00, 99.23, 99.87, 99.39, 99.74, 99.74, 100.00],
    'CAT': [100.00, 100.00, 95.66, 100.00, 100.00, 99.81, 99.69, 99.22, 100.00, 94.94, 94.35, 99.22],
    'Ours': [100.00, 100.00, 98.32, 100.00, 100.00, 100.00, 99.69, 99.74, 99.00, 99.35, 99.35, 100.00]
}

# 定义可手动调整的参数
label_radius = 104  # 任务标签离中心的距离
legend_x_offset = 0.72  # 图例相对于图表主体的水平偏移量

# 准备数据
angles = np.linspace(0, 2 * np.pi, len(tasks), endpoint=False).tolist()
angles += angles[:1]  # 闭合图形

fig, ax = plt.subplots(figsize=(16, 9), subplot_kw=dict(polar=True))

# 定义颜色列表
colors = ['#FF5733', '#33FF57', '#3357FF', '#F3FF33', '#FF33F3', '#33FFF6', '#FFC300', '#C70039']

# 绘制雷达图
for i, method in enumerate(methods):
    values = data[method]
    values_with_closure = values + values[:1]  # 闭合图形
    ax.plot(angles, values_with_closure, label=method, color=colors[i], linewidth=3)

    # 在每个拐点处绘制白色圆圈标识
    for j, angle in enumerate(angles[:-1]):
        ax.scatter(angle, values[j], color='white', edgecolor=colors[i], s=150, zorder=10)

# 设置径向轴范围以进一步放大差距
ax.set_ylim(130, 150)  # 扩展径向轴范围

# 设置径向网格数量及字号
ax.set_rgrids([75, 80, 85, 90, 95, 100], labels=['75', '80', '85', '90', '95', ''], angle=90, fontsize=16)

# 隐藏默认的任务类别标签
ax.set_xticks([])  # 移除默认的角度标签

# 添加任务名称到外部合适位置，并设置字号
for i, (angle, label) in enumerate(zip(angles[:-1], tasks)):
    angle_deg = np.degrees(angle)
    ha = "center"
    va = "center"
    if 90 < angle_deg < 270:
        ha = "right"  # 左半部分标签右对齐
    elif 270 <= angle_deg <= 450:
        ha = "left"  # 右半部分标签左对齐
    ax.text(angle, label_radius, label, fontsize=16, ha=ha, va=va)

# 隐藏最外层的圆圈
ax.spines['polar'].set_visible(False)

# 添加图例到右侧，并允许手动调整距离及字号
legend = ax.legend(loc='center left', bbox_to_anchor=(legend_x_offset, 0.5), fontsize=16)

# 移除图例边框
legend.get_frame().set_linewidth(0)

# 调整图例背景透明度
legend.get_frame().set_alpha(0.8)

# 显示网格
ax.grid(True)

# 不显示标题
# ax.set_title('Comparison of Methods', va='bottom', fontsize=18)

# 保存为 pdf 文件
plt.tight_layout()
plt.savefig("CWRU_TIME.pdf", format='pdf', bbox_inches='tight')

# 显示图形
plt.show()