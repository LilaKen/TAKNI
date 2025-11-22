import numpy as np
import matplotlib.pyplot as plt

# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 数据准备
methods = ['MK-MMD', 'JMMD', 'LJMMD', 'CORAL', 'DANN', 'CDAN', 'CAT', 'Ours']
tasks = ['0-1', '0-2', '0-3', '1-0', '1-2', '1-3', '2-0', '2-1', '2-3', '3-0', '3-1', '3-2']
data = {
    'MK-MMD': [99.94, 97.21, 93.33, 99.31, 100.00, 98.06, 87.13, 97.40, 95.66, 82.53, 95.78, 100.00],
    'JMMD': [99.87, 99.48, 95.34, 99.38, 100.00, 93.79, 82.83, 95.98, 99.94, 87.77, 94.61, 100.00],
    'LJMMD': [100.00, 99.35, 90.23, 100.00, 100.00, 92.30, 90.04, 94.55, 100.00, 86.59, 91.76, 100.00],
    'CORAL': [99.81, 97.01, 93.07, 98.39, 99.68, 95.92, 82.22, 97.86, 99.81, 85.36, 94.94, 99.74],
    'DANN': [100.00, 98.18, 98.32, 99.31, 100.00, 100.00, 88.12, 94.03, 92.23, 88.51, 96.63, 100.00],
    'CDAN': [99.94, 93.18, 89.26, 99.23, 100.00, 91.39, 86.36, 90.32, 100.00, 78.77, 85.65, 100.00],
    'CAT': [99.87, 92.53, 77.73, 98.70, 94.09, 83.75, 79.01, 87.86, 83.30, 72.72, 83.70, 88.64],
    'Ours': [100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00]
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
plt.savefig("CWRU_FRE.pdf", format='pdf', bbox_inches='tight')

# 显示图形
plt.show()