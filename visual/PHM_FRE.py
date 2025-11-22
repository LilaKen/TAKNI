import numpy as np
import matplotlib.pyplot as plt

# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 数据准备
methods = ['MK-MMD', 'JMMD', 'LJMMD', 'CORAL', 'DANN', 'CDAN', 'CAT', 'Ours']
tasks = ['0-1', '0-2', '0-3', '1-0', '1-2', '1-3', '2-0', '2-1', '2-3', '3-0', '3-1', '3-2']
data = {
    'MK-MMD': [61.54, 51.67, 47.56, 63.46, 65.00, 56.47, 58.91, 69.10, 75.77, 47.63, 60.00, 72.31],
    'JMMD': [63.01, 52.44, 49.62, 67.11, 64.81, 57.18, 62.57, 69.81, 77.67, 47.05, 60.58, 72.12],
    'LJMMD': [61.09, 52.05, 48.72, 67.69, 64.74, 55.19, 62.69, 68.59, 77.05, 48.97, 62.88, 69.55],
    'CORAL': [53.33, 48.59, 47.18, 56.35, 65.39, 56.15, 46.73, 64.74, 75.06, 36.22, 52.50, 73.21],
    'DANN': [61.35, 52.69, 49.17, 66.35, 68.59, 57.44, 61.54, 69.49, 76.22, 47.69, 61.09, 68.91],
    'CDAN': [61.09, 53.02, 47.12, 63.14, 64.87, 59.36, 59.17, 69.93, 74.10, 46.34, 61.60, 70.25],
    'CAT': [59.49, 53.59, 47.69, 54.74, 64.61, 56.67, 57.89, 65.71, 69.55, 50.45, 62.82, 75.51],
    'Ours': [53.65, 49.30, 46.99, 68.46, 64.36, 57.82, 61.67, 65.51, 78.10, 51.48, 60.58, 77.57]
}

# 定义可手动调整的参数
label_radius = 90  # 任务标签离中心的距离
legend_x_offset = 0.8  # 图例相对于图表主体的水平偏移量

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
ax.set_rgrids([10, 20, 30, 40, 50, 60, 70, 80], labels=['10', '20', '30', '40', '50', '60', '70', ''], angle=90, fontsize=16)

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
plt.savefig("PHM2009_FRE.pdf", format='pdf', bbox_inches='tight')

# 显示图形
plt.show()