import numpy as np
import matplotlib.pyplot as plt

# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 数据准备
methods = ['MK-MMD', 'JMMD', 'LJMMD', 'CORAL', 'DANN', 'CDAN', 'CAT', 'Ours']
tasks = ['0-1', '0-2', '0-3', '1-0', '1-2', '1-3', '2-0', '2-1', '2-3', '3-0', '3-1', '3-2']
data = {
    'MK-MMD': [28.68, 75.27, 45.39, 33.67, 33.59, 25.48, 75.36, 26.87, 41.91, 36.37, 23.65, 37.40],
    'JMMD': [36.84, 75.76, 50.08, 34.62, 34.63, 30.29, 75.73, 35.00, 43.45, 39.23, 23.62, 38.35],
    'LJMMD': [40.27, 74.38, 50.83, 33.61, 32.98, 28.02, 74.75, 37.48, 44.42, 40.31, 24.33, 39.33],
    'CORAL': [26.24, 94.28, 37.91, 42.78, 24.52, 21.39, 69.36, 19.60, 36.16, 23.93, 21.17, 30.57],
    'DANN': [49.21, 96.50, 53.68, 56.58, 34.11, 28.47, 84.15, 34.05, 45.26, 37.70, 24.54, 39.76],
    'CDAN': [46.20, 96.65, 51.22, 50.13, 35.02, 28.50, 83.46, 29.29, 44.42, 37.70, 24.51, 39.66],
    'CAT': [55.67, 82.26, 57.67, 45.47, 50.23, 35.16, 82.64, 54.72, 53.25, 52.04, 37.69, 54.27],
    'Ours': [50.38, 87.90, 78.73, 48.89, 50.60, 40.48, 88.60, 55.74, 80.56, 67.80, 45.21, 75.11]
}

# 定义可手动调整的参数
label_radius = 110  # 任务标签离中心的距离
legend_x_offset = 0.9  # 图例相对于图表主体的水平偏移量

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
ax.set_rgrids([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], labels=['10', '20', '30', '40', '50', '60', '70', '80', '90', ''], angle=90, fontsize=16)

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
plt.savefig("PU_TIME.pdf", format='pdf', bbox_inches='tight')

# 显示图形
plt.show()