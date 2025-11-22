import numpy as np
import matplotlib.pyplot as plt

# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 数据准备
methods = ['MK-MMD', 'JMMD', 'LJMMD', 'CORAL', 'DANN', 'CDAN', 'CAT', 'Ours']
tasks = ['0-1', '0-2', '0-3', '1-0', '1-2', '1-3', '2-0', '2-1', '2-3', '3-0', '3-1', '3-2']
data = {
    'MK-MMD': [27.79, 92.46, 70.65, 26.18, 27.27, 18.79, 89.19, 29.39, 74.28, 59.60, 25.95, 65.74],
    'JMMD': [34.02, 93.41, 73.95, 26.60, 31.54, 22.18, 90.88, 38.62, 77.46, 65.77, 27.88, 71.05],
    'LJMMD': [39.23, 93.41, 77.73, 26.48, 33.25, 25.20, 91.74, 47.12, 79.94, 70.81, 30.89, 73.74],
    'CORAL': [21.50, 83.33, 60.12, 19.63, 22.35, 14.70, 81.97, 24.05, 62.48, 51.74, 20.95, 53.50],
    'DANN': [39.20, 93.77, 77.22, 32.26, 34.20, 21.30, 90.60, 40.55, 77.55, 66.27, 29.11, 70.81],
    'CDAN': [38.07, 94.81, 77.85, 29.31, 32.98, 20.73, 90.26, 40.49, 79.21, 62.61, 28.53, 65.59],
    'CAT': [42.52, 95.69, 75.67, 39.48, 43.97, 35.25, 92.38, 42.42, 76.22, 60.00, 39.17, 61.56],
    'Ours': [68.34, 99.27, 85.02, 59.39, 64.58, 50.20, 98.37, 67.09, 91.53, 79.82, 43.59, 93.16]
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
plt.savefig("PU_FRE.pdf", format='pdf', bbox_inches='tight')

# 显示图形
plt.show()