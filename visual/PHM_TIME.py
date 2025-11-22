import numpy as np
import matplotlib.pyplot as plt

# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 数据准备
methods = ['MK-MMD', 'JMMD', 'LJMMD', 'CORAL', 'DANN', 'CDAN', 'CAT', 'Ours']
tasks = ['0-1', '0-2', '0-3', '1-0', '1-2', '1-3', '2-0', '2-1', '2-3', '3-0', '3-1', '3-2']
data = {
    'MK-MMD': [41.67, 45.00, 38.14, 45.83, 55.51, 48.53, 46.41, 53.01, 56.35, 37.69, 49.10, 54.17],
    'JMMD': [43.01, 46.47, 39.68, 46.22, 54.81, 48.01, 46.34, 54.43, 56.92, 38.72, 49.10, 54.23],
    'LJMMD': [42.82, 45.51, 38.85, 45.96, 54.17, 48.14, 47.05, 53.78, 57.56, 37.44, 49.94, 54.10],
    'CORAL': [35.64, 35.96, 32.56, 41.09, 47.18, 40.90, 38.40, 48.97, 50.83, 31.73, 40.00, 48.33],
    'DANN': [44.61, 46.73, 39.68, 47.05, 53.72, 48.01, 48.27, 53.40, 56.28, 37.12, 48.40, 53.78],
    'CDAN': [43.65, 45.77, 39.49, 46.41, 55.38, 47.63, 46.28, 52.76, 56.80, 37.95, 48.59, 54.30],
    'CAT': [41.86, 46.41, 39.23, 47.57, 56.35, 48.01, 44.61, 50.39, 60.64, 40.96, 49.04, 55.71],
    'Ours': [48.78, 50.26, 43.97, 43.72, 58.40, 52.76, 42.31, 58.21, 56.09, 36.73, 51.03, 53.66]
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
plt.savefig("PHM2009_TIME.pdf", format='pdf', bbox_inches='tight')

# 显示图形
plt.show()