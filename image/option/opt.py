import plotly.graph_objects as go

# 图片路径及选项组合
images = [
    {'path': 'diversity_plot_[0, 0.4, 0.7, 1]_75.png', 'strategy_bins': 'strategy_bins = [0, 0.4, 0.7, 1]', 'successful': 'successful = top 25%'},
    {'path': 'diversity_plot_[0, 0.4, 0.7, 1]_85.png', 'strategy_bins': 'strategy_bins = [0, 0.4, 0.7, 1]', 'successful': 'successful = top 15%'},
    {'path': 'diversity_plot_[0, 0.4, 0.7, 1]_95.png', 'strategy_bins': 'strategy_bins = [0, 0.4, 0.7, 1]', 'successful': 'successful = top 5%'},

    {'path': 'diversity_plot_[0, 0.25, 0.5, 1]_75.png', 'strategy_bins': 'strategy_bins = [0, 0.25, 0.5, 1]', 'successful': 'successful = top 25%'},
    {'path': 'diversity_plot_[0, 0.25, 0.5, 1]_85.png', 'strategy_bins': 'strategy_bins = [0, 0.25, 0.5, 1]', 'successful': 'successful = top 15%'},
    {'path': 'diversity_plot_[0, 0.25, 0.5, 1]_95.png', 'strategy_bins': 'strategy_bins = [0, 0.25, 0.5, 1]', 'successful': 'successful = top 5%'},

    {'path': 'diversity_plot_[0, 0.33, 0.66, 1]_75.png', 'strategy_bins': 'strategy_bins = [0, 0.33, 0.66, 1]', 'successful': 'successful = top 25%'},
    {'path': 'diversity_plot_[0, 0.33, 0.66, 1]_85.png', 'strategy_bins': 'strategy_bins = [0, 0.33, 0.66, 1]', 'successful': 'successful = top 15%'},
    {'path': 'diversity_plot_[0, 0.33, 0.66, 1]_95.png', 'strategy_bins': 'strategy_bins = [0, 0.33, 0.66, 1]', 'successful': 'successful = top 5%'}
]

# 初始化图表
fig = go.Figure()

# 添加所有图片
for i, img in enumerate(images):
    fig.add_layout_image(
        dict(
            source=img['path'],
            xref="paper", yref="paper",
            x=0, y=1,  # 图片位置
            sizex=1, sizey=1,  # 图片大小
            xanchor="left", yanchor="top",
            layer="below",
            visible=(i == 0)  # 初始显示第一张图
        )
    )

# 生成选项 A 和 B 的唯一值列表
strategy_bins_list = list(set(img['strategy_bins'] for img in images))
successful_list = list(set(img['successful'] for img in images))

# 构建可见性矩阵
visibility_matrix = []
for bin_option in strategy_bins_list:
    for success_option in successful_list:
        visibility = [img['strategy_bins'] == bin_option and img['successful'] == success_option for img in images]
        visibility_matrix.append(visibility)

# 下拉菜单按钮 A（strategy_bins）
buttons_a = [
    dict(
        label=bin_option,
        method="update",
        args=[{"visible": visibility_matrix[i::len(successful_list)][0]}]
    ) for i, bin_option in enumerate(strategy_bins_list)
]

# 下拉菜单按钮 B（successful）
buttons_b = [
    dict(
        label=success_option,
        method="update",
        args=[{"visible": visibility_matrix[j::len(strategy_bins_list)][0]}]
    ) for j, success_option in enumerate(successful_list)
]

# 设置布局
fig.update_layout(
    updatemenus=[
        dict(
            buttons=buttons_a,
            direction="down",
            x=0.49, y=1.066,
            xanchor="left",
            yanchor="top",
            showactive=True,
            name="Strategy Bins"
        ),
        dict(
            buttons=buttons_b,
            direction="down",
            x=1, y=1.066,
            xanchor="right",
            yanchor="top",
            showactive=True,
            name="Successful Percentiles"
        )
    ],
    title="Engagement and Growth Metrics by Strategy and Success Levels",
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    width=1000,
    height=1000
)

fig.show()
