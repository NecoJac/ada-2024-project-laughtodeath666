import pandas as pd
import plotly.graph_objects as go

# 权重列表
weights = ['[0.5, 0.3, 0.2]', '[0.3, 0.5, 0.2]', '[0.2, 0.3, 0.5]', '[0.33, 0.33, 0.34]']

# 创建一个空的 Figure
fig = go.Figure()

# 初始化按钮和 traces
traces = []
buttons = []

# 遍历每个权重，加载数据并添加到图形中
for i, weight in enumerate(weights):
    # 读取数据
    df = pd.read_csv(f'pivot_df{weight}.csv')

    # 转换数据为长格式
    df_long = df.melt(id_vars='year_month', var_name='duration', value_name='popularity')
    df_long['year_month'] = df_long['year_month'].astype(str)

    # 添加 traces
    for duration in df_long['duration'].unique():
        data = df_long[df_long['duration'] == duration]
        fig.add_trace(go.Scatter(
            x=data['year_month'],
            y=data['popularity'],
            mode='lines+markers',
            name=duration,
            visible=(i == 0),  # 初始时第一个权重可见
            hovertemplate=
            '<b>Duration</b>: %{name}<br>'
            '<b>Time</b>: %{x}<br>'
            '<b>Popularity</b>: %{y}<extra></extra>'
        ))

    # 创建当前按钮的可见性列表
    visibility = [False] * (len(weights) * len(df_long['duration'].unique()))
    start_index = i * len(df_long['duration'].unique())
    end_index = start_index + len(df_long['duration'].unique())
    visibility[start_index:end_index] = [True] * len(df_long['duration'].unique())

    # 添加按钮
    buttons.append(dict(
        label=f"Weights = {weight}",
        method="update",
        args=[{"visible": visibility}, {"title": f"Evolution of Video Duration Preferences (weights = {weight})"}]
    ))

# 添加下拉菜单
fig.update_layout(
    updatemenus=[dict(
        buttons=buttons,
        direction="down",
        x=1.0,
        y=1.15,
        xanchor="right",
        yanchor="top",
        showactive=True
    )],
    title="Evolution of Video Duration Preferences (weights = [0.5, 0.3, 0.2])",
    xaxis=dict(
        title='Time',
        tickangle=45
    ),
    yaxis=dict(
        title='Average Popularity Score'
    ),
    legend_title='Duration (minutes)',
    hovermode='closest',
    width=1100,
    height=600
)

# 显示图形
fig.show()
fig.write_html('line_duration.html')