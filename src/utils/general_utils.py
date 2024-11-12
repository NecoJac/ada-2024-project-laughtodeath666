import matplotlib.pyplot as plt
import seaborn as sns

def diversity_plot_insights(merged_metrics):
    """Create key insight visualizations"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Diversity vs Views
    sns.scatterplot(
        data=merged_metrics,
        x='diversity',
        y='view_count',
        hue='strategy',
        alpha=0.5,
        ax=axes[0,0]
    )
    axes[0,0].set_title('Content Diversity vs Average Views')
    axes[0,0].set_yscale('log')  # Use log scale

    # Add a secondary y-axis to show avg_views
    ax2 = axes[0,0].twinx()
    ax2.set_ylabel('Avg. Views', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    for strategy, c in zip(merged_metrics['strategy'].unique(), ['green', 'red', 'blue']):
        strategy_data = merged_metrics[merged_metrics['strategy'] == strategy]
        avg_diversity = strategy_data['diversity'].mean()
        avg_views = strategy_data['view_count'].mean()
        ax2.plot(avg_diversity, avg_views, '*', label=f'{strategy} (avg views)', color=c, alpha=1, markersize=15)

    # 2. Number of categories vs Engagement
    sns.scatterplot(
        data=merged_metrics,
        x='diversity',
        y='engagement_rate',
        hue='strategy',
        alpha=0.5,
        ax=axes[0,1]
    )
    axes[0,1].set_title('Content Diversity vs Average Engagement_rate')

    # Add a secondary y-axis to show avg_engagement
    ax2=axes[0,1].twinx()
    ax2.set_ylabel('Avg. Engagement_rate', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    for strategy, c in zip(merged_metrics['strategy'].unique(), ['green', 'red', 'blue']):
        strategy_data = merged_metrics[merged_metrics['strategy'] == strategy]
        avg_diversity = strategy_data['diversity'].mean()
        avg_engagement = strategy_data['engagement_rate'].mean()
        ax2.plot(avg_diversity, avg_engagement, '*', label=f'{strategy} (avg engagement)', color=c, alpha=1, markersize=15)

    # 3. Growth rate vs Diversity
    sns.scatterplot(
        data=merged_metrics,
        x='diversity',
        y='weekly_sub_growth',
        hue='strategy',
        alpha=0.5,
        ax=axes[1,0]
    )
    axes[1,0].set_title('Content Diversity vs Weekly_sub_growth')

    # Add a secondary y-axis to show avg_weekly_sub_growth
    ax2=axes[1,0].twinx()
    ax2.set_ylabel('Avg. Weekly_sub_growth', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    for strategy, c in zip(merged_metrics['strategy'].unique(), ['green', 'red', 'blue']):
        strategy_data = merged_metrics[merged_metrics['strategy'] == strategy]
        avg_diversity = strategy_data['diversity'].mean()
        avg_weekly_sub_growth = strategy_data['weekly_sub_growth'].mean()
        ax2.plot(avg_diversity, avg_weekly_sub_growth, '*', label=f'{strategy} (avg weekly_sub_growth)', color=c, alpha=1, markersize=15)

    # 4. Main category ratio vs Growth rate
    sns.scatterplot(
        data=merged_metrics,
        x='diversity',
        y='weekly_view_growth',
        hue='strategy',
        alpha=0.5,
        ax=axes[1,1]
    )
    axes[1,1].set_title('Content Diversity vs Weekly_view_growth')

    # Add a secondary y-axis to show avg_weekly_view_growth
    ax2=axes[1,1].twinx()
    ax2.set_ylabel('Avg. Weekly_view_growth', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    for strategy, c in zip(merged_metrics['strategy'].unique(), ['green', 'red', 'blue']):
        strategy_data = merged_metrics[merged_metrics['strategy'] == strategy]
        avg_diversity = strategy_data['diversity'].mean()
        avg_weekly_view_growth = strategy_data['weekly_view_growth'].mean()
        ax2.plot(avg_diversity, avg_weekly_view_growth, '*', label=f'{strategy} (avg weekly_view_growth)', color=c, alpha=1, markersize=15)


    plt.tight_layout()
    return fig

def plot_engagement_metrics(df_analysis):
    """
    Plot engagement metric charts
    """
    plot_data = df_analysis

    # Calculate engagement rate
    plot_data['engagement_rate'] = ((plot_data['like_count']+plot_data['dislike_count']+plot_data['num_comms']) / plot_data['view_count'] * 100).fillna(0)
    plot_data['engagement_rate'] = plot_data['engagement_rate'].clip(upper=100)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    metrics = ['view_count', 'like_count', 'dislike_count','num_comms', 'popularity_score', 'engagement_rate']
    titles = ['Views by Duration', 'Likes by Duration','Dislikes by Duration','Comments by Duration', 'Popularity Score by Duration', 'Engagement Rate by Duration']

    xticklabels =plot_data['duration_category'].unique().sort_values()

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        row = i // 3
        col = i % 3

        # Create box plot
        sns.boxplot(
            x='duration_category',
            y=metric,
            data=plot_data,
            ax=axes[row, col]
        )

        # Set x-axis ticks and labels correctly
        axes[row, col].set_xticks(range(len(plot_data['duration_category'].unique())))
        axes[row, col].set_xticklabels(
            xticklabels,
            rotation=45
        )

        axes[row, col].set_title(title)
        if metric in ['view_count', 'like_count', 'dislike_count','num_comms']:
            axes[row, col].set_yscale('log')
        axes[row, col].set_xlabel('Duration (minutes)')

    plt.tight_layout()

    # Calculate statistics, explicitly specify observed parameter
    engagement_stats = df_analysis.groupby('duration_category', observed=True)['engagement_rate'].agg([
        'mean', 'median', 'std'
    ]).round(2)

    return fig, engagement_stats

def analyze_temporal_trends(df_analysis):
    # Pre-calculate year and month
    df_analysis['year_month'] = df_analysis['upload_date'].dt.to_period('M')

    # Explicitly specify observed parameter
    temporal_trends = (df_analysis.groupby(
        ['year_month', 'duration_category'],
        observed=True
    )['popularity_score'].mean().reset_index())

    plt.figure(figsize=(15, 8))

    # Reshape data using pivot_table
    pivot_data = temporal_trends.pivot(
        index='year_month',
        columns='duration_category',
        values='popularity_score'
    )

    # Plot trend lines
    for col in pivot_data.columns:
        if not pivot_data[col].empty:
            plt.plot(range(len(pivot_data)), pivot_data[col], label=col, marker='o', markersize=4)

    # Set x-axis labels
    step = max(len(pivot_data) // 10, 1)
    plt.xticks(
        range(0, len(pivot_data), step),
        [str(idx) for idx in pivot_data.index[::step]],
        rotation=45
    )

    plt.title('Evolution of Video Duration Preferences')
    plt.xlabel('Time')
    plt.ylabel('Average Popularity Score')
    plt.legend(title='Duration (minutes)', bbox_to_anchor=(1.05, 1))
    plt.grid(True, alpha=0.3)

    return plt

def plot_engagement_metrics(df_analysis):
    """
    Plot engagement metric charts
    """
    plot_data = df_analysis

    # Calculate engagement rate
    plot_data['engagement_rate'] = ((plot_data['like_count']+plot_data['dislike_count']+plot_data['num_comms']) / plot_data['view_count'] * 100).fillna(0)
    plot_data['engagement_rate'] = plot_data['engagement_rate'].clip(upper=100)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    metrics = ['view_count', 'like_count', 'dislike_count','num_comms', 'popularity_score', 'engagement_rate']
    titles = ['Views by Duration', 'Likes by Duration','Dislikes by Duration','Comments by Duration', 'Popularity Score by Duration', 'Engagement Rate by Duration']

    xticklabels =plot_data['duration_category'].unique().sort_values()

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        row = i // 3
        col = i % 3

        # Create box plot
        sns.boxplot(
            x='duration_category',
            y=metric,
            data=plot_data,
            ax=axes[row, col]
        )

        # Set x-axis ticks and labels correctly
        axes[row, col].set_xticks(range(len(plot_data['duration_category'].unique())))
        axes[row, col].set_xticklabels(
            xticklabels,
            rotation=45
        )

        axes[row, col].set_title(title)
        if metric in ['view_count', 'like_count', 'dislike_count','num_comms']:
            axes[row, col].set_yscale('log')
        axes[row, col].set_xlabel('Duration (minutes)')

    plt.tight_layout()

    # Calculate statistics, explicitly specify observed parameter
    engagement_stats = df_analysis.groupby('duration_category', observed=True)['engagement_rate'].agg([
        'mean', 'median', 'std'
    ]).round(2)

    return fig, engagement_stats

def analyze_temporal_trends(df_analysis):
    # Pre-calculate year and month
    df_analysis['year_month'] = df_analysis['upload_date'].dt.to_period('M')

    # Explicitly specify observed parameter
    temporal_trends = (df_analysis.groupby(
        ['year_month', 'duration_category'],
        observed=True
    )['popularity_score'].mean().reset_index())

    plt.figure(figsize=(15, 8))

    # Reshape data using pivot_table
    pivot_data = temporal_trends.pivot(
        index='year_month',
        columns='duration_category',
        values='popularity_score'
    )

    # Plot trend lines
    for col in pivot_data.columns:
        if not pivot_data[col].empty:
            plt.plot(range(len(pivot_data)), pivot_data[col], label=col, marker='o', markersize=4)

    # Set x-axis labels
    step = max(len(pivot_data) // 10, 1)
    plt.xticks(
        range(0, len(pivot_data), step),
        [str(idx) for idx in pivot_data.index[::step]],
        rotation=45
    )

    plt.title('Evolution of Video Duration Preferences')
    plt.xlabel('Time')
    plt.ylabel('Average Popularity Score')
    plt.legend(title='Duration (minutes)', bbox_to_anchor=(1.05, 1))
    plt.grid(True, alpha=0.3)

    return plt
