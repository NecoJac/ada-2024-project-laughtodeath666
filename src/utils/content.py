# src/utils/content.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from numba.cuda.libdeviceimpl import retty


def preprocess_data(df_metadata, df_timeseries, min_videos=0):
    """Preprocess data, filter and aggregate in advance"""
    # 1. Calculate number of videos for each channel
    channel_video_counts = df_metadata['channel_id'].value_counts()

    # 2. Select channels that meet minimum video requirement
    valid_channels = channel_video_counts[channel_video_counts >= min_videos].index

    # 3. Filter data
    filtered_metadata = df_metadata[df_metadata['channel_id'].isin(valid_channels)].copy()
    filtered_timeseries = df_timeseries[df_timeseries['channel'].isin(valid_channels)].copy()

    return filtered_metadata, filtered_timeseries


def calculate_channel_metrics(filtered_metadata, filtered_timeseries):
    """Calculate channel metrics"""
    # Calculate basic metrics grouped by channel
    channel_metrics = filtered_metadata.groupby('channel_id').agg({
        'view_count': 'mean',
        'like_count': 'mean',
        'categories': lambda x: list(x)
    }).reset_index()

    # Calculate engagement rate
    channel_metrics['engagement_rate'] = channel_metrics['like_count'] / channel_metrics['view_count']

    # Calculate category distribution and diversity (information entropy) for each channel
    def calculate_diversity(categories):
        category_counts = pd.Series(categories).value_counts()
        proportions = category_counts / len(categories)
        return -np.sum(proportions * np.log(proportions))

    channel_metrics['diversity'] = channel_metrics['categories'].apply(calculate_diversity)
    channel_metrics['num_categories'] = channel_metrics['categories'].apply(lambda x: len(set(x)))
    channel_metrics['main_category'] = channel_metrics['categories'].apply(lambda x: pd.Series(x).mode()[0])
    channel_metrics['main_category_ratio'] = channel_metrics['categories'].apply(
        lambda x: pd.Series(x).value_counts().iloc[0] / len(x)
    )

    # Calculate growth metrics grouped by channel
    growth_metrics = filtered_timeseries.groupby('channel').agg({
        'delta_subs': ['sum', 'count'],  # Total growth and number of data points
        'delta_views': 'sum',
        'views': 'last',  # Latest views
        'subs': 'last'  # Latest subscribers
    })

    growth_metrics.columns = ['sub_growth', 'weeks', 'view_growth', 'final_views', 'final_subs']
    growth_metrics['weekly_sub_growth'] = growth_metrics['sub_growth'] / growth_metrics['weeks']
    growth_metrics['weekly_view_growth'] = growth_metrics['view_growth'] / growth_metrics['weeks']

    # Merge metrics
    merged_metrics = channel_metrics.merge(
        growth_metrics,
        left_on='channel_id',
        right_index=True,
        how='left'
    ).dropna()

    # Add channel strategy labels
    merged_metrics['strategy'] = pd.cut(
        merged_metrics['main_category_ratio'],
        bins=[0, 0.4, 0.7, 1],
        labels=['Diversified', 'Mixed', 'Specialized']
    )

    return merged_metrics


def identify_patterns(merged_metrics, percentile=75):
    """Identify patterns of successful channels"""

    # Define multi-dimensional success thresholds
    view_threshold = merged_metrics['view_count'].quantile(percentile / 100)
    engagement_threshold = merged_metrics['engagement_rate'].quantile(percentile / 100)
    sub_growth_threshold = merged_metrics['weekly_sub_growth'].quantile(percentile / 100)
    view_growth_threshold = merged_metrics['weekly_view_growth'].quantile(percentile / 100)

    # Identify successful channels
    successful_channels = merged_metrics[
        (merged_metrics['view_count'] >= view_threshold) |
        (merged_metrics['engagement_rate'] >= engagement_threshold) |
        (merged_metrics['weekly_sub_growth'] >= sub_growth_threshold) |
        (merged_metrics['weekly_view_growth'] >= view_growth_threshold)
        ]

    # Calculate characteristics
    patterns = {
        'avg_diversity': successful_channels['diversity'].mean(),
        'avg_categories': successful_channels['num_categories'].mean(),
        'main_category_ratio': successful_channels['main_category_ratio'].mean(),
        'channel_count': len(successful_channels),
        'avg_views': successful_channels['view_count'].mean(),
        'avg_engagement': successful_channels['engagement_rate'].mean(),
        'avg_weekly_sub_growth': successful_channels['weekly_sub_growth'].mean(),
        'avg_weekly_view_growth': successful_channels['weekly_view_growth'].mean(),
        'avg_weeks_active': successful_channels['weeks'].mean(),
    }

    # Analyze by strategy groups
    for strategy in successful_channels['strategy'].unique():
        strategy_channels = successful_channels[successful_channels['strategy'] == strategy]
        patterns[strategy] = {
            'proportion': len(strategy_channels) / len(successful_channels),
            'avg_weekly_sub_growth': strategy_channels['weekly_sub_growth'].mean(),
            'avg_weekly_view_growth': strategy_channels['weekly_view_growth'].mean(),
            'avg_diversity': strategy_channels['diversity'].mean()
        }

    print("\n=== Content Strategy Analysis Results ===")
    print(f"\nNumber of successful channels: {patterns['channel_count']}")
    print(f"Average diversity index: {patterns['avg_diversity']:.2f}")
    print(f"Average number of categories: {patterns['avg_categories']:.1f}")
    print(f"Average main category ratio: {patterns['main_category_ratio'] * 100:.1f}%")
    print(f"Average views: {patterns['avg_views']:,.0f}")
    print(f"Average engagement: {patterns['avg_engagement']:.3f}")
    print(f"Average weekly subscriber growth: {patterns['avg_weekly_sub_growth']:.1f}")
    print(f"Average weekly view growth: {patterns['avg_weekly_view_growth']:.1f}")
    print(f"Average active weeks: {patterns['avg_weeks_active']:.1f}")
    print("\nSuccessful Strategy Characteristics:")
    for strategy, stats in patterns.items():
        if isinstance(stats, dict):  # Ensure it's strategy data rather than other statistics
            print(f"\nStrategy: {strategy}")
            print(f"Proportion: {stats['proportion']:.1%}")
            print(f"Weekly average subscriber growth: {stats['avg_weekly_sub_growth']:.1f}")
            print(f"Weekly average view growth: {stats['avg_weekly_view_growth']:,.0f}")
            print(f"Content diversity: {stats['avg_diversity']:.2f}")

    return patterns, successful_channels


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

    plt.tight_layout()
    plt.show()
    return fig


def ANOVA(merged_metrics):
    # Perform an analysis of variance (ANOVA) to determine if there are significant differences between content strategies

    # for View Count
    f_view_count, p_view_count = stats.f_oneway(
        merged_metrics[merged_metrics['strategy'] == 'Diversified']['view_count'],
        merged_metrics[merged_metrics['strategy'] == 'Mixed']['view_count'],
        merged_metrics[merged_metrics['strategy'] == 'Specialized']['view_count']
    )

    # for Engagement Rate
    f_engagement_rate, p_engagement_rate = stats.f_oneway(
        merged_metrics[merged_metrics['strategy'] == 'Diversified']['engagement_rate'],
        merged_metrics[merged_metrics['strategy'] == 'Mixed']['engagement_rate'],
        merged_metrics[merged_metrics['strategy'] == 'Specialized']['engagement_rate']
    )

    # for Weekly Subscriber Growth
    f_sub_growth, p_sub_growth = stats.f_oneway(
        merged_metrics[merged_metrics['strategy'] == 'Diversified']['weekly_sub_growth'],
        merged_metrics[merged_metrics['strategy'] == 'Mixed']['weekly_sub_growth'],
        merged_metrics[merged_metrics['strategy'] == 'Specialized']['weekly_sub_growth']
    )

    # for Weekly View Growth
    f_view_growth, p_view_growth = stats.f_oneway(
        merged_metrics[merged_metrics['strategy'] == 'Diversified']['weekly_view_growth'],
        merged_metrics[merged_metrics['strategy'] == 'Mixed']['weekly_view_growth'],
        merged_metrics[merged_metrics['strategy'] == 'Specialized']['weekly_view_growth']
    )

    # Display results
    print("Analysis of Variance Results:")
    print(f"View Count - F: {f_view_count}, p-value: {p_view_count}")
    print(f"Engagement Rate - F: {f_engagement_rate}, p-value: {p_engagement_rate}")
    print(f"Weekly Subscriber Growth - F: {f_sub_growth}, p-value: {p_sub_growth}")
    print(f"Weekly View Growth - F: {f_view_growth}, p-value: {p_view_growth}")


def variance4each(merged_metrics):
    # Conduct the analysis on each main_category to determine if strategies differ significantly within the category
    analysis_variance_results = {}

    for category in merged_metrics['main_category'].unique():
        data = merged_metrics[merged_metrics['main_category'] == category]

        f_view, p_view = stats.f_oneway(
            data[data['strategy'] == 'Diversified']['view_count'],
            data[data['strategy'] == 'Mixed']['view_count'],
            data[data['strategy'] == 'Specialized']['view_count']
        )
        f_engagement, p_engagement = stats.f_oneway(
            data[data['strategy'] == 'Diversified']['engagement_rate'],
            data[data['strategy'] == 'Mixed']['engagement_rate'],
            data[data['strategy'] == 'Specialized']['engagement_rate']
        )
        f_sub_growth, p_sub_growth = stats.f_oneway(
            data[data['strategy'] == 'Diversified']['weekly_sub_growth'],
            data[data['strategy'] == 'Mixed']['weekly_sub_growth'],
            data[data['strategy'] == 'Specialized']['weekly_sub_growth']
        )
        f_view_growth, p_view_growth = stats.f_oneway(
            data[data['strategy'] == 'Diversified']['weekly_view_growth'],
            data[data['strategy'] == 'Mixed']['weekly_view_growth'],
            data[data['strategy'] == 'Specialized']['weekly_view_growth']
        )

        # Store results for each category
        analysis_variance_results[category] = {
            'view_count': (f_view, p_view),
            'engagement_rate': (f_engagement, p_engagement),
            'weekly_sub_growth': (f_sub_growth, p_sub_growth),
            'weekly_view_growth': (f_view_growth, p_view_growth)
        }

    # Convert analysis variance results to DataFrames for display
    analysis_variance_results_df = pd.DataFrame([
        {'Category': category, 'Metric': metric, 'F-Value': f_value, 'P-Value': p_value}
        for category, metrics in analysis_variance_results.items()
        for metric, (f_value, p_value) in metrics.items()
    ])
    # Display results
    print("The Analysis of Variance Results for Each Content Category:")
    print(analysis_variance_results_df.sort_values(by=['Category', 'Metric']).reset_index(drop=True))

    return analysis_variance_results


def significant_analysis(analysis_variance_results):
    # Initialize dictionaries to store significant and non-significant results
    significant_results = {}
    non_significant_results = {}

    # Iterate through results above to categorize based on p-value
    for category, results in analysis_variance_results.items():
        significant_results[category] = {}
        non_significant_results[category] = {}

        for metric, (f_value, p_value) in results.items():
            if p_value < 0.05:
                significant_results[category][metric] = (f_value, p_value)
            else:
                non_significant_results[category][metric] = (f_value, p_value)

    # Convert significant and non-significant results to DataFrames for display
    significant_results_df = pd.DataFrame(
        [(cat, metric, f, p) for cat, metrics in significant_results.items() for metric, (f, p) in metrics.items()],
        columns=['Category', 'Metric', 'F-Value', 'P-Value']
    )

    non_significant_results_df = pd.DataFrame(
        [(cat, metric, f, p) for cat, metrics in non_significant_results.items() for metric, (f, p) in metrics.items()],
        columns=['Category', 'Metric', 'F-Value', 'P-Value']
    )
    # Display significant and non-significant results
    print("Significant Results (p < 0.05):")
    print(significant_results_df.sort_values(by=['Category', 'Metric']).reset_index(drop=True))

    print("\nNon-Significant Results (p >= 0.05):")
    print(non_significant_results_df.sort_values(by=['Category', 'Metric']).reset_index(drop=True))

    return significant_results


def significant_means_value(significant_results, merged_metrics):
    # Dictionary to store mean values only for significant categories and metrics
    significant_means = {}

    # Iterate through significant categories and metrics
    for category, metrics in significant_results.items():
        category_data = merged_metrics[merged_metrics['main_category'] == category]
        category_means = {}

        # Calculate mean values for each metric within each strategy, using observed=True to silence the warning
        for metric in metrics.keys():  # Only process significant metrics
            means = category_data.groupby('strategy', observed=True)[metric].mean()  # Add observed=True here
            category_means[metric] = means

        # Only add categories with available means to the final results
        if category_means:
            significant_means[category] = category_means

    # Print only non-empty categories with significant metrics
    # Convert analysis variance results to DataFrames for display
    significant_means_flat = [
        {'Category': category, 'Metric': metric, 'Strategy': strategy, 'Mean_Value': mean_value}
        for category, metrics in significant_means.items()
        for metric, means in metrics.items()
        for strategy, mean_value in means.items()
    ]
    significant_means_df = pd.DataFrame(significant_means_flat)

    # Display significant and non-significant results
    print("Mean values for significant categories and metrics by strategy:")
    print(significant_means_df)

    return significant_means


def metrics_plot_insights(significant_means):
    # Creating a DataFrame with significant categories and metrics
    plot_data = []
    for category, metrics in significant_means.items():
        for metric, mean_values in metrics.items():
            for strategy, mean_value in mean_values.items():
                plot_data.append({
                    'Category_Metric': f"{category}\n - {metric.replace('_', ' ').title()}",
                    'Strategy': strategy,
                    'Mean Value': mean_value,
                    'Metric': metric
                })

    plot_df = pd.DataFrame(plot_data)

    metrics_to_plot = ["view_count", "weekly_sub_growth", "weekly_view_growth", "engagement_rate"]
    titles = {
        "view_count": "Mean View Count by Strategy",
        "weekly_sub_growth": "Mean Weekly Subscriber Growth by Strategy",
        "weekly_view_growth": "Mean Weekly View Growth by Strategy",
        "engagement_rate": "Mean Engagement Rate by Strategy"
    }

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.flatten()

    for i, metric in enumerate(metrics_to_plot):
        sns.barplot(
            data=plot_df[plot_df["Metric"] == metric],
            x="Category_Metric", y="Mean Value", hue="Strategy", palette="viridis", ax=axes[i]
        )
        axes[i].set_title(titles[metric])
        axes[i].set_xlabel("Category and Metric")
        axes[i].set_ylabel("Mean Value")

        # Rotate x-axis labels only for "view_count" and "weekly_sub_growth"
        if metric in ["view_count", "weekly_sub_growth"]:
            axes[i].tick_params(axis='x', rotation=5)

        axes[i].legend_.remove()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Strategy", loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.05))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()