# src/utils/duration.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import f_oneway

def calculate_popularity_score_vectorized(df,weights=[0.5,0.3,0.2]):
    """
    Vectorized calculation of popularity score
    """

    view_score = np.log1p(df['view_count'])
    like_score = np.log1p(df['like_count'])
    comment_score = np.log1p(df['num_comms'])


    max_view = np.log1p(df['view_count'].max())
    max_like = np.log1p(df['like_count'].max())
    max_comment = np.log1p(df['num_comms'].max())


    score = (
            weights[0] * (view_score / max_view) +
            weights[1] * (like_score / max_like) +
            weights[2] * (comment_score / max_comment)
    )

    return score


def optimal_video_length(df_metadata, df_comments):
    """
    Get the top 10 videos based on view_count, like_count, num_comms, and popularity_score, for each category and year,
    and calculate the average duration_minutes for each metric.
    """
    # Select only needed columns for merging to reduce memory usage
    df_metadata_subset = df_metadata[
        ['categories', 'display_id', 'duration', 'view_count', 'like_count', 'upload_date']].copy()
    df_comments_subset = df_comments[['display_id', 'num_comms']].copy()

    # Merge datasets
    df_analysis = pd.merge(
        df_metadata_subset,
        df_comments_subset,
        on='display_id',
        how='left'
    )
    df_analysis = df_analysis[df_analysis['categories'] != '']

    # Convert duration to minutes
    df_analysis['duration_minutes'] = df_analysis['duration'] / 60

    # Vectorized calculation of popularity score
    df_analysis['popularity_score'] = calculate_popularity_score_vectorized(df_analysis)

    # Pre-calculate year
    df_analysis['year'] = df_analysis['upload_date'].dt.year

    # Create the list and metrics
    top_10_results = []
    metrics = ['view_count', 'like_count', 'num_comms', 'popularity_score']

    # Group by category and year
    for (category, year), group in df_analysis.groupby(['categories', 'year']):
        result = {'categories': category, 'year': year}

        for metric in metrics:
            # Sort by the current metric and get the top 10
            top_10 = group.nlargest(10, metric)
            # Calculate the mean duration_minutes for the top 10 videos
            mean_duration = top_10['duration_minutes'].mean()
            # Store the result with the category and year
            result[f'max_{metric}_duration'] = mean_duration

        # Append the result for the current category and year
        top_10_results.append(result)

    # Convert the results to a DataFrame
    optimal_durations_df = pd.DataFrame(top_10_results)

    return optimal_durations_df


import plotly.express as px
import plotly.graph_objects as go


def plot_max_duration(optimal_durations_df):
    """
    Plot interactive max view_duration, max like_duration, max comment_duration,
    and max popularity_duration by year for each category using Plotly.
    """
    # List of metrics to plot
    metrics = ['max_view_count_duration', 'max_like_count_duration',
               'max_num_comms_duration', 'max_popularity_score_duration']
    titles = ['Max View_number Duration by Year', 'Max Like_number Duration by Year',
              'Max Comments_number Duration by Year', 'Max Popularity_score Duration by Year']

    # Loop through each metric to create an interactive plot
    for metric, title in zip(metrics, titles):
        fig = px.line(
            optimal_durations_df,
            x='year',
            y=metric,
            color='categories',
            markers=True,
            title=title,
            labels={'year': 'Year', metric: f"{metric.replace('_', ' ').title()} (minutes)"},
            template="plotly_white"
        )

        # Update layout for better visualization
        fig.update_layout(
            xaxis=dict(tickangle=45),
            yaxis=dict(title=f"{metric.replace('_', ' ').title()}"),
            legend_title="Categories",
            hovermode="x unified"
        )

        # Show the interactive figure
        fig.show()


def analyze_duration_metrics(df_metadata, df_comments):
    """
    Analyze relationships between video duration and various performance metrics
    """
    # Select only needed columns for merging to reduce memory usage
    df_metadata_subset = df_metadata[['display_id', 'duration', 'view_count', 'like_count', 'upload_date']].copy()
    df_comments_subset = df_comments[['display_id', 'num_comms']].copy()

    df_analysis = pd.merge(
        df_metadata_subset,
        df_comments_subset,
        on='display_id',
        how='left'
    )

    # Convert duration to minutes
    df_analysis['duration_minutes'] = df_analysis['duration'] / 60

    # Vectorized calculation of popularity score
    df_analysis['popularity_score'] = calculate_popularity_score_vectorized(df_analysis)

    # Create duration categories
    duration_bins = [0, 1, 3, 5, 10, 15, 30, np.inf]
    duration_labels = ['0-1', '1-3', '3-5', '5-10', '10-15', '15-30', '30+']
    df_analysis['duration_category'] = pd.cut(
        df_analysis['duration_minutes'],
        bins=duration_bins,
        labels=duration_labels
    )

    df_analysis['year_month'] = df_analysis['upload_date'].dt.to_period('M')

    return df_analysis

def plot_engagement_metrics(df_analysis):
    """
    Plot engagement metric charts
    """
    plot_data = df_analysis

    # Calculate engagement rate
    plot_data['engagement_rate'] = (plot_data['like_count'] / plot_data['view_count'] * 100).fillna(0)
    plot_data['engagement_rate'] = plot_data['engagement_rate'].clip(upper=100)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    metrics = ['view_count', 'like_count', 'num_comms', 'popularity_score', 'engagement_rate']
    titles = ['Views by Duration', 'Likes by Duration', 'Comments by Duration', 'Popularity Score by Duration', 'Engagement Rate by Duration']

    for i, (metric, title) in enumerate(zip(metrics[:5], titles)):
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
            plot_data['duration_category'].unique().sort_values(),
            rotation=45
        )

        axes[row, col].set_title(title)
        if metric in ['view_count', 'like_count', 'num_comms']:
            axes[row, col].set_yscale('log')
        axes[row, col].set_xlabel('Duration (minutes)')

    plt.tight_layout()


def analyze_temporal_trends(df_analysis):

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
    plt.tight_layout()
    plt.show()

    return plt

def verify_popularity_weights(df_analysis):
    """Verify the stability of the weight for caculating popularity score"""
    df_analysis=df_analysis.copy()
    weights=[[0.5,0.3,0.2],[0.3,0.5,0.2],[0.2,0.3,0.5],[0.33,0.33,0.34]]

    print(f"===Stability Verification for Popularity Weights===")
    for weight in weights:
        print(f"\n\tPopularity Weights: View={weight[0]}, Like={weight[1]}, Comment={weight[2]}")
        df_analysis['popularity_score'] = calculate_popularity_score_vectorized(df_analysis,weight)
        # Calculate the correlation between the popularity score and the view count
        metric='popularity_score'
        title='Popularity Score by Duration'
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(
            x='duration_category',
            y=metric,
            data=df_analysis,
        )

        # Set x-axis ticks and labels correctly
        ax.set_xticks(range(len(df_analysis['duration_category'].unique())))
        ax.set_xticklabels(
            df_analysis['duration_category'].unique().sort_values(),
            rotation=45
        )

        ax.set_title(title)
        ax.set_xlabel('Duration (minutes)')
        plt.show()
        plot=analyze_temporal_trends(df_analysis.copy())


def analyze_duration_metrics_by_category(df_metadata, df_comments):
    """
    Analyze relationships between video duration and various performance metrics by category
    """
    # Select only needed columns for merging to reduce memory usage

    df_metadata_subset = df_metadata[['categories', 'display_id', 'duration', 'view_count', 'like_count', 'upload_date']].copy()
    df_comments_subset = df_comments[['display_id', 'num_comms']].copy()

    # Merge datasets
    df_analysis = pd.merge(
        df_metadata_subset,
        df_comments_subset,
        on='display_id',
        how='left'
    )
    df_analysis = df_analysis[df_analysis['categories'] != '']

    # Convert duration to minutes
    df_analysis['duration_minutes'] = df_analysis['duration'] / 60

    # Vectorized calculation of popularity score
    df_analysis['popularity_score'] = calculate_popularity_score_vectorized(df_analysis)

    # Create duration categories
    duration_bins = [0, 1, 3, 5, 10, 15, 30, np.inf]
    duration_labels = ['0-1', '1-3', '3-5', '5-10', '10-15', '15-30', '30+']
    df_analysis['duration_category'] = pd.cut(
        df_analysis['duration_minutes'],
        bins=duration_bins,
        labels=duration_labels
    )
    df_analysis['year_month'] = df_analysis['upload_date'].dt.to_period('M')
    return df_analysis

import plotly.graph_objects as go
def analyze_temporal_trends_by_category(df_analysis):
    """
    Analyze and plot the evolution of video duration preferences by category
    """
    # Ensure 'upload_date' is in datetime format if not already
    if not pd.api.types.is_datetime64_any_dtype(df_analysis['upload_date']):
        df_analysis['upload_date'] = pd.to_datetime(df_analysis['upload_date'])

    # Pre-calculate year and month
    df_analysis['year_month'] = df_analysis['upload_date'].dt.to_period('M')

    # Get unique categories
    categories = df_analysis['categories'].dropna().unique()

    # Create a figure with multiple subplots
    num_categories = len(categories)
    cols = 3
    rows = (num_categories + cols - 1) // cols  # Ensure enough rows for all subplots

    for i, category in enumerate(categories):
        # Filter data for the category
        category_data = df_analysis[df_analysis['categories'] == category]

        # Calculate temporal trends
        temporal_trends = (category_data.groupby(
            ['year_month', 'duration_category'],
            observed=True
        )['popularity_score'].mean().reset_index())

        # Reshape data using pivot_table
        df = temporal_trends.pivot(
            index='year_month',
            columns='duration_category',
            values='popularity_score'
        )
        df_long = df.melt(id_vars='year_month', var_name='duration', value_name='popularity')

        # Create a Plotly interactive line chart
        fig = go.Figure()

        # Add traces for each duration category
        for duration in df_long['duration'].unique():
            data = df_long[df_long['duration'] == duration]
            fig.add_trace(go.Scatter(
                x=data['year_month'],
                y=data['popularity'],
                mode='lines+markers',
                name=duration,
                hovertemplate=
                '<b>Duration</b>: ' + duration + '<br>' +
                '<b>Time</b>: %{x}<br>' +
                '<b>Popularity</b>: %{y}<extra></extra>'
            ))

        # Update Layout for interactivity
        fig.update_layout(
            title='Evolution of Video Duration Preferences:'+category,
            width=1200,  # 设置图的宽度，例如 1200px
            height=600,  # 设置图的高度，例如 600px
            xaxis=dict(
                title='Time',
                tickangle=45,  # Rotate x-axis labels 45 degrees
                tickmode='auto',
                nticks=20  # Reduce x-axis tick density
            ),
            yaxis=dict(
                title='Average Popularity Score'
            ),
            legend_title='Duration (minutes)',  # Add legend title
            hovermode='closest',  # Make hover interactive and detailed
        )

        # Show the figure
        fig.show()
        fig.write_html(category+".html")

def plot_duration_distribution_pie(df_analysis):
    """
    Plot a pie chart showing the distribution of videos across duration categories.
    """
    # Calculate the counts for each duration category
    duration_counts = df_analysis['duration_category'].value_counts()

    # Create the pie chart
    plt.figure(figsize=(8,6))
    plt.pie(
        duration_counts,
        labels=duration_counts.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=sns.color_palette('pastel', len(duration_counts))
    )

    # Add a title
    plt.title('Distribution of Videos by Duration Category')
    plt.legend(title='Duration (minutes)',loc='lower right', bbox_to_anchor=(1.2, 0.5))
    plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
    plt.show()

def cohens_d(group1, group2):
    # Calculate the mean and standard deviation
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)

    # Calculate Cohen's d
    pooled_std = np.sqrt(((std1 ** 2) + (std2 ** 2)) / 2)
    return abs(mean1 - mean2) / pooled_std


def prepare_data_before(df_analysis):
    # Filter out data from 2006 to 2013
    df_filtered = df_analysis[
        (df_analysis['upload_date'] >= '2006-01-01') &
        (df_analysis['upload_date'] <= '2013-12-31')
        ].copy()

    # Aggregate data
    agg_metrics = df_filtered.groupby(
        ['categories', 'duration_category', df_filtered['upload_date'].dt.year],
        observed=False  # 显式设置 observed 参数
    )['popularity_score'].mean().reset_index()

    agg_metrics.rename(columns={'upload_date': 'year'}, inplace=True)

    # Filter out groups without data
    agg_metrics = agg_metrics.dropna(subset=['popularity_score'])

    return agg_metrics


def prepare_data_after(df_analysis):
    # Filter out data from 2014 to 2019
    df_filtered = df_analysis[
        (df_analysis['upload_date'] >= '2014-01-01') &
        (df_analysis['upload_date'] <= '2019-12-31')
        ].copy()

    # Aggregate data
    agg_metrics = df_filtered.groupby(
        ['categories', 'duration_category', df_filtered['upload_date'].dt.year],
        observed=False  # 显式设置 observed 参数
    )['popularity_score'].mean().reset_index()

    agg_metrics.rename(columns={'upload_date': 'year'}, inplace=True)

    # Filter out groups without data
    agg_metrics = agg_metrics.dropna(subset=['popularity_score'])

    return agg_metrics


def compute_significance(agg_metrics):
    results = []

    # Get all different categories
    categories = agg_metrics['categories'].unique()

    # Bonferroni calibration
    num_comparisons = len(categories) * len(agg_metrics['duration_category'].unique())
    alpha = 0.01 / num_comparisons  # Adjusted significance threshold

    # for all categories
    for category in categories:
        category_data = agg_metrics[agg_metrics['categories'] == category]

        duration_groups = [
            group['popularity_score'].values
            for _, group in category_data.groupby('duration_category', observed=False)
        ]

        # Perform ANOVA test
        if len(duration_groups) > 1:
            f_stat, p_val = f_oneway(*duration_groups)

            # Calculate the size of the effect
            if len(duration_groups) > 1:
                d_value = cohens_d(duration_groups[0], duration_groups[1])
            else:
                d_value = np.nan  # If not enough groups for Cohen's d

            # get results
            results.append({
                'category': category,
                'f_stat': f_stat,
                'p_value': p_val,
                'effect_size': d_value,
                'is_significant': p_val < alpha
            })

    results_df = pd.DataFrame(results)

    return results_df


def compute_significance_for_worst_only(agg_metrics):
    results = []

    # for all categories
    for category in agg_metrics['categories'].unique():
        category_data = agg_metrics[agg_metrics['categories'] == category]

        # find worst groups
        worst_duration = category_data.loc[category_data['popularity_score'].idxmin(), 'duration_category']
        worst_group = category_data[category_data['duration_category'] == worst_duration]

        # find other groups
        other_groups = category_data[category_data['duration_category'] != worst_duration]

        # Perform ANOVA test
        if len(worst_group) > 0 and len(other_groups) > 0:
            duration_groups = [
                worst_group['popularity_score'].values,
                other_groups['popularity_score'].values
            ]

            f_stat, p_val = f_oneway(*duration_groups)

            # Calculate the size of the effect
            effect_size = cohens_d(worst_group['popularity_score'], other_groups['popularity_score'])

            # get results
            results.append({
                'category': category,
                'worst_duration': worst_duration,
                'f_stat': f_stat,
                'p_value': p_val,
                'cohen_d': effect_size
            })

    results_df = pd.DataFrame(results)

    # Bonferroni calibration
    n_tests = len(results_df)
    results_df['p_value_bonferroni'] = results_df['p_value'] * n_tests
    results_df['is_significant_bonferroni'] = results_df['p_value_bonferroni'] < 0.01

    return results_df


def compute_significance_for_best_only(agg_metrics):
    results = []

    # for all categories
    for category in agg_metrics['categories'].unique():
        category_data = agg_metrics[agg_metrics['categories'] == category]

        # find best groups
        best_duration = category_data.loc[category_data['popularity_score'].idxmax(), 'duration_category']
        best_group = category_data[category_data['duration_category'] == best_duration]

        # find other groups
        other_groups = category_data[category_data['duration_category'] != best_duration]

        # Perform ANOVA test
        if len(best_group) > 0 and len(other_groups) > 0:
            duration_groups = [
                best_group['popularity_score'].values,
                other_groups['popularity_score'].values
            ]

            f_stat, p_val = f_oneway(*duration_groups)

            # Calculate the size of the effect
            effect_size = cohens_d(best_group['popularity_score'], other_groups['popularity_score'])

            # get results
            results.append({
                'category': category,
                'best_duration': best_duration,
                'f_stat': f_stat,
                'p_value': p_val,
                'cohen_d': effect_size
            })

    results_df = pd.DataFrame(results)

    # Bonferroni calibration
    n_tests = len(results_df)
    results_df['p_value_bonferroni'] = results_df['p_value'] * n_tests
    results_df['is_significant_bonferroni'] = results_df['p_value_bonferroni'] < 0.05

    return results_df


def group_categories_by_significance(
    significance_results1, significance_results2, significance_results3, significance_results4
):
    # merge results
    merged_results = significance_results1[['category', 'is_significant_bonferroni']].rename(
        columns={'is_significant_bonferroni': 'sig1'}
    ).merge(
        significance_results2[['category', 'is_significant_bonferroni']].rename(
            columns={'is_significant_bonferroni': 'sig2'}
        ),
        on='category',
        how='outer'
    ).merge(
        significance_results3[['category', 'is_significant_bonferroni']].rename(
            columns={'is_significant_bonferroni': 'sig3'}
        ),
        on='category',
        how='outer'
    ).merge(
        significance_results4[['category', 'is_significant_bonferroni']].rename(
            columns={'is_significant_bonferroni': 'sig4'}
        ),
        on='category',
        how='outer'
    )

    merged_results.fillna(False, inplace=True)

    # Create Boolean matrix
    merged_results['matrix'] = merged_results[['sig1', 'sig2', 'sig3', 'sig4']].apply(
        lambda row: tuple(row), axis=1
    )

    # Group and merge categories by matrix
    grouped = merged_results.groupby('matrix')['category'].apply(list).reset_index()

    grouped.rename(columns={'category': 'categories'}, inplace=True)

    return grouped