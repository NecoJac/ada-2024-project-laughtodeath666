# src/utils/duration.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def calculate_popularity_score_vectorized(df):
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
            0.5 * (view_score / max_view) +
            0.3 * (like_score / max_like) +
            0.2 * (comment_score / max_comment)
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


def plot_max_duration(optimal_durations_df):
    """
    Plot max view_duration, max like_duration, max comment_duration, and max popularity_duration by year for each category.
    """
    sns.set(style="whitegrid")

    # Set up the plotting layout
    plt.figure(figsize=(15, 12))

    # List of metrics to plot
    metrics = ['max_view_count_duration', 'max_like_count_duration', 'max_num_comms_duration',
               'max_popularity_score_duration']
    titles = ['Max View_number Duration by Year', 'Max Like_number Duration by Year',
              'Max Comments_number Duration by Year', 'Max Popularity_score Duration by Year']

    # Iterate through each metric and create a separate plot
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        plt.subplot(2, 2, i + 1)

        # Create line plot for the current metric
        sns.lineplot(x='year', y=metric, hue='categories', data=optimal_durations_df, marker='o')

        # Title and labels
        plt.title(title)
        plt.xlabel('Year')
        plt.ylabel(f'{metric.replace("_", " ").title()} (minutes)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()


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
    duration_bins = [0, 1, 3, 5, 10, 15, 30, 60, np.inf]
    duration_labels = ['0-1', '1-3', '3-5', '5-10', '10-15', '15-30', '30-60', '60+']
    df_analysis['duration_category'] = pd.cut(
        df_analysis['duration_minutes'],
        bins=duration_bins,
        labels=duration_labels
    )

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
            plot_data['duration_category'].unique(),
            rotation=45
        )

        axes[row, col].set_title(title)
        if metric in ['view_count', 'like_count', 'num_comms']:
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
    plt.tight_layout()
    plt.show()

    return plt