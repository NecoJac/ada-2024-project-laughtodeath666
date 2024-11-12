import numpy as np
import pandas as pd

# fill missing values 
def fill_channel_means(df, columns_to_fill):
    """
    Fill NaN values in specified columns with the mean value of the same channel_id
    """
    # Create a copy to avoid modifying the original DataFrame
    df_filled = df.copy()
    
    # Calculate means for each channel_id
    channel_means = df.groupby('channel_id')[columns_to_fill].transform('mean')
    
    # Fill NaN values with channel means
    for col in columns_to_fill:
        df_filled[col] = df_filled[col].fillna(channel_means[col])
        
        # If any NaN values remain (for channels with all NaN values),
        # fill with the overall mean of the column
        if df_filled[col].isna().any():
            overall_mean = df[col].mean()
            df_filled[col] = df_filled[col].fillna(overall_mean)
    
    return df_filled

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

def calculate_channel_metrics(df_timeseries,filtered_metadata,filtered_timeseries):
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
    growth_metrics = df_timeseries.groupby('channel').agg({
        'delta_subs': ['sum', 'count'],  # Total growth and number of data points
        'delta_views': 'sum',
        'views': 'last',  # Latest views
        'subs': 'last'    # Latest subscribers
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

def calculate_popularity_score_vectorized(df,weight=[0.5,0.3,0.2]):
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
            weight[0] * (view_score / max_view) +
            weight[1] * (like_score / max_like) +
            weight[2] * (comment_score / max_comment)
    )

    return score

def calculate_popularity_score_pca(df):
    """
    Calculate popularity score using PCA
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    # Standardize data
    view_score = np.log1p(df['view_count'])
    like_score = np.log1p(df['like_count'])
    comment_score = np.log1p(df['num_comms'])


    max_view = np.log1p(df['view_count'].max())
    max_like = np.log1p(df['like_count'].max())
    max_comment = np.log1p(df['num_comms'].max())


    data = np.array([view_score, like_score, comment_score]).T
    pca=PCA(n_components=1)
    score_pca = pca.fit_transform(data)

    minmax_scaler = MinMaxScaler()
    score_pca = minmax_scaler.fit_transform(score_pca)

    return score_pca

def analyze_duration_metrics(df_metadata,df_comments,popularity_method='weighted_sum',popularity_weight=[0.5, 0.3, 0.2]):
    """
    Analyze relationships between video duration and various performance metrics
    """
    # Select only needed columns for merging to reduce memory usage
    df_metadata_subset = df_metadata[['display_id', 'duration', 'view_count', 'like_count','dislike_count', 'upload_date']].copy()
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
    if popularity_method == 'weighted_sum':
        df_analysis['popularity_score'] = calculate_popularity_score_vectorized(df_analysis,popularity_weight)
    elif popularity_method == 'pca':
        df_analysis['popularity_score'] = calculate_popularity_score_pca(df_analysis)

    # Create duration categories
    duration_bins = [0, 1, 3, 5, 10, 15, 30, 60, np.inf]
    duration_labels = ['0-1', '1-3', '3-5', '5-10', '10-15', '15-30', '30-60', '60+']
    df_analysis['duration_category'] = pd.cut(
        df_analysis['duration_minutes'],
        bins=duration_bins,
        labels=duration_labels
    )

    return df_analysis