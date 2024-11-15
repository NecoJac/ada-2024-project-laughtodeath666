# src/data/loader.py
import pandas as pd


def load_data():
    """Load YouTube data from files"""
    df_id = pd.read_csv('./data/channel_ids_sampled.csv')
    df_channels = pd.read_feather('./data/df_channels_en_sampled.feather')
    df_timeseries = pd.read_feather('./data/df_timeseries_en_sampled.feather')
    df_comments = pd.read_feather('./data/num_comments_sampled.feather')
    df_metadata = pd.read_feather('./data/yt_metadata_en_sampled.feather')

    return df_id, df_channels, df_timeseries, df_comments, df_metadata
