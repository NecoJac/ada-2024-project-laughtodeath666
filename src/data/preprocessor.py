# src/data/preprocessor.py

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