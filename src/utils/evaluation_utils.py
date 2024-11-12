def identify_patterns(merged_metrics, percentile=75):
    """Identify patterns of successful channels"""
    
    # Define multi-dimensional success thresholds
    view_threshold = merged_metrics['view_count'].quantile(percentile/100)
    engagement_threshold = merged_metrics['engagement_rate'].quantile(percentile/100)
    sub_growth_threshold = merged_metrics['weekly_sub_growth'].quantile(percentile/100)
    view_growth_threshold = merged_metrics['weekly_view_growth'].quantile(percentile/100)
    
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
    print(f"Average main category ratio: {patterns['main_category_ratio']*100:.1f}%")
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