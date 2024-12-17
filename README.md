# YouTube's Inner World: A Data Exploration of Channel and Video Dynamics

## Data Story & Results Notebook

Check out our data story [here](https://necojac.github.io/ada-story/)

Check out our results notebook [here](https://github.com/epfl-ada/ada-2024-project-laughtodeath666/blob/main/results.ipynb)

## Abstract

This project explores strategies to enhance the popularity of YouTube channels using the YouNiverse dataset. Our analysis focuses on two key factors: video category diversity and optimal video duration, examining their correlation with engagement metrics such as views, likes, dislikes, comments, and subscriber growth. To provide a deeper understanding of video performance, we propose methods to quantify popularity more effectively. This comprehensive approach aims to offer actionable insights for content creators to optimize their strategies and maximize audience engagement. Additionally, during our analysis, unusual trends emerged, prompting an investigation into their potential connection to the rise of short videos. By addressing these anomalies, we aim to uncover the broader implications of evolving content preferences and provide creators with a data-driven foundation for navigating the dynamic YouTube ecosystem.

## Research Questions
1. How does video category diversity impact a YouTube channel? Will a more diverse video category be better, or will a more focused category be more popular?
2. What is the optimal video duration for a YouTube video? How does video duration correlate with key engagement metrics? Does the optimal duration vary across different video categories?
3. How do external factors affect video popularity trends? Will the rise of short video platforms have an impact on the duration of videos on Youtube?

## Methodology
1. Sample Data: Considering the limitations of computing resources. We select a subset of YouTube channels from df_channels_en.tsv. The sample dataset includes channel_id with varying video category diversity and have the same proportion of video categories as the original dataset. Then, we use these channel_ids to filter data from df_timeseries_en.tsv, num_comments.tsv and yt_metadata_helper.feather.

1. Data Preprocessing: We identified missing or NaN values in each data frame and addressed these by imputing values. Specifically, missing values were filled using the mean computed per 'channel_id' to maintain consistency across related entries.

1. Statistical Analysis: 

   a) Mean, median and standard deviation of different inputs are calculated for further comparison and analysis.

   b) The Shannon Entropy Formula is used to calculate the diversity of a channel's content based on the categories of its videos.  A higher entropy value indicates that the content is more evenly distributed across categories. A lower entropy value means the content is more concentrated in a few categories, indicating less diversity.

   c) Analysis of Variance : To explore the relationships between diversity and performance within different categories, relevant hypotheses are proposed, and then the corresponding F-values and p-values are calculated to categorize whether content strategy (Diversified, Mixed, Specialized) has a noticeable impact on metrics such as view count, engagement rate, and subscriber growth. 

1. Metrics Calculation: 

   a) The Engagement Rate is calculated as the ratio of like_count to view_count.

   b) The Popularity Score is calculated based on the weights of different normalized factors to reflect overall video performance.

1. Visualization: 

   a) Scatter Plot: The plots explore the relationship between content diversity and average view_count for each channel,  the relationship between content diversity and engagement rate, the relationship between content diversity and weekly subscriber growth and the relationship between content diversity and weekly view growth. Logarithmic Scale is used for better visualization.

   b) Bar chart: Bar charts are used to show the mean view count, mean weekly subscriber growth, mean weekly view growth and mean engagement rate of different content strategies.

   c) Line plot: Optimal video length for different categories are shown in line plot by different metrics. Also, the average popularity score over time is plotted for each video duration category.

   d) Box plot: Several box plots are generated to visualize how different the video performances vary by video duration categories.

## Timeline

1. 16.11.2024-30.11.2024

   Diversity Outlier Investigation.

   Analyze highly successful diversified channels, study their unique strategies and characteristics.

2. 30.11.2024-06.12.2024

   Further analyze the data, especially in conjunction with the differentiation of audience preferences for videos of different durations from 2016 to 2019. 

   Find external reasons that lead to YouTube viewers' preference for duration.

3. 07.12.2024-13.12.2024

   Build a basic framework to tell the data story. 

   Relate the content strategies with the analysis of duration preferences for better storytelling perspective.

4. 14.12.2024-20.12.2024

   Select relevant data and visualization tools, complete and improve the final story presentation.

## Organization within the team
- Shengze: Diversity Outlier Investigation, organize the data story blog
- Shuhua: Study the impact of events on preference duration, organize the GitHub repository
- Xuanrui: Frequency domain analysis, time series fitting and forecasting, organize the data story blog
- Xinran: Relate the content strategies with the analysis of duration preferences, organize the data story
- Xinyue: Build a framework to tell the data story, organize the data story


## Quickstart

```bash
# clone project
git clone <git@github.com:epfl-ada/ada-2024-project-laughtodeath666.git>
cd <project repo>

# [OPTIONAL] create conda environment
conda create -n <env_name> python=3.10 
conda activate <env_name>


# install requirements
pip install -r pip_requirements.txt
```

Download the Sampled dataset from this following link and add the data files to the directory 'data'.
> [Sampled Data](https://drive.google.com/drive/folders/1MgLEmq6yTFvKvjFenrXe93HaLlVR8xqw?usp=sharing)

If you want to get the original unsampled dataset, you can download it from the following link.
> [Original Data](https://github.com/epfl-dlab/YouNiverse/tree/master)

## Project Structure

The directory structure of this project looks like this:

```
├── data                        <- Project data files
│
├── image/                      <- Image files 
│   
├── src/                        <- Source code
│   ├── data/                   <- Data loading and preprocessing
│   │   ├── sample_data.ipynb   <- Sample data generation
│   │   ├── loader.py           <- Data loading functions
│   │   └── preprocessor.py     <- Data preprocessing functions
│   │
│   ├── utils/                  <- Analysis utilities
│   │   ├── content.py          <- Content diversity strategy analysis
│   │   └── duration.py         <- Video duration analysis
|   |   └── frequency.py        <- Frequency domain analysis
│
├── results.ipynb               <- a well-structured notebook showing the results
│
├── .gitignore                  <- List of files ignored by git
├── pip_requirements.txt        <- File for installing python dependencies
└── README.md
```

