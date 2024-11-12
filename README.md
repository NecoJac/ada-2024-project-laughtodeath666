
# Your project name
This is a template repo for your project to help you organise and document your code better. 
Please use this structure for your project and document the installation, usage and structure as below.

## Abstract

In this project, we aim to explore strategies to enhance the popularity of a YouTube channel using the Youniverse dataset. Our analysis focuses on two key aspects: the impact of video category diversity and the optimal video duration. Specifically, we investigate how these factors correlate with key engagement metrics, including the number of views, likes, dislikes, comments, the growth number of subscribers and views. To better capture video popularity, we also examine and propose methods for quantifying the degree of popularity. This comprehensive approach seeks to provide actionable insights for content creators to optimize their video strategies and achieve greater audience engagement.

## Research Questions
1. How does video category diversity impact a YouTube channel? Will a more diverse video category be better, or will a more focused category be more popular?
2. What is the optimal video duration for a YouTube video? How does video duration correlate with key engagement metrics? Does the optimal duration vary across different video categories?
3. How can we quantify the degree of popularity of a YouTube video? What are the key factors that contribute to video popularity? How can we measure if the degree of the popularity is precise?

## Methodology
1. Sample Data: Considering the limitations of computing resources. We select a subset of YouTube channels from df_channels_en.tsv. The sample dataset includes channel_id with varying video category diversity and have the same proportion of video categories as the original dataset. Then, we use these channel_ids to filter data from df_timeseries_en.tsv, num_comments.tsv and yt_metadata_helper.feather.

## Proposed Timeline

## Organisation within the Team

## Quickstart

```bash
# clone project
git clone <project link>
cd <project repo>

# [OPTIONAL] create conda environment
conda create -n <env_name> python=3.11 or ...
conda activate <env_name>


# install requirements
pip install -r pip_requirements.txt
```



### How to use the library
Tell us how the code is arranged, any explanations goes here.



## Project Structure

The directory structure of new project looks like this:

```
├── data                        <- Project data files
│
├── src                         <- Source code
│   ├── data                            <- Data directory
│   ├── models                          <- Model directory
│   ├── utils                           <- Utility directory
│   ├── scripts                         <- Shell scripts
│
├── tests                       <- Tests of any kind
│
├── results.ipynb               <- a well-structured notebook showing the results
│
├── .gitignore                  <- List of files ignored by git
├── pip_requirements.txt        <- File for installing python dependencies
└── README.md
```

