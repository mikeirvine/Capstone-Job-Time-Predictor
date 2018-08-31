# Well Site Job Time Predictor
Mike Irvine - Module 3 Capstone Project

August 31, 2018

Galvanize Data Science Immersive - Denver

## Executive Summary
- Text

<img src="hyperlink">

## Table of Contents

1. [Context & Key Question](#context_key_question)
2. [Data Source](#data_source)
3. [Exporatory Data Analysis](#eda)
4. [Feature Engineering](#feature_eng)
5. [Modeling](#modeling)
6. [Results](#results)
7. [Future Work](#future_work)
8. [References](#references)

## Context & Key Question: <a name="context_key_question"></a>
### How long should it take to complete a job at a production or well site? This is the question that Engage Mobilize, a digital field ticking solution provider, wants to share with its operator and service contractor customers to further improve the transparency of field operations.

INSERT PICTURE

#### Background:
- The oil and natural gas industry in the US is currently experiencing a wave of technology innovation that is rapidly decreasing the cost to drill
- Despite the wave of innovation in drilling, most operators and service contractors have limited visibility to the day to day costs and efficiency of field operations that support the well and production sites
- Transactions between operators and service contractors are still mostly handled through paper tickets, which limits companies' ability to efficiently track and manage operations at the sites
- An example of a service that is handled through paper transactions is dropping off and picking up water for drilling
#### Engage Mobilize:
- Engage Mobilize provides a digital field ticketing solution that enables operators and service contractors to have real-time visibility to field operations at well and production sites
- Engage Mobilize is looking to further improve their service by predicting the expected completion time jobs that are transacted through the platform
- Providing this insight to the ecosystem of companies that support or oversee field operations will further increase transparency, efficiency and safety
#### Using Machine Learning to Predict Job Time Completion:
- The objective of this project is to build a model that predicts how long a particular type of job should take when the job is entered into the system by a service contractor or operator
- Engage Mobilize currently tracks how long a particular job takes, but is not comparing that time to the expected completion time
- With a predicted completion time for each job, Engage Mobilize will use its geo-fencing and time tracking technology to trigger notifications to companies if the job is taking longer than anticipated
- This insight will help companies better manage operations through increased visibility, efficiency and safefy

***PUT SOMETHING HERE TO SHOW GEOFENCE PICTURE AND HOW IT WORKS***

## Data Source: <a name="data_source"></a>
### Engage Mobilize provided a job dataset, which included ~11,000 jobs with 100+ data fields across 3 job types. Each job that was completed included the amount of time it took to complete (i.e., "workTime") which is the target variable.

The data provided includes:
- Text

## Exploratory Data Analysis: <a name="eda"></a>
### EDA revealed that ~90% of the jobs were water and slickline job types, and about half of those jobs had a completed status and a work time value > 0. For each equipment type, the mean time to complete a water job is XXX and a slickline job is YYY.

#### Job Completion Average Times & Insights:
- There are different equipment types to complete both water and slickline jobs, which impact the amount of time it takes to complete a job
- The mean time to complete a job was aggregated by equipment type
- Mean time to complete a job by equipment type is the baseline to compare a predictive model
- Key question: is a predictive model better than simply using the mean to predict job time completion?

|Job Type | Equipment Type | Job Completion Mean | Job Completion Std Dev | 
|---------|----------------|---------------------|------------------------|
|Slickline|  226           |    3.33 hrs         |       1.03 hrs         |
|Slickline|  235           |    4.01 hrs         |       1.20 hrs         |
|Slickline|  239           |    2.91 hrs         |       0.68 hrs         |
|Slickline|  249           |    4.23 hrs         |       0.70 hrs         |
|Slickline|  250           |    2.54 hrs         |       0.40 hrs         |
|*Slickline - Total*|  N/A           |    3.34 hrs         |       1.04 hrs         |
|Water    |  9             |    3.11 hrs         |       0.50 hrs         |
|Water    |  161           |    0.86 hrs         |       0.66 hrs         |
|*Water - Total*    |  N/A           |    0.87 hrs         |       0.67 hrs         |


Insights:
- The equipment type for the same job type impacts how long it takes to complete a job
- However, the standard deviation is significant, which indicates that some service contractors are much faster or slower than others - opportunity to shed light on this to improve service times and narrow the variance

#### Feature Analysis: 
- Text

#### Text:
- Text

#### Approach - *focus on company-wide equipment demand for a subset of the large equipment category*:
- Filtered data to water and slickline jobs only, jobs that were marked as completed, and jobs with a work time value > 0

Text

|Stat |    Value 
|-------|----------------|
|count  |  3668 |
|mean |        15.6 |
|std    |      24.6 |
|min     |    1  |
|25%      |     4  |
|50%       |    8  |
|75%      |    17  |
|max      |  300  |


## Feature Engineering: <a name="feature_eng"></a>
### Text
Text

## Modeling: <a name="modeling"></a>
### Text

Key Highlights of Modeling Approach:
- Text

## Results: <a name="results"></a>
### Text

Summary results by model on test dataset:

|Model |    RMSE 
|-------|-------------|
|Linear  |  11.17 |
|Lasso |      11.18 |
|Random Forest    |      10.46 |
|Gradient Boosting     |    12.68  |
|MLP      |     12.19  |
|Same Month Avg      |     11.19  |

Text

Text

|Feature       |          Correlation    |
|--------------|-------------------------|
|units_rented    |                 1|
|same_month_avg_units_rented   |   0.94|
|same_month_avg_days_rented   |    0.89|
|prior_month_units_rented    |     0.82|
|prior_month_total_days_rented  |  0.78|
|rental_type_daily           |    -0.16|
|rental_type_weekly           |   -0.17|
|prior_month_avg_price_per_day |  -0.18|

Text

1. same_month_avg_units_rented
2. same_month_avg_days_rented
3. prior_month_total_days_rented
4. prior_month_units_rented
5. product_type_20-220

Text

<img src="hyperlink">

Text

<img src="hyperlink">

## Future Work: <a name="future_work"></a>
### Text

Text
