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
#### Example of Using Predicted Job Time Completion:
- When a new job is created, the predictive model(s) will predict an expected time to complete a job (for actual work time - not waiting, driving, hauling, etc.) and that predicted time will be populated on the job record
- The app uses GPS to track the location of the worker completing the job, so when the worker enters the geo-fence around the job site, the 'work time' clock starts
- If the work time exceeds the predicted time to complete that job by a certain threshold, the geo-fence around the job site will turn yellow indicated a job is taking longer than predicted. If it exceeds a second threshold, the geo-fence will turn red and a notification will be sent to a manager
- The service contractors performing the jobs, and the operators overseeing the well or production site will use this information to increase visibility, efficiency, and safety so operations can be better managed

***PUT SOMETHING HERE TO SHOW GEOFENCE PICTURE AND HOW IT WORKS***

## Data Source: <a name="data_source"></a>
### Engage Mobilize provided a job dataset, which included ~11,000 jobs with 100+ data fields across 3 job types. Each job that was completed included the amount of time it took to complete (i.e., "workTime") which is the target variable.

The data provided includes:
- 11,084 records, where each record is a job
- Each record has 112 fields ranging from start / complete dates, company / worker identifiers, location details, job types, volumes, and more
- The data is for well and production sites in Utah and North Dakota

## Exploratory Data Analysis: <a name="eda"></a>
### EDA revealed that ~90% of the jobs were water and slickline job types, and about half of those jobs had a completed status and a work time value > 0. For each equipment type, the mean time to complete a water job is 0.87 hrs and a slickline job is 3.34 hrs.

#### What are slickline and water jobs?
**Slickline**: A slickline is a thin cable introduced into a well to deliver and retrieve tools downhole. Service contractors are brought in to complete this service at well sites.
**Water**: Water is used in the drilling / fracking process, so service contractors are hired to pick up used water, and drop off clean water at the well and production sites.

#### Job Completion Average Times & Insights:
- There are different equipment types to complete both water and slickline jobs, which impact the amount of time it takes to complete a job
- The mean time to complete a job was aggregated by equipment type
- Mean time to complete a job by equipment type is the baseline to compare a predictive model
- Key question: is a predictive model better than simply using the mean to predict job time completion?

|**Job Type** | **Equipment Type** | **Job Completion Mean** | **Job Completion Std Dev** | 
|---------|----------------|---------------------|------------------------|
|Slickline|  226           |    3.33 hrs         |       1.03 hrs         |
|Slickline|  235           |    4.01 hrs         |       1.20 hrs         |
|Slickline|  239           |    2.91 hrs         |       0.68 hrs         |
|Slickline|  249           |    4.23 hrs         |       0.70 hrs         |
|Slickline|  250           |    2.54 hrs         |       0.40 hrs         |
|**Slickline - Total**|  **N/A**           |    **3.34 hrs**         |       **1.04 hrs**         |
|Water    |  9             |    3.11 hrs         |       0.50 hrs         |
|Water    |  161           |    0.86 hrs         |       0.66 hrs         |
|**Water - Total**    |  **N/A**           |    **0.87 hrs**         |       **0.67 hrs**         |


**Insights**:
- The equipment type for the same job type impacts how long it takes to complete a job
- However, the standard deviation is significant, which indicates that some service contractors are much faster or slower than others - opportunity to shed light on this to improve service times and narrow the variance
- See below for a histogram of work times for slickline and water jobs that highlights the variance across jobs

<img src="https://github.com/mikeirvine/Capstone-Job-Time-Predictor/blob/master/imgs/work_time_dist.png">

#### Feature Analysis: 
- Based on experience, the Engage Mobilize team thought that job type, equipment type, volume, location, and time of year would be the most predictive factors of how long a job should take to complete
- Most other types of fields in the dataset do not appear to be predictive of the time it would take to complete a job
**Key Fields to Consider for a Model:**
- **Job Type**: Analysis revealed that the dataset included 7593 water jobs (Production Water-Bbl) and 3280 slickline jobs. There were fewer than 250 records across seven other job types
- **Location**: Each job has detailed location information, but there is also a higher level location field called 'businessRegionID' where >90% of jobs are in 5 regions
- **Time of Year**: The dataset included jobs completed over the last 10 months - the month of a job should be a good predictor as a job in the winter may take longer than a job in the summer
- **Volume**: Slickline and water jobs both use a 'volume' field which indicates the volume of work completed (e.g., volume of water picked up or dropped off at a well or production site). However, this field is NOT entered until AFTER a job is completed, so it cannot be used as a predictor
- **Equipment Type**: Slickline jobs use 5 types of equipment and water jobs use two types of equipment (although predominantly just one type). This is a critical predictor of work time as different types / sizes of equipment (e.g., size of a water truck) complete different variations of that job type, so work time will vary
- **Amount**: Amount is only relevant for slickline jobs, and it is the charge per barrel by the service contractor. This information is known when a new job record is created, so it can be used as a predictor
- **Work Time**: Work time is the amount of time of work it took (not including en route, hauling, or wait times) to complete the job, measured in hours (with decimals every 15 mins (e.g., 1.5 hours is 1 hour and 30 mins)

## Feature Engineering: <a name="feature_eng"></a>
### Feature engineering was limited to creating three types of dummy features based on equipment type, location, and time categorical variables
- **Equipment Type**: Created dummy features based on the equipment type categorical variable (5 equipment type dummy features for slickline jobs and 2 equipment type dummy features for water jobs
- **Location**: Each job had high level and detailed location information, so I focused on creating dummy variables for the region location variable. Most jobs are located in 5 regions, and the rest are located across a few dozen regions. I created dummy features for the top 5 regions and then created an 'other' region for all other jobs.
- **Time**: Each job has a created date, and since the time of year may impact work time, I created a dummy feature for the month of year

The final feature set included: amount, equipment types, region, and month of year. As I tested models (details below), I tested different combinations of the feature set to see which combination optimized the models.

## Modeling: <a name="modeling"></a>
### Approach - *build separate models for water and slickline jobs*:
Given that the vast majority of the dataset contains only water and slickline jobs, and a key feature (amount) is only available for slickline jobs, I focused on building two separate models - one for slickline jobs and one for water jobs.
**Steps:**
- Filtered data to water and slickline jobs only, jobs that were marked as completed, and jobs with a work time value > 0. This reduced the dataset to ~2500 slickline jobs and ~3500 water jobs, after outliers +/- 3 standard deviations from the mean were removed
- Split the dataset into a slickline dataset and a water dataset
- Removed all fields except for the core feature set, which include features related to amount (slickline only), equipment types, region, and month of year
- Created dummy features for the categorical features
- Performed a train / test / split on each dataset, with a test size of 0.25
```python
    train_slick_df, test_slick_df = train_test_split(slick_df, test_size=0.25, random_state=42)
    train_water_df, test_water_df = train_test_split(water_df, test_size=0.25, random_state=42)
```
- Scaled / standardized the dataset using Sklearn's Standard Scaler
- Built 4 models to test results: Linear Regression, Lasso Regression, Random Forest and Gradient Boosting
- Used Sklearn's RandomizedSearchCV to perform a randomized grid search across parameters for the Random Forest and Gradient Boosting models - used 3 cross validations and 1000 iterations (how many randomized searches to test) to identify optimal parameters. **Gradient Boosting example:**
```python
# Number of trees in random forest
n_estimators_rf = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 20)]
# Number of features to consider at every split
max_features_rf = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth_rf = [int(x) for x in np.linspace(5, 110, num = 15)]
max_depth_rf.append(None)
# Minimum number of samples required to split a node
min_samples_split_rf = [2, 5, 10, 15]
# Minimum number of samples required at each leaf node
min_samples_leaf_rf = [1, 2, 4, 8]
# Method of selecting samples for training each tree
bootstrap_rf = [True, False]
# Create the random grid
random_grid_rf = {'n_estimators': n_estimators_rf, 'max_features': max_features_rf, 'max_depth': max_depth_rf, 'min_samples_split': min_samples_split_rf, 'min_samples_leaf': min_samples_leaf_rf, 'bootstrap': bootstrap_rf}
# Create the random forest object
rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid_rf, n_iter = 1000, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train_slick_std, y_train_slick)
# View the best parameters
print(rf_random.best_params_)
# Save the best model (configured with the best performing combination of parameters)
best_rf = rf_random.best_estimator_
# Predict work time on the test set using the best model, and check the RMSE results
y_pred_test_s = best_rf.predict(X_test_slick_std)
test_rmse = np.sqrt(mean_squared_error(y_test_slick, y_pred_test_s))
```
- Assessed model results using root mean squared error (RMSE) of the predicted work time vs the actual work time
- Compared results to mean work time for the equipment type to answer the question on whether a model is more predictive than just simply using the mean for that equipment type to predict work time

## Results: <a name="results"></a>
### After testing various combinations of features and models, a random forest model performed best with the full feature set (amount, region, equipment type, and month) for the slickline jobs, whereas no model could outperform simply using the mean for the water jobs
#### Slickline Model:
- I tested various combinations of features (amount, equipment type, region, and month) and various parameters for the Linear, Lasso, Random Forest and Gradient Boosting algorithms
- Random Forest and Gradient Boosting performed the best, with Random Forest edging out Gradient Boosting by a little in terms of RMSE
- Both Random Forest and Gradient Boosting models performed the best with the full feature set, compared to reducing the feature set to just amount and equipment type
- Optimal parameters were identified using a randomized grid search of 1000 iterations - parameters below:
```python
rf_s = RandomForestRegressor(n_estimators=600, min_samples_split=10, min_samples_leaf=2, max_features='auto', max_depth=100, bootstrap=True)
gbr_s = GradientBoostingRegressor(n_estimators=600, min_samples_split=15, min_samples_leaf=8, max_features=None, max_depth=7, learning_rate=0.005)
```
*Key Takeaways*:
- The best RMSE on the test set for the Random Forest model was **0.354**, and the best for Gradient Boosting was **0.362**, which indicates that both models' predictions, on average, are ***~20 minutes*** off the actual work time value
- These are very accurate results considering that the average slickline job takes ***3 hours and 20 minutes*** to complete
- If the mean slickline job work time was used to predict the work time for a job, the RMSE is **0.863**, which is, on average, ***~50 minutes*** off the actual work time value
- The machine learning models' predictions are more than twice as accurate compared to just using the mean work time, so there is evidence that the features are predictive and there is a benefit to using a machine learning model to predict work time
- Detailed results across models against the test set:

|**Model**        | **RMSE**        |
|-----------------|-----------------|
|Random Forest    |  0.354          |
|Gradient Boosting|  0.362          |
|Linear Regression|  0.440          |
|Lasso Regression |  0.527          |
|Slickline Job Mean |  0.863          |

What features are the most important in predicting work time?
- Looking at the feature importances for the random forest reveals that amount is by far the most importance feature, with the rest of the features providing significantly less impact. Below are the top five in terms of importance.

|**Feature**      | **Importance**  |
|-----------------|-----------------|
|Amount           |  0.869          |
|Equipment ID 235 |  0.060          |
|February         |  0.013          |
|January          |  0.008          |
|Region - Other   |  0.005          |




#### Water Model:
- For the water model, a key challenge was not having a numerical feature since 'amount' only applied to slickline jobs, and 'volume' (of barrels of water) is not entered on a job record until after the work time is completed (so it cannot be used in the model as predictions are needed when the job is first created)
- I tried various combinations of the feature set available for water jobs (region, equipment type, and job month) but each model's performance wasn't any better when compared to the mean work time for a water job

*Key Takeaways*:
- The features available to predict water job work time don't add any predictive power beyond just using the mean work time
- The mean work time RMSE is ***0.648*** which and every model was about the same as each model was basically predicting the mean as well
- Using the mean work time RMSE is a very *inaccurate* predictor given that the mean work time for a water job is only 0.87 hours
- My assumption is that service contractor is likely the biggest indicator of work time for a water job as there's is such a wide variance in work time across water jobs

Would adding an estimated volume when the job is created help improve predictions for water jobs?
- I attempted putting 'volume' back in the water model just to see if it was an accurate predictor of water job work time
- Surprisingly, there was NOT any improvement in RMSE for the water model, even when using 'volume'
- Intuitively, one would think that the number of barrels of water a truck picked up or dropped off would impact the amount of work time, but this is not the case
- This further validates that service contractor is likely the best predictor as there is a lot of variance across water service contractors in terms of work time
- Detailed results across models against the test set:

|**Model**        | **RMSE**        |
|-----------------|-----------------|
|Random Forest    |  0.649          |
|Gradient Boosting|  0.663          |
|Linear Regression|  0.655          |
|Lasso Regression |  0.656          |
|Water Job Mean   |  0.648          |





Text

<img src="hyperlink">

Text

<img src="hyperlink">

## Future Work: <a name="future_work"></a>
### Text

- Segment service providers by performance - there's significant std deviation for both water jobs and slickline jobs - may be able to identify companies with a trend of significantly higher work times for the same type of job
- why is volume not a predictor? likely there's just variance across service providers in terms of performance / efficiency...instead of work time being based on volume
