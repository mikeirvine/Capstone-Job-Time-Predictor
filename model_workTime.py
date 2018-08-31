import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV



'''
Script Overview:
Model.py

'''

class WorkTimeCleaner():

    def __init__(self):
        pass

    def load_data(self, filename, filetype='xlsx'):
        '''
        Input: file type and filename (w/ folder path) of the job extract
        Output: pandas dataframe of the job extract
        '''
        if filetype == 'csv':
            df = pd.read_csv(filename)
        elif filetype == 'xlsx':
            df = pd.read_excel(filename)
        return df

    def remove_outliers(self, job_df):
        '''
        Input:
        Output:
        '''
        df_no_outliers = job_df.loc[(job_df['workTime'] >= job_df['workTime_mean'] - 3 * job_df['workTime_std']) & (job_df['workTime'] <= job_df['workTime_mean'] + 3 * job_df['workTime_std']), :]
        return df_no_outliers

    def create_dummies(self, dummy_col, job_df, col_prefix):
        '''
        Input: dummy_col: categorical feature column to create dummies, df: dataframe to add dummies, col_prefix: string to add as prefix to dummy column name
        Output: updated dataframe with categorical features replaced as dummy features
        '''
        dum_col = job_df[dummy_col]
        dummies = pd.get_dummies(dum_col, prefix=col_prefix)
        #df = df.drop([dummy_col], axis=1)
        df_w_dummies = job_df.merge(dummies, left_index=True, right_index=True)
        return df_w_dummies

    def calc_mean_std(self, job_df):
        '''
        Input:
        Output:
        '''
        # add mean & std dev columns for each businessEquipmentID to remove outliers
        workTime_stddev = job_df.groupby('businessEquipmentID').agg({'workTime':'std'}).reset_index()
        workTime_stddev.rename(columns={'workTime':'workTime_std'}, inplace=True)
        workTime_mean = job_df.groupby('businessEquipmentID').agg({'workTime':'mean'}).reset_index()
        workTime_mean.rename(columns={'workTime':'workTime_mean'}, inplace=True)
        job_df = pd.merge(job_df, workTime_stddev, how='left', on='businessEquipmentID')
        job_df = pd.merge(job_df, workTime_mean, how='left', on='businessEquipmentID')
        return job_df

    def split_location_details(self, job_df):
        '''
        Input:
        Output:
        '''
        wellSite_df_loc1 = job_df['wellSite'].str.split(' ', 1, expand=True)
        cols = ['wellSite_loc1', 'wellSite_loc2']
        wellSite_df_loc1.columns = cols
        wellSite_df_loc2 = wellSite_df_loc1['wellSite_loc2'].str.split('-', 1, expand=True)
        cols = ['wellSite_loc2', 'wellSite_loc3']
        wellSite_df_loc2.columns = cols
        wellSite_df_loc1.drop('wellSite_loc2', axis=1, inplace=True)
        wellSite_df_loc2.drop('wellSite_loc3', axis=1, inplace=True)
        job_df = pd.merge(job_df, wellSite_df_loc1, left_index=True, right_index=True)
        job_df = pd.merge(job_df, wellSite_df_loc2, left_index=True, right_index=True)
        return job_df

    def transform(self, job_df, time_df, job_types, cols_to_keep, model_name):
        '''
        Input:
        Output:
        '''
        # remove jobs that do NOT have a complete status
        job_df = job_df.loc[job_df['status'] == 'complete', :]

        # remove jobs that have zero for workTime
        job_df = job_df.loc[job_df['workTime'] != 0, :]

        # keep only records with a job type in 'job_types' list
        job_df = job_df.loc[job_df['equipmentName'].isin(job_types), :]

        # create multiple wellSite location columns
        job_df = self.split_location_details(job_df)

        # add total_time feature to job_df (added only for analysis of df)
        job_df = pd.merge(job_df, time_df, how='left', on='jobID')

        # drop all columns except 'cols_to_keep'
        job_df = job_df[cols_to_keep]

        # create month of job column
        job_df.loc[:, 'job_month'] = job_df['createdDate'].dt.month
        job_df.drop('createdDate', axis=1, inplace=True)

        # add workTime mean & std dev columns for each businessEquipmentID to remove outliers
        job_df = self.calc_mean_std(job_df)

        # remove workTime outliers (+- 3 std devs), then drop mean & std devs to recalc w/o outliers
        job_df = self.remove_outliers(job_df)
        job_df.drop(['workTime_std', 'workTime_mean'], axis=1, inplace=True)

        # recalculate mean/std dev with outliers removed
        job_df = self.calc_mean_std(job_df)

        # create region column from businessRegionID, which aggregates the regions with few records into an 'OTHER' bucket
        conditions = [
            job_df.loc[:, 'businessRegionID'] == 'UTE',
            job_df.loc[:, 'businessRegionID'] == 'NBU 1',
            job_df.loc[:, 'businessRegionID'] == 'NBU 9',
            job_df.loc[:, 'businessRegionID'] == '*CC',
            job_df.loc[:, 'businessRegionID'] == 'BONAN']
        choices = ['UTE', 'NBU 1', 'NBU 9', '*CC', 'BONAN']
        job_df.loc[:, 'region'] = np.select(conditions, choices, default='OTHER')

        # create dummies
        if model_name == 'slick':
            job_df = self.create_dummies('businessEquipmentID', job_df, 'businessEquipmentID')
            #job_df = self.create_dummies('businessRegionID', job_df, 'businessRegionID')
            #job_df = self.create_dummies('region', job_df, 'region')
            #job_df = self.create_dummies('job_month', job_df, 'job_month')
            #job_df = self.create_dummies('wellSite_loc1', job_df, 'wellSite_loc1')
        if model_name == 'water':
            job_df = self.create_dummies('businessEquipmentID', job_df, 'businessEquipmentID')
            #job_df = self.create_dummies('businessRegionID', job_df, 'businessRegionID')
            #job_df = self.create_dummies('region', job_df, 'region')
            #job_df = self.create_dummies('job_month', job_df, 'job_month')
            #job_df = self.create_dummies('wellSite_loc1', job_df, 'wellSite_loc1')

        # drop 5 records with nulls for businessRegionID
        job_df.dropna(inplace=True)

        return job_df

    def get_time_df(self):
        '''
        Input:
        Output:
        '''
        # create time_df, which includes total minutes of workTime
        time_df = pd.read_excel('/Users/mwirvine/Galvanize/dsi-immersive/Capstone-EngageMob-Data/Data Extract 24 Aug.xlsx', 'time and latlong')
        time_df = time_df.loc[time_df['timeType'] == 'Work', :]
        time_df['total_time'] = time_df['endTime'] - time_df['startTime']

        # aggregate jobIDs to get total_time
        time_df = time_df.groupby('jobID').agg({'total_time':'sum'}).reset_index()
        total_time = pd.DatetimeIndex(time_df['total_time'])
        total_time = total_time.hour * 60 + total_time.minute
        total_time = pd.DataFrame(total_time)
        time_df = pd.merge(time_df, total_time, left_index=True, right_index=True)
        time_df.drop('total_time_x', axis=1, inplace=True)
        time_df.rename(columns={'total_time_y':'total_time'}, inplace=True)

        return time_df

def run_k_fold(model, X_train, y_train):
	''' Returns error for k-fold cross validation. '''
	err, index, num_folds = 0, 0, 5
	kf = KFold(n_splits=num_folds)
	error = np.empty(num_folds)
	for train, test in kf.split(X_train):
		model.fit(X_train[train], y_train[train])
		pred = model.predict(X_train[test])
		error[index] = np.sqrt(mean_squared_error(pred, y_train[test]))
		index += 1

	return np.mean(error)

def prepare_Xy(train_df, test_df, cols_to_drop):
    '''
    Input:
    Output:
    '''
    features = train_df.drop(cols_to_drop, axis=1).columns
    X_train = train_df.drop(cols_to_drop, axis=1).values
    y_train = train_df['workTime'].values
    X_test = test_df.drop(cols_to_drop, axis=1).values
    y_test = test_df['workTime'].values
    return X_train, X_test, y_train, y_test, features

if __name__ == '__main__':

    data_cleaner = WorkTimeCleaner()

    '''LOAD & CLEAN DATA'''
    # load data
    job_df = data_cleaner.load_data('/Users/mwirvine/Galvanize/dsi-immersive/Capstone-EngageMob-Data/DATA.xlsx')

    # keep only records with a job type in 'job_types' list
    slick_job = ['Slickline'] # slickline only jobs
    water_job = ['Production Water-Bbl'] # water only jobs

    # use these columns to clean data and make dummies
    cols_to_keep = ['jobID', 'businessEquipmentID', 'volume', 'equipmentName', 'quantity', 'amount', 'workTime', 'businessRegionID', 'createdDate', 'wellSite_loc1', 'wellSite_loc2', 'total_time']

    # get time_df
    time_df = data_cleaner.get_time_df()

    # transform job_df into a slickline df and a water df
    slick_df = data_cleaner.transform(job_df, time_df, slick_job, cols_to_keep, 'slick')
    water_df = data_cleaner.transform(job_df, time_df, water_job, cols_to_keep, 'water')

    '''CODE TO CALCULATE TOTAL_TIME & DELTAS BETWEEN TOTAL_TIME & WORKTIME'''
    # # calculate delta between total_time and workTime for investigation
    # job_df.loc[:, 'total_time_hrs'] = job_df.loc[:, 'total_time'] / 60
    # job_df.loc[:, 'delta'] = job_df.loc[:, 'workTime'] - job_df.loc[:, 'total_time_hrs']
    # #job_df = job_df.loc[job_df['delta'] < .25,:]

    '''TRAIN/TEST/SPLIT'''
    # create train / test dfs for slick and water dfs
    train_slick_df, test_slick_df = train_test_split(slick_df, test_size=0.25, random_state=42)
    train_water_df, test_water_df = train_test_split(water_df, test_size=0.25, random_state=42)

    '''PREPARE MATRICES'''
    # prepare slickline X, y matrices
    slick_cols_to_drop = ['jobID', 'volume', 'equipmentName', 'workTime', 'businessRegionID', 'wellSite_loc1', 'wellSite_loc2', 'job_month', 'workTime_std', 'workTime_mean', 'businessEquipmentID', 'total_time', 'region', 'quantity']
    X_train_slick, X_test_slick, y_train_slick, y_test_slick, features_slick = prepare_Xy(train_slick_df, test_slick_df, slick_cols_to_drop)
    # prepare water X, y matrices
    water_cols_to_drop = ['jobID', 'amount', 'equipmentName', 'workTime', 'businessRegionID', 'wellSite_loc1', 'wellSite_loc2', 'job_month', 'workTime_std', 'workTime_mean', 'businessEquipmentID', 'total_time', 'region', 'quantity']
    X_train_water, X_test_water, y_train_water, y_test_water, features_water = prepare_Xy(train_water_df, test_water_df, water_cols_to_drop)

    '''STANDARDIZE / SCALE DATA'''
    # scale water data
    water_scaler = StandardScaler()
    water_scaler.fit(X_train_water, y_train_water)
    X_train_water_std = water_scaler.transform(X_train_water)
    X_test_water_std = water_scaler.transform(X_test_water)
    del X_train_water
    del X_test_water
    # scale slick data
    slick_scaler = StandardScaler()
    slick_scaler.fit(X_train_slick, y_train_slick)
    X_train_slick_std = slick_scaler.transform(X_train_slick)
    X_test_slick_std = slick_scaler.transform(X_test_slick)
    del X_train_slick
    del X_test_slick


    '''TEST MODELS - WATER'''
    linear_w = LinearRegression(n_jobs=-1)
    lasso_w = Lasso(alpha=.1)
    rf_w = RandomForestRegressor(n_estimators=500, max_features='sqrt', n_jobs=-1, min_samples_leaf=4, min_samples_split=8)
    #rf = RandomForestRegressor(n_estimators=200, min_samples_split=5, min_samples_leaf=4, max_features='auto', max_depth=10, bootstrap=True) # best params based on randomized search cv
    gbr_w = GradientBoostingRegressor(n_estimators=500, max_depth=5, learning_rate=.1, max_features='sqrt', min_samples_leaf=4, min_samples_split=8)
    models_w = [(linear_w, 'Linear - Water'), (lasso_w, 'Lasso - Water'), (rf_w, 'Random Forest - Water'), (gbr_w, 'Gradient Boosting - Water')]

    for model in models_w:
        model[0].fit(X_train_water_std, y_train_water)
        y_pred_train_w = model[0].predict(X_train_water_std)
        y_pred_test_w = model[0].predict(X_test_water_std)
        train_rmse_w = np.sqrt(mean_squared_error(y_train_water, y_pred_train_w))
        test_rmse_w = np.sqrt(mean_squared_error(y_test_water, y_pred_test_w))
        print('{} RMSE train results: {:.3f}'.format(model[1], train_rmse_w))
        print('{} RMSE test results: {:.3f}'.format(model[1], test_rmse_w))
        #test_job_df.loc[:, 'workTime_pred_' + model[1]] = y_pred_test
        #test_job_df['workTime_pred_' + model[1]] = y_pred_test
        #train_job_df['workTime_pred_' + model[1]] = y_pred_train

    # calculate rmse workTime_mean as baseline comparison
    y_pred_workTime_mean_w = test_water_df.loc[:, 'workTime_mean'].values
    rmse_workTime_mean_w = np.sqrt(mean_squared_error(y_test_water, y_pred_workTime_mean_w))
    print("workTime mean RMSE results for water model: {:.3f}".format(rmse_workTime_mean_w))

    # run KFold cross validation
    for model in models_w:
        kfold_error = run_k_fold(model[0], X_train_water_std, y_train_water)
        print('{} RMSE k-fold results for water model: {:.3f}'.format(model[1], kfold_error))

    print('----------------------------------------------------')

    '''TEST MODELS - SLICK'''
    linear_s = LinearRegression(n_jobs=-1)
    lasso_s = Lasso(alpha=.1)
    rf_s = RandomForestRegressor(n_estimators=600, min_samples_split=10, min_samples_leaf=2, max_features='auto', max_depth=100, bootstrap=True) # best params based on randomized search cv
    gbr_s = GradientBoostingRegressor(n_estimators=500, max_depth=5, learning_rate=.1, max_features='sqrt', min_samples_leaf=4, min_samples_split=8)
    models_s = [(linear_s, 'Linear - Slick'), (lasso_s, 'Lasso - Slick'), (rf_s, 'Random Forest - Slick'), (gbr_s, 'Gradient Boosting - Slick')]

    for model in models_s:
        model[0].fit(X_train_slick_std, y_train_slick)
        y_pred_train_s = model[0].predict(X_train_slick_std)
        y_pred_test_s = model[0].predict(X_test_slick_std)
        train_rmse_s = np.sqrt(mean_squared_error(y_train_slick, y_pred_train_s))
        test_rmse_s = np.sqrt(mean_squared_error(y_test_slick, y_pred_test_s))
        print('{} RMSE train results: {:.3f}'.format(model[1], train_rmse_s))
        print('{} RMSE test results: {:.3f}'.format(model[1], test_rmse_s))
        #test_job_df.loc[:, 'workTime_pred_' + model[1]] = y_pred_test
        #test_job_df['workTime_pred_' + model[1]] = y_pred_test
        #train_job_df['workTime_pred_' + model[1]] = y_pred_train

    # calculate rmse workTime_mean as baseline comparison
    y_pred_workTime_mean_s = test_slick_df.loc[:, 'workTime_mean'].values
    rmse_workTime_mean_s = np.sqrt(mean_squared_error(y_test_slick, y_pred_workTime_mean_s))
    print("workTime mean RMSE results for slick model: {:.3f}".format(rmse_workTime_mean_s))

    # run KFold cross validation
    for model in models_s:
        kfold_error = run_k_fold(model[0], X_train_slick_std, y_train_slick)
        print('{} RMSE k-fold results for slick model: {:.3f}'.format(model[1], kfold_error))

    # Random Forest randomized search
    # Number of trees in random forest
    n_estimators_rf = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features_rf = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth_rf = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth_rf.append(None)
    # Minimum number of samples required to split a node
    min_samples_split_rf = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf_rf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap_rf = [True, False]
    # Create the random grid
    random_grid_rf = {'n_estimators': n_estimators_rf,
                   'max_features': max_features_rf,
                   'max_depth': max_depth_rf,
                   'min_samples_split': min_samples_split_rf,
                   'min_samples_leaf': min_samples_leaf_rf,
                   'bootstrap': bootstrap_rf}

    rf = RandomForestRegressor()
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid_rf, n_iter = 200, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    # Fit the random search model
    rf_random.fit(X_train_slick_std, y_train_slick)
    print(rf_random.best_params_)
    best_rf = rf_random.best_estimator_
    y_pred_train_s = best_rf.predict(X_train_slick_std)
    y_pred_test_s = best_rf.predict(X_test_slick_std)
    train_rmse = np.sqrt(mean_squared_error(y_train_slick, y_pred_train_s))
    test_rmse = np.sqrt(mean_squared_error(y_test_slick, y_pred_test_s))
    print('{} RMSE train results: {:.3f}'.format('Best RF - Slick', train_rmse))
    print('{} RMSE test results: {:.3f}'.format('Best RF - Slick', test_rmse))

    # Number of trees in gradient boosting
    n_estimators_gb = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Learning rate
    learning_rate_gb = [x for x in np.linspace(0,.3,10)]
    # Number of features to consider at every split
    max_features_gb = ['auto', 'sqrt', None]
    # Maximum number of levels in tree
    max_depth_gb = [1,2,3,4,5,6,7,8,9,10]
    max_depth_gb.append(None)
    # Minimum number of samples required to split a node
    min_samples_split_gb = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf_gb = [1, 2, 4]

    # Create the random grid
    random_grid_gb = {'n_estimators': n_estimators_gb,
                   'max_features': max_features_gb,
                   'max_depth': max_depth_gb,
                   'min_samples_split': min_samples_split_gb,
                   'min_samples_leaf': min_samples_leaf_gb}

    gb = GradientBoostingRegressor()
    gb_random = RandomizedSearchCV(estimator = gb, param_distributions = random_grid_gb, n_iter = 200, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    # Fit the random search model
    gb_random.fit(X_train_slick_std, y_train_slick)
    print(gb_random.best_params_)
    best_gb = gb_random.best_estimator_
    y_pred_train_s = best_gb.predict(X_train_slick_std)
    y_pred_test_s = best_gb.predict(X_test_slick_std)
    train_rmse = np.sqrt(mean_squared_error(y_train_slick, y_pred_train_s))
    test_rmse = np.sqrt(mean_squared_error(y_test_slick, y_pred_test_s))
    print('{} RMSE train results: {:.3f}'.format('Best GB - Slick', train_rmse))
    print('{} RMSE test results: {:.3f}'.format('Best GB - Slick', test_rmse))



    '''TOMORROW:
    - for rob: look at the rmse of Production Water Bbl vs Slickline
    - need to create a hold out set
    - call Rob in AM
    - dummy more features
    - try some more models and tune
    - look at residuals
    - try model just for water
    - implement a k-fold
    - remove some of the wellSite_loc1 or businessRegionID that are sparse?
    '''

    # for later
    corr_matrix_slick = slick_df.corr()
    corr_matrix_slick['workTime'].sort_values(ascending=False)
    coeffs_linear_s = linear_s.coef_
    coeffs_lasso_s = lasso_s.coef_
    feats_coef_linear_s = sorted(zip(coeffs_linear_s, features_slick), reverse=True)
    feats_coef_lasso_s = sorted(zip(coeffs_lasso_s, features_slick), reverse=True)

'''
NOTES:

Pre-processiong:
- Agreed with Rob to only keep Production Water-Bbl (7593) and Slickline (3280) jobs. They're the vast majority of records (11084)
- businessEquipmentID is the job type (16 job types; top 6 job types make up most of the jobs. Removed everything but the top 6.
- status column is the job status - for fitting a model, removed everything that is not a 'complete' status
- 3592 "complete" jobs have 0 for workTime. Removed them
- removed outliers +/- 3 std devs - this is the norm per Frank
- tried to calculate workTime using startTime / endTime for 'Work' timeTypes, but could not get times to reconcile. Rob said older dates will have more issues. Agreed to just use workTime as the target variable and look into startTime / endTime later
- confirmed with Rob to calculate workTime averages by businessEquipmentID as that is the type of equipment (e.g., 120 barrel truck vs 180 barrel truck). equipmentName is the type of job
- quantity - unit of measure - water jobs are 1 = barrels (mostly), slick jobs are 10 = runs. Variable is categorical. Will drop this feature as there's little variation between job types
- amount - charge per barrel for slickline jobs ONLY (no values for water jobs) - this feature is known when the job record is created so it can be used as a predictor for slickline - drop for water
- volume - this is the amount of product that was picked up or dropped off - not filled in by worker until AFTER the work is completed - CANNOT use it model as it's data leakage

Charts:
- distribution of workTime by businessEquipmentID

Readme notes:
- Through EDA, found that amount (charge per barrel) is only relevant to slickline jobs.
- Volume is not entered until after the workTime is completed, so cannot be used as a predictor
- Mean is the best predictor for water, and surprisingly, volume is NOT a good predictor (even if we pretend volume was available when the job was created), which is an interesting finding -- shouldn't the volume of a water job be an indicator of how long the job takes?

Questions:

- businessregionid vs wellSite
- two createdDates - difference?
- for discussion later: cannot get workTime and startTime / endTime for timeType = 'Work' to reconcile. Rob said the issue is likely greater for older dates (platform was new and there were some issues). Can circle back on this question
- what is quantity?

Model Results:
NOTE: have a slickline model and a water model

- Test 1: first run on two models
- Water features: businessEquipmentID dummy
- Slick features = amount, businessEquipmentID dummy
Linear - Water RMSE train results: 0.663
Linear - Water RMSE test results: 0.648
Lasso - Water RMSE train results: 0.671
Lasso - Water RMSE test results: 0.656
Random Forest - Water RMSE train results: 0.664
Random Forest - Water RMSE test results: 0.650
Gradient Boosting - Water RMSE train results: 0.663
Gradient Boosting - Water RMSE test results: 0.648
workTime mean RMSE results for water model: 0.648
Linear - Water RMSE k-fold results for water model: 0.662
Lasso - Water RMSE k-fold results for water model: 0.669
Random Forest - Water RMSE k-fold results for water model: 0.666
Gradient Boosting - Water RMSE k-fold results for water model: 0.662
----------------------------------------------------
Linear - Slick RMSE train results: 0.549
Linear - Slick RMSE test results: 0.475
Lasso - Slick RMSE train results: 0.577
Lasso - Slick RMSE test results: 0.527
Random Forest - Slick RMSE train results: 0.408
Random Forest - Slick RMSE test results: 0.384
Gradient Boosting - Slick RMSE train results: 0.371
Gradient Boosting - Slick RMSE test results: 0.512
workTime mean RMSE results for slick model: 0.863
Linear - Slick RMSE k-fold results for slick model: 0.555
Lasso - Slick RMSE k-fold results for slick model: 0.586
Random Forest - Slick RMSE k-fold results for slick model: 0.439
Gradient Boosting - Slick RMSE k-fold results for slick model: 0.452
- CONCLUSION: slick is predictive, particularly because of the 'amount' feature, water model is just predicting the mean

- Test 2: added region as a dummy feature to water - no improvement
Linear - Water RMSE train results: 0.662
Linear - Water RMSE test results: 58837013939.158
Lasso - Water RMSE train results: 0.671
Lasso - Water RMSE test results: 0.656
Random Forest - Water RMSE train results: 0.666
Random Forest - Water RMSE test results: 0.651
Gradient Boosting - Water RMSE train results: 0.662
Gradient Boosting - Water RMSE test results: 0.646
workTime mean RMSE results for water model: 0.648
Linear - Water RMSE k-fold results for water model: 28739450382.945
Lasso - Water RMSE k-fold results for water model: 0.669
Random Forest - Water RMSE k-fold results for water model: 0.667
Gradient Boosting - Water RMSE k-fold results for water model: 0.661

- Test 3: added job_month as a dummy feature to water, removed region. no improvement
Linear - Water RMSE train results: 0.654
Linear - Water RMSE test results: 0.639
Lasso - Water RMSE train results: 0.671
Lasso - Water RMSE test results: 0.656
Random Forest - Water RMSE train results: 0.654
Random Forest - Water RMSE test results: 0.641
Gradient Boosting - Water RMSE train results: 0.654
Gradient Boosting - Water RMSE test results: 0.639
workTime mean RMSE results for water model: 0.648
Linear - Water RMSE k-fold results for water model: 0.655
Lasso - Water RMSE k-fold results for water model: 0.669
Random Forest - Water RMSE k-fold results for water model: 0.660
Gradient Boosting - Water RMSE k-fold results for water model: 0.655
- CONCLUSION: no features can predict water. In presentation, say water predictions should just be the average...but then add volume to show what a volume feature could do

- Test 4: added volume back as a feature to see if it is predictive (even though it's not availble at the time a job is created)
Linear - Water RMSE train results: 0.659
Linear - Water RMSE test results: 0.655
Lasso - Water RMSE train results: 0.671
Lasso - Water RMSE test results: 0.656
Random Forest - Water RMSE train results: 0.640
Random Forest - Water RMSE test results: 0.649
Gradient Boosting - Water RMSE train results: 0.624
Gradient Boosting - Water RMSE test results: 0.663
workTime mean RMSE results for water model: 0.648
Linear - Water RMSE k-fold results for water model: 0.663
Lasso - Water RMSE k-fold results for water model: 0.669
Random Forest - Water RMSE k-fold results for water model: 0.658
Gradient Boosting - Water RMSE k-fold results for water model: 0.663
CONCLUSION - found that volume is NOT a predictor...best predictor is just the average (region, month don't have an impact either)

SLICK ONLY

- Test 5: added region as a dummy feature (also with businessEquipmentID dummy)
Linear - Slick RMSE train results: 0.544
Linear - Slick RMSE test results: 0.470
Lasso - Slick RMSE train results: 0.577
Lasso - Slick RMSE test results: 0.527
Random Forest - Slick RMSE train results: 0.425
Random Forest - Slick RMSE test results: 0.428
Gradient Boosting - Slick RMSE train results: 0.348
Gradient Boosting - Slick RMSE test results: 0.469
workTime mean RMSE results for slick model: 0.863
Linear - Slick RMSE k-fold results for slick model: 0.553
Lasso - Slick RMSE k-fold results for slick model: 0.586
Random Forest - Slick RMSE k-fold results for slick model: 0.471
Gradient Boosting - Slick RMSE k-fold results for slick model: 0.466
- CONCLUSION: no improvement (rf got worse actually) when adding region

- Test 6: added month as a dummy, removed region
Linear - Slick RMSE train results: 0.534
Linear - Slick RMSE test results: 0.444
Lasso - Slick RMSE train results: 0.577
Lasso - Slick RMSE test results: 0.527
Random Forest - Slick RMSE train results: 0.403
Random Forest - Slick RMSE test results: 0.426
Gradient Boosting - Slick RMSE train results: 0.321
Gradient Boosting - Slick RMSE test results: 0.409
workTime mean RMSE results for slick model: 0.863
Linear - Slick RMSE k-fold results for slick model: 0.545
Lasso - Slick RMSE k-fold results for slick model: 0.586
Random Forest - Slick RMSE k-fold results for slick model: 0.465
Gradient Boosting - Slick RMSE k-fold results for slick model: 0.444
- CONCLUSION: region does improve for gradient boosting

- Test 7: added month and region as dummies
Linear - Slick RMSE train results: 0.530
Linear - Slick RMSE test results: 0.440
Lasso - Slick RMSE train results: 0.577
Lasso - Slick RMSE test results: 0.527
Random Forest - Slick RMSE train results: 0.435
Random Forest - Slick RMSE test results: 0.460
Gradient Boosting - Slick RMSE train results: 0.303
Gradient Boosting - Slick RMSE test results: 0.398
workTime mean RMSE results for slick model: 0.863
Linear - Slick RMSE k-fold results for slick model: 0.542
Lasso - Slick RMSE k-fold results for slick model: 0.586
Random Forest - Slick RMSE k-fold results for slick model: 0.501
Gradient Boosting - Slick RMSE k-fold results for slick model: 0.445
CONCLUSION: GB is getting much better...grid search it.

BEST MODELS:
RF (0.384) with only amount and businessEquipmentID
GB (0.398) with amount, businessEquipmentID, region and month

Best Models - GB and RF - TEST 1
- features: amount, businessEquipmentID, region and month
- RF: {'n_estimators': 600, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': 'auto', 'max_depth': 100, 'bootstrap': True}
Best RF - Slick RMSE train results: 0.340
Best RF - Slick RMSE test results: 0.353
- GB: {'n_estimators': 200, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 3}
Best GB - Slick RMSE train results: 0.362
Best GB - Slick RMSE test results: 0.373

Best Models - GB and RF - TEST 2
- features: amount, businessEquipmentID
{'n_estimators': 800, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 100, 'bootstrap': False}
Best RF - Slick RMSE train results: 0.369
Best RF - Slick RMSE test results: 0.390
{'n_estimators': 200, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 9}
Best GB - Slick RMSE train results: 0.372
Best GB - Slick RMSE test results: 0.481


- build two models, one for slickline and one for water
- slickline looks to be predictive
- if water is not predictive, then STILL use volume (but tell ROB that this is what he could predict if he had some sort of volume estimate). water estimate might be best just using the average. can present how volume could be predictive for water model and slick model IF Rob gathered an estimate when job was created



pd.options.mode.chained_assignment = None

'''
