import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


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
        df_no_outliers = job_df.loc[(job_df['workTime'] >= job_df['workTime_mean'] - 2 * job_df['workTime_std']) & (job_df['workTime'] <= job_df['workTime_mean'] + 2 * job_df['workTime_std']), :]
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

    def transform(self, job_df, time_df, job_types, cols_to_keep):
        '''
        Input:
        Output:
        '''
        # remove jobs that do NOT have a complete status
        job_df = job_df.loc[job_df['status'] == 'complete', :]

        # remove jobs that have zero for workTime
        job_df = job_df.loc[job_df['workTime'] != 0, :]

        # keep only records with a job type in 'job_type' list
        job_df = job_df.loc[job_df['businessEquipmentID'].isin(job_types), :]

        # create multiple wellSite location columns
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

        # add total_time feature to job_df (added only for analysis of df)
        job_df = pd.merge(job_df, time_df, how='left', on='jobID')

        # drop all columns except 'cols_to_keep'
        job_df = job_df[cols_to_keep]

        # create month of job column
        job_df.loc[:, 'job_month'] = job_df['createdDate'].dt.month
        job_df.drop('createdDate', axis=1, inplace=True)

        # add mean & std dev columns for each businessEquipmentID to remove outliers
        workTime_stddev = job_df.groupby('businessEquipmentID').agg({'workTime':'std'}).reset_index()
        workTime_stddev.rename(columns={'workTime':'workTime_std'}, inplace=True)
        workTime_mean = job_df.groupby('businessEquipmentID').agg({'workTime':'mean'}).reset_index()
        workTime_mean.rename(columns={'workTime':'workTime_mean'}, inplace=True)
        job_df = pd.merge(job_df, workTime_stddev, how='left', on='businessEquipmentID')
        job_df = pd.merge(job_df, workTime_mean, how='left', on='businessEquipmentID')

        # remove outliers
        job_df = self.remove_outliers(job_df)

        # recalculate mean & std dev columns for each businessEquipmentID w/ outliers removed
        job_df.drop(['workTime_std', 'workTime_mean'], axis=1, inplace=True)
        workTime_stddev = job_df.groupby('businessEquipmentID').agg({'workTime':'std'}).reset_index()
        workTime_stddev.rename(columns={'workTime':'workTime_std'}, inplace=True)
        workTime_mean = job_df.groupby('businessEquipmentID').agg({'workTime':'mean'}).reset_index()
        workTime_mean.rename(columns={'workTime':'workTime_mean'}, inplace=True)
        job_df = pd.merge(job_df, workTime_stddev, how='left', on='businessEquipmentID')
        job_df = pd.merge(job_df, workTime_mean, how='left', on='businessEquipmentID')

        # create dummies
        job_df = self.create_dummies('businessEquipmentID', job_df, 'businessEquipmentID')

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

if __name__ == '__main__':

    data_cleaner = WorkTimeCleaner()

    # load data
    job_df = data_cleaner.load_data('/Users/mwirvine/Galvanize/dsi-immersive/Capstone-EngageMob-Data/DATA.xlsx')

    # keep only records with a job type in 'job_type' list
    job_types = [161, 239, 226, 235, 249, 250, 255]
    # keep only these columns as features and the target
    cols_to_keep = ['jobID', 'businessEquipmentID', 'volume', 'equipmentName', 'quantity', 'amount', 'workTime', 'businessRegionID', 'createdDate', 'wellSite_loc1', 'wellSite_loc2', 'total_time']

    time_df = data_cleaner.get_time_df()

    job_df = data_cleaner.transform(job_df, time_df, job_types, cols_to_keep)

    job_df['total_time_hrs'] = job_df['total_time'] / 60
    job_df['delta'] = job_df['workTime'] - job_df['total_time_hrs']
    #job_df = job_df.loc[job_df['delta'] < .25,:]

    '''TEST MODELS'''
    X = job_df.drop(['jobID', 'equipmentName', 'workTime', 'businessRegionID', 'wellSite_loc1', 'wellSite_loc2', 'job_month', 'workTime_std', 'workTime_mean', 'businessEquipmentID', 'total_time', 'total_time_hrs', 'delta'], axis=1).values
    y = job_df['workTime'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    print(lr_model.score(X_test, y_test))
    lr_pred_train = lr_model.predict(X_train)
    lr_pred_test = lr_model.predict(X_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, lr_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, lr_pred_test))
    print('{} RMSE train results: {:.3f}'.format('linear', train_rmse))
    print('{} RMSE test results: {:.3f}'.format('linear', test_rmse))

    rf_model = RandomForestRegressor(n_estimators=300)
    rf_model.fit(X_train, y_train)
    print(rf_model.score(X_test, y_test))
    rf_pred_train = rf_model.predict(X_train)
    rf_pred_test = rf_model.predict(X_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, rf_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, rf_pred_test))
    print('{} RMSE train results: {:.3f}'.format('rf', train_rmse))
    print('{} RMSE test results: {:.3f}'.format('rf', test_rmse))


'''TOMORROW:
- need to create a hold out set
- call Rob in AM
- dummy more features
'''


    # for later
    corr_matrix = job_df.corr()
    corr_matrix['workTime'].sort_values(ascending=False)


'''
NOTES:

EDA:
- businessEquipmentID is the job type (16 job types; top 6 job types make up most of the jobs. REMOVE everything but the top 6.
- status column is the job status - for fitting a model, remove everything that is not a 'complete' status
- 3592 "complete" jobs have 0 for workTime. What to do? REMOVE them.

- removed outliers <> 3 std devs (can try 2 to see if model improves)

Charts:
- distribution of workTime by businessEquipmentID

Questions:
- Need workTime starttime and endtime to get more precise? Ask Rob. USE FIRST FILE FROM ROB!
- difference between equipmentName and businessEquipmentID. Name has 3 after my filtering (Production Water-Bbl, Slickline, Swabbing). Are we predicting workTime by name or by ID (there are 6 ID values after my filtering)
- What is quantity and amount?
- businessregionid vs wellSite
- two createdDates - difference?
- what data points are available when job is first created? want to ensure volume, amount, etc are there or else there will be data leakage
- if all jobs are completed (after i filter), and all IDs are unique, why do some jobs have timeTypeID 3 and 4 (work and haul)? Checked and all jobsNos and IDs are unique (no dups)
- Cannot get the workTime and my calculated total_time to line up perfectly - look at jobID 5236 as an example in the time and latlong tab. workTime is .75 but diff between startTime and endTime is only 20 mins as there's a blank entry for endTime. Tried taking out all jobIDs with a record that does not have an endTime. Pairs the data down to 2805 records (from 5932 records), and there are 252 records with a delta of > .25 meaning the workTime and total_time aren't aligned. Note: I divided total_time by 60, then subtracted that value from workTime to calculate a delta. Two options: build model only on those records where there is less .25 diff between total_time and workTime (2805 records - 252 records that don't align), or build the model on workTime as is (5932 records). Recommend that Rob add a column with workTime in minutes (or at least a precise total_time in hours:minutes). Also, recommend to build the model on workTime for now

Model Results:

- Test 1: simple linear model and rf model using workTime as target
- Features: volume, quantity, amount, businessEquipmentID dummies
- Linear test RMSE: 0.577
- Random forest RMSE: 0.564 (n_estimators=300)
- Conclusion: about 30 mins off

- Test 2: simple linear model and rf model using workTime as target
- Changes: only change from test 1 was removing outliers >< 2 std devs (instead of 3)
- Features: volume, quantity, amount, businessEquipmentID dummies
- Linear test RMSE: 0.563
- Random forest RMSE: 0.517 (n_estimators=300)
- Conclusion: model improved with removing more outliers at +- 2 std devs. Still about 30 mins off

- Test 3: simple linear model using total_time as target (only kept records w/in .25)
- Features: volume, quantity, amount, businessEquipmentID dummies
- Linear test RMSE: 40.495
- Conclusion: not really a difference in RMSE if I use total_time vs workTime (which makes sense)
'''
