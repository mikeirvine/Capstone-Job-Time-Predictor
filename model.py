import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
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

    def remove_outliers(self, df):
        '''
        Input: dummy_col: categorical feature column to create dummies, df: dataframe to add dummies, col_prefix: string to add as prefix to dummy column name
        Output: updated dataframe with categorical features replaced as dummy features
        '''
        df_no_outliers = df.loc[(df['workTime'] >= df['workTime_mean'] - 3 * df['workTime_std']) & (df['workTime'] <= df['workTime_mean'] + 3 * df['workTime_std']), :]
        return df_no_outliers

    def create_dummies(self, dummy_col, df, col_prefix):
        '''
        Input: dummy_col: categorical feature column to create dummies, df: dataframe to add dummies, col_prefix: string to add as prefix to dummy column name
        Output: updated dataframe with categorical features replaced as dummy features
        '''
        dum_col = df[dummy_col]
        dummies = pd.get_dummies(dum_col, prefix=col_prefix)
        #df = df.drop([dummy_col], axis=1)
        df_w_dummies = df.merge(dummies, left_index=True, right_index=True)
        return df_w_dummies

    def transform(self, df, job_types, cols_to_keep):
        '''
        Input:
        Output:
        '''
        # remove jobs that do NOT have a complete status
        df = df.loc[df['status'] == 'complete', :]

        # remove jobs that have zero for workTime
        df = df.loc[df['workTime'] != 0, :]

        # keep only records with a job type in 'job_type' list
        df = df.loc[df['businessEquipmentID'].isin(job_types), :]

        # create multiple wellSite location columns
        wellSite_df_loc1 = df['wellSite'].str.split(' ', 1, expand=True)
        cols = ['wellSite_loc1', 'wellSite_loc2']
        wellSite_df_loc1.columns = cols
        wellSite_df_loc2 = wellSite_df_loc1['wellSite_loc2'].str.split('-', 1, expand=True)
        cols = ['wellSite_loc2', 'wellSite_loc3']
        wellSite_df_loc2.columns = cols
        wellSite_df_loc1.drop('wellSite_loc2', axis=1, inplace=True)
        wellSite_df_loc2.drop('wellSite_loc3', axis=1, inplace=True)
        df = pd.merge(df, wellSite_df_loc1, left_index=True, right_index=True)
        df = pd.merge(df, wellSite_df_loc2, left_index=True, right_index=True)

        # drop all columns except 'cols_to_keep'
        df = df[cols_to_keep]

        # create month of job column
        df.loc[:, 'job_month'] = df['createdDate'].dt.month
        df.drop('createdDate', axis=1, inplace=True)

        # add mean & std dev columns for each businessEquipmentID
        workTime_stddev = df.groupby('businessEquipmentID').agg({'workTime':'std'}).reset_index()
        workTime_stddev.rename(columns={'workTime':'workTime_std'}, inplace=True)
        workTime_mean = df.groupby('businessEquipmentID').agg({'workTime':'mean'}).reset_index()
        workTime_mean.rename(columns={'workTime':'workTime_mean'}, inplace=True)
        df = pd.merge(df, workTime_stddev, how='left', on='businessEquipmentID')
        df = pd.merge(df, workTime_mean, how='left', on='businessEquipmentID')

        # remove outliers
        df = self.remove_outliers(df)

        # create dummies
        df = self.create_dummies('businessEquipmentID', df, 'businessEquipmentID')

        # drop 5 records with nulls for businessRegionID
        df.dropna(inplace=True)

        return df

if __name__ == '__main__':

    data_cleaner = WorkTimeCleaner()

    # load data
    job_df = data_cleaner.load_data('/Users/mwirvine/Galvanize/dsi-immersive/Capstone-EngageMob-Data/DATA.xlsx')

    # keep only records with a job type in 'job_type' list
    job_types = [161, 239, 226, 235, 249, 250, 255]
    # keep only these columns as features and the target
    cols_to_keep = ['businessEquipmentID', 'volume', 'equipmentName', 'quantity', 'amount', 'workTime', 'businessRegionID', 'createdDate', 'wellSite_loc1', 'wellSite_loc2']

    job_df = data_cleaner.transform(job_df, job_types, cols_to_keep)

    X = job_df.drop(['equipmentName', 'workTime', 'businessRegionID', 'wellSite_loc1', 'wellSite_loc2', 'job_month', 'workTime_std', 'workTime_mean', 'businessEquipmentID'], axis=1).values
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

    # businessRegionID has five nulls - remove?

    # create dummies, create X,y, scale, run models. remove outliers after dummies are created

    # X = df.drop('fraud', axis=1).values
    # y = df['fraud'].values

    # linear regression test



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
- Need workTime starttime and endtime to get more precise? Ask Rob.
- difference between equipmentName and businessEquipmentID. Name has 3 after my filtering (Production Water-Bbl, Slickline, Swabbing). Are we predicting workTime by name or by ID (there are 6 ID values after my filtering)
- What is quantity and amount?
- businessregionid vs wellSite
- two createdDates - difference?
- what data points are available when job is first created? want to ensure volume, amount, etc are there or else there will be data leakage
- if all jobs are completed (after i filter), and all IDs are unique, why do some jobs have timeTypeID 3 and 4 (work and haul)? Checked and all jobsNos and IDs are unique (no dups)

Model Results:

- Features: volume, quantity, amount, businessEquipmentID dummies
- Linear test RMSE: 0.517



'''
