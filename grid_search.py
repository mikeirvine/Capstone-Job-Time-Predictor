from sklearn.model_selection import RandomizedSearchCV

'''Code to implement a RandomizedSearchCV for Gradient Boosting and Random Forest'''

'''RANDOM FOREST'''
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
random_grid_rf = {'n_estimators': n_estimators_rf,
               'max_features': max_features_rf,
               'max_depth': max_depth_rf,
               'min_samples_split': min_samples_split_rf,
               'min_samples_leaf': min_samples_leaf_rf,
               'bootstrap': bootstrap_rf}

rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid_rf, n_iter = 1000, cv = 3, verbose=2, random_state=42, n_jobs = -1)
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

'''GRADIENT BOOSTING'''
# Number of trees in gradient boosting
n_estimators_gb = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Learning rate
learning_rate_gb = [.0001, .001, .005, .01, .02, .03, .04, .05, .075, .1, .15, .2, .25, .3, .4, .5]
# Number of features to consider at every split
max_features_gb = ['auto', 'sqrt', None]
# Maximum number of levels in tree
max_depth_gb = [1,2,3,4,5,6,7,8,9,10,15,20]
max_depth_gb.append(None)
# Minimum number of samples required to split a node
min_samples_split_gb = [2, 5, 10, 15]
# Minimum number of samples required at each leaf node
min_samples_leaf_gb = [1, 2, 4, 8]

# Create the random grid
random_grid_gb = {'n_estimators': n_estimators_gb,
                'learning_rate': learning_rate_gb,
               'max_features': max_features_gb,
               'max_depth': max_depth_gb,
               'min_samples_split': min_samples_split_gb,
               'min_samples_leaf': min_samples_leaf_gb}

gb = GradientBoostingRegressor()
gb_random = RandomizedSearchCV(estimator = gb, param_distributions = random_grid_gb, n_iter = 1000, cv = 3, verbose=2, random_state=42, n_jobs = -1)
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
