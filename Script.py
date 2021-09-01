    
import time
start_time = time.time()

import pandas as pd

# For Ordinal Encoding categorical variables and splitting data.
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

# For training model
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor

# =============================================================================
# import lazypredict
# from lazypredict.Supervised import LazyRegressor
# =============================================================================
print("Load Succesfull")


# Loading the Data
# Set index_col=0 in the code cell below to use the id column to index the DataFrame

# Load the training data - NOTE: Be sure to change data file paths to your own directory.
train = pd.read_csv("C:/Users/Number Killer/Desktop/script/train.csv", index_col=0)
test = pd.read_csv("C:/Users/Number Killer/Desktop/script/test.csv", index_col=0)
# Preview the data, if needed.
# print(train.head())
# print(train.describe())


# Seperate the target (`y`) from the training features (`features`).

# Separate target from features
y = train['target']
features = train.drop(['target'], axis=1)

# List of features for later use
feature_list = list(features.columns)

# Preview features
# features.head()



# List of categorical columns
object_cols = [col for col in features.columns if 'cat' in col]

# ordinal-encode categorical columns
X = features.copy()
X_test = test.copy()

# Ordinal Encode input variables
ordinal_encoder = OrdinalEncoder()
X[object_cols] = ordinal_encoder.fit_transform(features[object_cols])
X_test[object_cols] = ordinal_encoder.transform(test[object_cols])

# Preview the ordinal-encoded features
# print(X.head())


# Split validation set from the training data.
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=1)

# These are the parameters for the LightGBM model. Taken directly from a public notebook on Kaggle and slightly modified by me: https://www.kaggle.com/michaelisaac15/repeat-lgbm-with-fe
params = {
    'metric': 'rmse',       
    'n_estimators': 10000,    
    'reg_alpha': 15,
    'reg_lambda': 17.396730654687218,
    'learning_rate': 0.09985133666265425,
    'max_depth': 5,
    'num_leaves': 3,
    'min_child_samples': 10,    
    'max_bin': 523,
    'n_jobs': 4,
    'colsample_bytree': 0.11807135201147481,    
}
   
model = LGBMRegressor(**params) # The double-asterisk denotes dictionary unpacking.

# Train the model
model.fit(X_train, y_train)
preds_valid = model.predict(X_valid)
print("RMSE: %s" % mean_squared_error(y_valid, preds_valid, squared=False))


# =============================================================================
# # This code block was first used to determine the best model to use. BE CAREFUL. It may run for hours.
# # Define the model 
# model = LazyRegressor(verbose=1, ignore_warnings=False, custom_metric=None)
# 
# # Train the model (will take about 10 minutes to run)
# models, predictions = model.fit(X_train, X_valid, y_train, y_valid)
# 
# print(models)
# =============================================================================

# Make predictions on entire data set.
predictions = model.predict(X_test)

# # Save the predictions to a CSV file
# output = pd.DataFrame({'Id': X_test.index,
#                         'target': predictions})
# output.to_csv('submission.csv', index=False)

print("Success")
print("--- %s seconds ---" % (time.time() - start_time))