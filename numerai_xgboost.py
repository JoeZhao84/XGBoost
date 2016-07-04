
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import numpy as np
import csv

csv_file = 'numerai_training_data.csv'
df = pd.read_csv(csv_file)


# We'll impute missing values using the median for numeric columns and the most
# common value for string columns.
# This is based on some nice code by 'sveitser' at http://stackoverflow.com/a/25562948
from sklearn.base import TransformerMixin
class DataFrameImputer(TransformerMixin):
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
            index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.fill)

big_X = df.drop(['target'], axis=1)
big_X_imputed = DataFrameImputer().fit_transform(big_X)

# Prepare the inputs for the model
X = big_X_imputed[0:df.shape[0]].as_matrix()
y = df['target']

from sklearn.cross_validation import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X,y)

# You can experiment with many other options here, using the same .fit() and .predict()
# methods; see http://scikit-learn.org
# This example uses the current build of XGBoost, from https://github.com/dmlc/xgboost
gbm = xgb.XGBClassifier(max_depth=3, n_estimators=200, learning_rate=0.03).fit(train_X, train_y)
predictions = gbm.predict(test_X)

from sklearn.metrics import accuracy_score
accuracy_score = accuracy_score(test_y,predictions)

print(accuracy_score)