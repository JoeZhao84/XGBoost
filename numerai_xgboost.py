
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
from sklearn import cross_validation, grid_search
from sklearn.metrics import roc_auc_score
from xgboost.sklearn import XGBClassifier

xgb1 = xgb.XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27).fit(train_X, train_y)
scores = cross_validation.cross_val_score(xgb1, X, y, cv=5)
pred = xgb1.predict(test_X)
auc = roc_auc_score(test_y, pred)

print("Accuracy: %0.7f (+/- %0.7f)" % (scores.mean(), scores.std() * 2))
print("ROC%0.4f " % auc)

# tuning
param_test1 = {
 'max_depth':[3,5,7,9],
 'min_child_weight':[1,3,5]
}
gsearch1 = grid_search.GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27),
 param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1.fit(train_X, train_y)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


# You can experiment with many other options here, using the same .fit() and .predict()
# methods; see http://scikit-learn.org
# This example uses the current build of XGBoost, from https://github.com/dmlc/xgboost
# gbm = xgb.XGBClassifier(max_depth=3, n_estimators=200, learning_rate=0.03).fit(train_X, train_y)
# predictions = gbm.predict(test_X)
#
# from sklearn.metrics import accuracy_score
# accuracy_score = accuracy_score(test_y,predictions)
#
# print(accuracy_score)