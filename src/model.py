import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV

def load_raw_data(filepath):
    pd.read_csv(filepath)

def log_transform(X, y=None):
    return X.apply(lambda x: np.log(x + 1))

def day_of_year(d):
    return d.timetuple().tm_yday

from sklearn.base import BaseEstimator, TransformerMixin

def prepare_data(df):
    df = df.assign(price_per_sqft=df.price / df.sqft_living)
    # drop lat and long (?)
    df = df.drop(['id', 'price'], axis=1)
    return df.assign(date=pd.to_datetime(df.date.str.replace("T000000", ""), format="%Y%m%d").apply(day_of_year))

class ZipcodeTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y):
        self.averages = y.groupby(X.zipcode).agg({'price_per_sqft': 'mean'})
        return self

    def transform(self, X):
        X = X.assign(zipcode_average=X['zipcode']
                     .map(self.averages.to_dict()['price_per_sqft']))
        # In case there is a zipcode that was not seen during fitting we
        # assign it the mean of all means
        X[X.zipcode_average==np.NaN]['zipcode_average'] = self.averages.mean()
        return X

def get_pipeline(estimator=None):
    steps = [
        ('zipcode_average', ZipcodeTransformer()),
        ('scaler', MinMaxScaler()),
        ('log', FunctionTransformer(np.log1p)),
    ]

    if estimator is not None:
        steps.append(('estimator', estimator))

    return Pipeline(steps)

class MultiGridSearch(BaseEstimator):

    def __init__(self,
                 estimators,
                 param_grids=None,
                 param_distributions=None,
                 scoring=None,
                 n_jobs=1,
                 cv=None,
                 randomized=False):
        self._estimators = estimators
        self._param_grids = param_grids
        self._scoring = scoring
        self._n_jobs = n_jobs
        self._cv = cv
        self._randomized = randomized

    def fit(self, X, y):
        if self._randomized:
            search_class = RandomizedSearchCV
        else:
            search_class = GridSearchCV

        self.best_params_ = dict()
        self.best_score_ = dict()

        for n in range(len(self._estimators)):
            clf = self._estimators[n]

            if self._randomized:
                search = RandomizedSearchCV(clf, self._param_grids[n], cv=self._cv, n_jobs=self._n_jobs)
            else:
                search = GridSearchCV(clf, self._param_grids[n], cv=self._cv, n_jobs=self._n_jobs)

            search.fit(X, y)
            self.best_params_[clf] = search.best_params_
            self.best_score_[clf] = search.best_score_


def multi_grid_search(grid_search_class, estimators, param_grids):
    for estimator in estimators:
        grid_search_class()

def compare_estimators(estimators, X_train, y_train, X_test, y_test):
    return map(lambda clf: clf.fit(X_train, y_train).score(X_test, y_test), estimators)
