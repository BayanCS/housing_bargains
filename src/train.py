#!/usr/bin/env python
"""
Script used for training a model.
"""

from utils import ZipcodeTransformer

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.externals import joblib
from scipy.stats import expon, uniform

import numpy as np
import pandas as pd
import argparse

def prepare_data(df):
    """
    Removes the price and id features, and add price per square foot of living area.
    Also, converts the datetime column to an equal to the number of days since the year started.
    """
    df = df.assign(price_per_sqft=df.price / df.sqft_living)
    df = df.drop(['id', 'price'], axis=1)
    return df.assign(date=pd.to_datetime(df.date.str.replace("T000000", ""),
                                         format="%Y%m%d").apply(lambda d: d.timetuple().tm_yday))

def get_pipeline(estimator=None):
    """
    Constructs a pipeline with the given estimator as its final step.
    """
    steps = [
        ('zipcode_average', ZipcodeTransformer()),
        ('scaler', MinMaxScaler()),
        ('log', FunctionTransformer(np.log1p)),
    ]

    if estimator is not None:
        steps.append(('estimator', estimator))

    return Pipeline(steps)

def train(df, param_search=False):
    """
    Creates a pipeline with a GradientBoostingRegressor as its estimator,
    and trains it on the given data.
    Optionally, it can perform a randomized parameter search.
    """
    y = df['price_per_sqft']
    X = df.drop('price_per_sqft', axis=1)

    if param_search:
        randomized_search = RandomizedSearchCV(
            get_pipeline(GradientBoostingRegressor(random_state=42)),
            param_distributions={"estimator__learning_rate": expon(scale=.03),
                                "estimator__n_estimators":[100, 500, 700, 800],
                                "estimator__subsample": uniform(0., 1.) },
            random_state=42,
            cv=5)
        print("Performing randomized paramater search. Hang on, this might take a while...")
        randomized_search.fit(X, y)
        print("Model trained with parameters:")
        for param, value in randomized_search.best_params_.iteritems():
            print("{}: {}".format(param.split("__"[-1]), value))
        clf = randomized_search.best_estimator_
    else:
        clf = get_pipeline(GradientBoostingRegressor(random_state=42))
        clf.fit(X, y)

    return clf


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--csv_file",
                        help="path to the csv file storing training data")
    parser.add_argument("-m", "--model_path",
                        help="where to save the trained model")
    parser.add_argument("-p", "--param-search", type=bool,
                        help=("whether to perform a randomized parameter search; slows "
                              "down training but the resulting model might perform better"))
    args = parser.parse_args()

    data = prepare_data(pd.read_csv(args.csv_file))
    clf = train(data, args.param_search)

    joblib.dump(clf, args.model_path)
