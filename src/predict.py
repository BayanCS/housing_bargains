#!/usr/bin/env python
"""
Script usd for obtaining predictions from a model.
"""
from utils import ZipcodeTransformer

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

import numpy as np
import pandas as pd
import argparse

def prepare_data(df):
    """
    Drops the id from the dataframe.
    Also, converts the datetime column to an equal to the number of days since the year started.
    """
    df = df.drop(['id'], axis=1)
    return df.assign(date=pd.to_datetime(df.date.str.replace("T000000", ""),
                                         format="%Y%m%d").apply(lambda d: d.timetuple().tm_yday))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--csv_file",
                        help="path to the csv file storing data to predict from")
    parser.add_argument("-n", "--index", type=int,
                        help="the row for which to predict. If not present, all rows will be predicted")
    parser.add_argument("-m", "--model_path",
                        help="location of the trained model")
    args = parser.parse_args()

    index = args.index
    data = prepare_data(pd.read_csv(args.csv_file))

    if "price" in data.columns:
        data = data.drop("price", axis=1)

    if index is not None:
        data = data.loc[[index]]

    clf = joblib.load(args.model_path)

    predictions = clf.predict(data)

    size = len(predictions)
    print("Obtained {} prediction{}:".format(size, "s" if size > 1 else ""))
    for sqft, price_per_sqft in zip(data.sqft_living, predictions):
        print(" - Price per square foot: {} Total price: {}"
              .format(price_per_sqft, price_per_sqft * sqft))
