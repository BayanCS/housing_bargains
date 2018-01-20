from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class ZipcodeTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer for converting the zipcode column to the mean house price at that zipcode.
    """

    def fit(self, X, y):
        self.average_prices = y.groupby(X.zipcode).agg('mean')
        return self

    def transform(self, X):
        X = X.assign(zipcode_average=X['zipcode'].map(self.average_prices))
        # In case there is a zipcode that was not seen during fitting we
        # assign it the mean of all means
        X.loc[X.zipcode_average==np.NaN, 'zipcode_average'] = self.average_prices.mean()
        return X
