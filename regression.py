import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class Regression():
    def __init__(train_file):
        self.create_regr()
        pass
    def get_data():
        """
        Get data and split into train and test.
        More specifically - self.X_train, self.Y_train
        self.X_test, self.Y_test
        """
        pass
    def preprocess_data():
        """
        I guess you need to preprocess both train and test
        """
        pass

    def get_all_features():
        """
        Return list of all the feature names
        """
        pass

    def create_regr(self):
        # self.regr = RandomForestRegressor(max_depth=5, random_state=0)
        self.regr = LinearRegression()

    def get_MSE(self, feature_list):
        regr.fit(self.X_train[feature_list], self.Y_train)
        return mean_squared_error(self.Y_test, 
            self.regr.predict(self.X_test[feature_list]))

