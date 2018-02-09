import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, scale, LabelEncoder

class Supervisor():
    """
    Helper class for loading, preprocessing and executing the regression
    models.
    """
    def __init__(self, train_file, is_reg_flag=False):
        """
        Initializes the object.

        Args:
        train_file  :  CSV file with training data.
        is_reg_flag :  Flag indicating the problem is linear regression.
                       Otherwise, the problem is logistic regression. 
        """

        self.get_data(train_file)
        self.reg_flag = is_reg_flag
        if is_reg_flag:
            self.create_regr()
        else:
            self.create_log_regr()

    def get_obj_fn(self, feature_list):
        """
        Returns -ve of MSE or Accuracy based on is_reg_flag

        Args:
        feature_list: List of features

        Returns -ve of Mean Squared Error for the given feature list.
        This allows us to maximize this objective function like accuracy.
        (Or)
        Returns accuracy.
        """
        
        if self.reg_flag:
            return (-1)*self.get_MSE(feature_list)
        else:
            return self.get_accuracy(feature_list)

    def get_all_features(self):
        """
        Returns list of all the features.
        """
        return list(self.X_train.columns)

    def preprocess_data(self):
        """
        Processes data before use.
        """
        # Handle NaN values by replacing them with mode
        data = self.dataset.fillna(self.dataset.mode().iloc[0])

        # Split data into train and test
        train, test = train_test_split(data, test_size=0.33, shuffle=False)

        # X -> features
        # Y -> Class / Actual value
        self.X_train = train[train.columns[:-1]]
        self.Y_train = train[train.columns[-1]].to_frame()
        self.X_test = test[test.columns[:-1]]
        self.Y_test = test[test.columns[-1]].to_frame()

        # scaling_cols -> columns with dataType float64 or int64
        # encoding_cols-> columns with dataType object
        scaling_cols = list(self.X_train.select_dtypes(include=['float64', 'int64']).columns)
        encoding_cols = list(self.X_train.select_dtypes(include=['object']).columns)

        # Scales values between 0 and 1. 
        self.X_train = self.min_max_scaling(self.X_train, scaling_cols)
        self.X_test = self.min_max_scaling(self.X_test, scaling_cols)

        # Scales values to standard normal distribution.
        # # # self.X_train = self.standard_normal_scaling(self.X_train, scaling_cols)
        # # # self.X_test = self.standard_normal_scaling(self.X_test, scaling_cols)

        # Converts categorical variables to numericals.
        self.X_train, self.X_test = self.label_encoding(self.X_train, self.X_test)

        # Converts Y-values to numericals.
        Y_train_col_name = self.Y_train.columns[0]

        if self.Y_train[Y_train_col_name].dtypes == 'object':
            self.Y_train, self.Y_test = self.label_encoding(self.Y_train, self.Y_test)
        else:
            self.Y_train = self.min_max_scaling(self.Y_train, self.Y_train.columns.values)
            self.Y_test = self.min_max_scaling(self.Y_test, self.Y_test.columns.values)
            

    ## Private Functions

    def get_data(self, train_file):
        """
        Get data from file.

        Args:
        train_file: File to load data from.
        """

        self.dataset = pd.read_csv(train_file)

        # #Hardcoded files

        # Importing training data set
        # self.X_train=pd.read_csv('X_train.csv')
        # self.Y_train=pd.read_csv('Y_train.csv')

        # # Importing testing data set
        # self.X_test=pd.read_csv('X_test.csv')
        # self.Y_test=pd.read_csv('Y_test.csv')


    def get_MSE(self, feature_list):
        """
        Returns Mean Squared Error for the given feature list.

        Args:
        feature_list:  List of features to run model.

        Returns MSE for the features list.
        """

        self.model.fit(self.X_train[feature_list], self.Y_train)
        return mean_squared_error(self.Y_test, 
            self.model.predict(self.X_test[feature_list]))

    def get_accuracy(self, feature_list):
        """
        Returns Accuracy for the given feature list.

        Args:
        feature_list:  List of features to run model.

        Returns Accuracy for the features list.
        """

        if len(feature_list) == 0:
            return 0
        
        self.model.fit(self.X_train[feature_list], self.Y_train)
        return accuracy_score(self.Y_test, 
            self.model.predict(self.X_test[feature_list]))

    def min_max_scaling(self, data, cols):
        """
        Scales continuous values to between 0 and 1     

        Args:
        data :  Dataframe to work on.
        cols :  Columns to scale.

        Returns modified dataframe.   
        """

        min_max=MinMaxScaler()
        
        for col in cols:
            data[col] = min_max.fit_transform(data[col].reshape(-1, 1))

        return data

    def standard_normal_scaling(self, data, cols):
        """
        Scales continuous values to standard normal distribution.     

        Args:
        data :  Dataframe to work on.
        cols :  Columns to scale.

        Returns modified dataframe.   
        """

        data[cols] = scale(data[cols])

        return data

    def label_encoding(self, train_data, test_data):
        """
        Converts categorical variables to numeric values

        Args:
        data :  Dataframe to work on.
        cols :  Columns to scale.

        Returns modified dataframe.   
        """

        le=LabelEncoder()
        for col in train_data.columns.values:
            # Encoding only categorical variables
            if train_data[col].dtypes=='object':
                # Using whole data to form an exhaustive list of levels
                data = train_data[col].append(test_data[col])
                le.fit(data.values)
                train_data[col]=le.transform(train_data[col])
                test_data[col]=le.transform(test_data[col])

        return train_data, test_data

    def create_regr(self):
        """
        Creates model of Linear Regression.
        """

        # self.regr = RandomForestRegressor(max_depth=5, random_state=0)
        self.model = LinearRegression()

    def create_log_regr(self):
        """
        Creates model of Logistic Regression.
        """
        self.model = LogisticRegression()

