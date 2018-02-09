import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

class Regression():
    def __init__(self, train_file):
        self.get_data(train_file)
        self.create_regr()
        self.create_log_regr()

    def get_MSE(self, feature_list):
        self.regr.fit(self.X_train[feature_list], self.Y_train)
        return mean_squared_error(self.Y_test, 
            self.regr.predict(self.X_test[feature_list]))

    def get_log_MSE(self, feature_list):
        self.log_regr.fit(self.X_train[feature_list], self.Y_train)
        return mean_squared_error(self.Y_test, 
            self.log_regr.predict(self.X_test[feature_list]))

    def get_all_features(self):
        """
        Return list of all the feature names
        """
        return list(self.X_train.columns)

    def preprocess_data(self):

        data = self.dataset.dropna(axis=0, how='any')

        train, test = train_test_split(data, test_size=0.33, shuffle=False)

        self.X_train = train[train.columns[:-1]]
        self.Y_train = train[train.columns[-1]].to_frame()
        self.X_test = test[test.columns[:-1]]
        self.Y_test = test[test.columns[-1]].to_frame()

        scaling_cols = list(self.X_train.select_dtypes(include=['float64', 'int64']).columns)
        encoding_cols = list(self.X_train.select_dtypes(include=['object']).columns)

        self.X_train = self.min_max_scaling(self.X_train, scaling_cols)
        self.X_test = self.min_max_scaling(self.X_test, scaling_cols)

        # # # self.X_train = self.standard_normal_scaling(self.X_train, scaling_cols)
        # # # self.X_test = self.standard_normal_scaling(self.X_test, scaling_cols)

        self.X_train, self.X_test = self.label_encoding(self.X_train, self.X_test)

        Y_train_col_name = self.Y_train.columns[0]

        if self.Y_train[Y_train_col_name].dtypes == 'object':
            self.Y_train, self.Y_test = self.label_encoding(self.Y_train, self.Y_test)
        else:
            self.Y_train = self.min_max_scaling(self.Y_train, self.Y_train.columns.values)
            self.Y_test = self.min_max_scaling(self.Y_test, self.Y_test.columns.values)
            

    ## Private Functions

    def get_data(self, train_file):
        """
        Get data
        """

        self.dataset = pd.read_csv(train_file)

        # #Hardcoded files

        # Importing training data set
        # self.X_train=pd.read_csv('X_train.csv')
        # self.Y_train=pd.read_csv('Y_train.csv')

        # # Importing testing data set
        # self.X_test=pd.read_csv('X_test.csv')
        # self.Y_test=pd.read_csv('Y_test.csv')

    def min_max_scaling(self, data, cols):
        #Scaling continuous values to between 0 and 1
        from sklearn.preprocessing import MinMaxScaler
        min_max=MinMaxScaler()
        
        for col in cols:
            data[col] = min_max.fit_transform(data[col].reshape(-1, 1))

        return data

    def standard_normal_scaling(self, data, cols):
        #Scaling continuous values to standard normal distribution
        from sklearn.preprocessing import scale
        data[cols] = scale(data[cols])

        return data

    def label_encoding(self, train_data, test_data):
        #Converting categorical variables to Numeric values
        from sklearn.preprocessing import LabelEncoder
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

    def one_hot_encoding(self):
        ###Doesn't work for now

        from sklearn.preprocessing import OneHotEncoder
        enc=OneHotEncoder(sparse=False)
        columns=encoding_cols

        for col in columns:
            # creating an exhaustive list of all possible categorical values
            data=self.X_train[[col]].append(self.X_test[[col]])
            enc.fit(data)
            # Fitting One Hot Encoding on train data
            temp = enc.transform(self.X_train[[col]])
            # Changing the encoded features into a data frame with new column names
            temp=pd.DataFrame(temp,columns=[(col+"_"+str(i)) for i in data[col]
                .value_counts().index])
            # In side by side concatenation index values should be same
            # Setting the index values similar to the X_train data frame
            temp=temp.set_index(self.X_train.index.values)
            # adding the new One Hot Encoded varibales to the train data frame
            self.X_train=pd.concat([self.X_train,temp],axis=1)
            # fitting One Hot Encoding on test data
            temp = enc.transform(self.X_test[[col]])
            # changing it into data frame and adding column names
            temp=pd.DataFrame(temp,columns=[(col+"_"+str(i)) for i in data[col]
                .value_counts().index])
            # Setting the index for proper concatenation
            temp=temp.set_index(self.X_test.index.values)
            # adding the new One Hot Encoded varibales to test data frame
            self.X_test=pd.concat([self.X_test,temp],axis=1)

    def create_regr(self):
        # self.regr = RandomForestRegressor(max_depth=5, random_state=0)
        self.regr = LinearRegression()

    def create_log_regr(self):
        self.log_regr = LogisticRegression()