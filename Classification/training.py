# K-Nearest Neighbours (KNN)
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib


class Training(object):
    def __init__(self):
        self.X = None
        self.Y = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def load_dataset(self):
        random.seed(42)
        dataset = pd.read_csv('./Classification/dataset.csv')  # -*- coding: utf-8 -*-
        self.X = dataset[["Group"]]
        self.Y = dataset["PLACEHOLDER"]

    def encoding_variables_train(self):
        # Encoding our categorical variables for Y
        label_encoding_y = LabelEncoder()
        self.Y = label_encoding_y.fit_transform(self.Y)

        # Encoding our categorical variables for x
        label_encoding_x = LabelEncoder()
        self.X = label_encoding_x.fit_transform(self.X)

        # Reshaping X as data has a single feature
        self.X = np.array(self.X)
        self.X = self.X.reshape(-1, 1)

        # encoding, so we will use 'OneHotEncoding', to give dummy data
        one_hot_encoder = OneHotEncoder()
        self.X = one_hot_encoder.fit_transform(self.X).toarray()

        # Avoiding the dummy variable trap
        self.X = self.X[:, 1:]

    def encoding_variables_test(self, x_to_encode):
        # Encoding our categorical variables for Y
        label_encoding_y = LabelEncoder()
        self.Y = label_encoding_y.fit_transform(self.Y)

        # Encoding our categorical variables for x
        label_encoding_x = LabelEncoder()
        self.X = label_encoding_x.fit_transform(self.X)

        # Reshaping X as data has a single feature
        self.X = np.array(self.X)
        self.X = self.X.reshape(-1, 1)

        #  One hot encoding
        one_hot_encoder = OneHotEncoder()
        self.X = one_hot_encoder.fit_transform(self.X).toarray()

        # Avoiding the dummy variable trap
        self.X = self.X[:, 1:]

        # Encoding training
        encoded_x = label_encoding_x.transform(x_to_encode)

        # Reshaping X as data has a single feature
        encoded_x = np.array(encoded_x)
        encoded_x = encoded_x.reshape(-1, 1)

        # encoding, so we will use 'OneHotEncoding', to give dummy data
        encoded_x = one_hot_encoder.transform(encoded_x).toarray()

        # Avoiding the dummy variable trap
        encoded_x = encoded_x[:, 1:]
        return encoded_x

    def decoding_y(self, predicted_y):
        # decoding our categorical variables for Y
        label_encoding_y = LabelEncoder()
        self.Y = label_encoding_y.fit_transform(self.Y)
        return label_encoding_y.inverse_transform(predicted_y)[0]

    def split_to_train_and_test(self):
        # splitting the dataset into a training set and a test set
        # test_size=0.20 which is setting 20% of our data for test set
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.Y,
                                                                                test_size=0.20,
                                                                                random_state=42)

    def feature_scaling_train(self):
        # feature scaling
        sc_x = StandardScaler()
        scaler_x = sc_x.fit(self.x_train)
        self.x_train = scaler_x.transform(self.x_train)
        return self.x_train

    def feature_scaling_test(self, x_to_scale):
        # feature scaling
        sc_x = StandardScaler()
        scaler_x = sc_x.fit(self.x_train)
        x_train_scaled = scaler_x.transform(x_to_scale)
        return x_train_scaled

    def train(self, scaled_x_train):
        classifier = KNeighborsClassifier(n_neighbors=1, algorithm='auto',
                                          metric='minkowski', p=2)
        classifier.fit(scaled_x_train, self.y_train)
        joblib.dump(classifier, './saved_model/food_prediction')
        print('Done training')


if __name__ == '__main__':
    prediction = Training()
    prediction.load_dataset()
    prediction.encoding_variables_train()
    prediction.split_to_train_and_test()
    scaled_x_train_output = prediction.feature_scaling_train()
    prediction.train(scaled_x_train_output)
