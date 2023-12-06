'''
This file contains a class that compares our implementation of multiple linear regression with the implementation from sklearn.
This was asked for in the assignment in question 1.
'''

from multiple_linear_regression import MultipleLinearRegression as MultipleLinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

class ModelComparer:
    def __init__(self, our_model:MultipleLinearRegression, sklearn_model:object):
        self._our_model = our_model
        self._sklearn_model = sklearn_model
    
    def compare_coefficients(self) -> None:
        print("Our model coefficients:", (self._our_model._intercept))
        print("Sklearn model coefficients:", [self._sklearn_model.intercept_, *self._sklearn_model.coef_])
    
    def compare_predictions(self, X:np.ndarray, y:np.ndarray) -> None:
        '''
        X is a numpy array with a shape (n, p) containing the input data (n units, p features)
        y is a numpy array with a 1d shape containing the output data
        '''
        # Make and print predictions using the trained models
        our_predictions = self._our_model.predict(X)
        sklearn_predictions = self._sklearn_model.predict(X)
        print("MSE our model:",  mean_squared_error(y, our_predictions))
        print("MSE sklearn model:", mean_squared_error(y, sklearn_predictions))
    
    def compare(self, X:np.ndarray, y:np.ndarray) -> None:
        '''
        X is a numpy array with a shape (n, p) containing the input data (n units, p features)
        y is a numpy array with a 1d shape containing the output data
        '''
        # Compare the coefficients and predictions
        self.compare_coefficients()
        self.compare_predictions(X, y)
