'''
This file contains a class that implements multiple linear regression.
'''

from sklearn.datasets import load_breast_cancer
from typing import Tuple
import numpy as np

class MultipleLinearRegression:
    def __init__(self, dataset_name:str='breast_cancer_data'):
        self._intercept = None
        self._dataset_name = dataset_name

    def train(self, X:np.ndarray, y:np.ndarray) -> None:
        '''
        X is a numpy array with a shape (n, p) containing the input data (n units, p features)
        y is a numpy array with a 1d shape containing the output data
        '''
        # Add a column of ones for the intercept term 
        X_intercept = np.column_stack((np.ones(len(X)), X))
        
        # Calculate coefficients using equation (10) from the assignment
        try: # Try to invert the matrix
            self._intercept = np.linalg.inv(X_intercept.T @ X_intercept) @ X_intercept.T @ y
        except np.linalg.LinAlgError as e: # If the matrix is not invertible, raise an error
            raise ValueError("The input matrix is not invertible. Try adding a small positive constant to the diagonal.") from e
        
    def predict(self, X:np.ndarray) -> np.ndarray:
        '''
        X is a numpy array with a shape (n, p) containing the input data (n units, p features)
        Returns a numpy array containing the predicted values
        '''
        # Add a column of ones for the intercept term
        X_intercept = np.column_stack((np.ones(len(X)), X))
        
        # Calculate predictions using the learned coefficients
        predictions = X_intercept @ self._intercept
        return predictions
    
    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Returns the input data X and the output data y.
        '''
        # Load the breast cancer dataset
        if self._dataset_name == 'breast_cancer':
            breast_cancer_data = load_breast_cancer()
        else: # Raise an error if the dataset name is invalid
            raise ValueError("Invalid dataset name. Try 'breast_cancer'.")
        X = breast_cancer_data.data
        y = breast_cancer_data.target
        return X, y
