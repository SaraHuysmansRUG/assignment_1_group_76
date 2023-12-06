'''
This file contains a class that can be used to plot the results of linear regression.
'''

from multiple_linear_regression import MultipleLinearRegression as MultipleLinearRegression
import matplotlib.pyplot as plt
from typing import List
import numpy as np

class RegressionPlotter:
    def __init__(self):
        self._data = None
        
    def set_data(self, X:np.ndarray, y:np.ndarray) -> None:
        '''
        X is a numpy array with a shape (n, p) containing the input data (n units, p features)
        y is a numpy array with a 1d shape containing the output data
        '''
        self._X = X
        self._y = y
        self._data = np.column_stack((X, y))

    def plot_linear_regression_line(self, model:MultipleLinearRegression, feature:int, target:int) -> None:
        '''
        model is an instance of LinearRegression
        feature is an integer representing the feature index
        target is an integer representing the target index
        '''
        # Train the model on the data
        model.train(self._data[:, feature], self._data[:, target])

        # Plot the data and regression line
        plt.scatter(self._data[:, feature], self._data[:, target], label='Data')
        plt.plot(self._data[:, feature], model.predict(self._data[:, feature]), color='red', label='Regression Line')
        plt.xlabel(f'Feature {feature}')
        plt.ylabel(f'Target')
        plt.legend()
        plt.show()

    def plot_linear_regression_plane(self, model:MultipleLinearRegression, features:List[int], target:int) -> None:
        '''
        model is an instance of LinearRegression
        features is a list containing two feature indices
        target is an integer representing the target index
        '''
        # select the features
        feature1 = features[0]
        feature2 = features[1]
        
        # Train the model on the selected features
        model.train(self._data[:, features], self._data[:, target])
        
        # Predict the target values for the meshgrid features
        X, y = np.meshgrid(self._data[:, feature1], self._data[:, feature2])
        predictions = model.predict(np.column_stack((X.ravel(), y.ravel()))).reshape(X.shape)
        
        # Create a 3D figure
        fig = plt.figure(figsize=(10, 10))
        axes = fig.add_subplot(111, projection='3d')

        # Plot the linear regression plane
        axes.plot_surface(X, y, predictions, alpha=0.1, color='red')
        axes.scatter(self._data[:, feature1], self._data[:, feature2], self._data[:, target], alpha=1, color='blue')
        axes.set_xlabel(f'Feature {feature1}')
        axes.set_ylabel(f'Feature {feature2}')
        axes.set_zlabel(f'Target')

        plt.show()

    def plot_multiple_linear_regression(self, model:MultipleLinearRegression, features:List[int], target:int) -> None:
        '''
        model is an instance of LinearRegression
        feature is an integer representing the feature index
        target is an integer representing the target index
        '''
        # Create a figure with subplots
        fig, axes = plt.subplots(len(features), 1, figsize=(6, 6*len(features)))
        
        for i, feature in enumerate(features):
            # Train the model on the selected feature
            model.train(self._data[:, feature], self._data[:, target])
            predictions = model.predict(self._data[:, feature])
            
            # Plot the data and regression line in each subplot
            axes[i].scatter(self._data[:, feature], self._data[:, target], label='Data')
            axes[i].plot(self._data[:, feature], predictions, color='red', label='Regression Line')
            axes[i].set_xlabel(f'Feature {feature}')
            axes[i].set_ylabel(f'Target')
            axes[i].legend()
        
        plt.tight_layout(pad=5.0)
        plt.show()

    def plot_linear_regression(self, model:MultipleLinearRegression, features:List[int], target:int) -> None:
        '''
        model is an instance of LinearRegression
        features is a list containing two feature indices
        target is an integer representing the target index
        '''
        if self._data is None:
            raise ValueError("No data provided for plotting. Set the 'data' attribute first.")

        # Plot the data and regression line
        if len(features) == 1:
            self.plot_linear_regression_line(model, features, target)
        elif len(features) == 2:
            self.plot_linear_regression_plane(model, features, target)
        else:
            self.plot_multiple_linear_regression(model, features, target)
