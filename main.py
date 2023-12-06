'''
Assignment 1: Multiple Linear Regression using OOP
We have implemented a multiple linear regression model and compared it with the implementation from sklearn.
Plotting the regression line(s) and plane is also possible.
There is also the possibility to save the model coefficients in either json or csv format.
'''

from multiple_linear_regression import MultipleLinearRegression
from sklearn.linear_model import LinearRegression
from regression_plotter import RegressionPlotter
from compare_with_sklearn import ModelComparer
from model_saver import ModelSaver

def main() -> None:
    model = MultipleLinearRegression(dataset_name='breast_cancer')
    plotter = RegressionPlotter()
    saver = ModelSaver(format_type='json')

    # Use the get_data method to obtain X and y
    X, y = model.get_data()
    
    # Train the model on the generated data
    model.train(X, y)
    
    # Get the intercept
    print(f"Intercept: {model._intercept}")

    # Make predictions using the trained model
    predictions = model.predict(X)

    # Print the ground truth and predicted values
    print("Ground truth:", y)
    print("Predicted value:", predictions)
    
    # Compare our model with sklearn
    sklearn_model = LinearRegression()
    sklearn_model.fit(X, y)
    print("Our model coefficients:", (model._intercept))
    print("Sklearn model coefficients:", [sklearn_model.intercept_, *sklearn_model.coef_])
    our_predictions = model.predict(X)
    sklearn_predictions = sklearn_model.predict(X)
    print("MSE our model:",  mean_squared_error(y, our_predictions))
    print("MSE sklearn model:", mean_squared_error(y, sklearn_predictions))  

    # Plot the regression line
    plotter.set_data(X, y)
    plotter.plot_linear_regression(model, [5], 1)
    
    # Plot the regression plane
    plotter.set_data(X, y)
    plotter.plot_linear_regression_plane(model, [2,3], 1)
    
    # Plot multiple regression lines
    plotter.set_data(X, y)
    plotter.plot_linear_regression(model, [3,4,5,6,7,8], 1)

    # Save the model parameters in json format
    saver.save_model_parameters(model, 'model_parameters.json')

    # Load the model parameters from json file
    saver.load_model_parameters(model, 'model_parameters.json')

if __name__ == "__main__":
    main()
