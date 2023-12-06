# OOP - 2023/24 - Assignment 1

This is the base repository for assignment 1.
Please follow the instructions given in the [PDF](https://brightspace.rug.nl/content/enforced/243046-WBAI045-05.2023-2024.1/2023_24_OOP.pdf) for the content of the exercise.

## How to carry out your assignment

1. Clone this template into a private repository.
2. Please add your partner and `oop-otoz` to the collaborators.
3. Create a new branch called `submission`.
4. Create your code in the `main` branch.
5. Once you are done with the assignment (or earlier), create a pull request from the `main` branch to your `submission` branch and add `oop-otoz` to the reviewers.

The assignment is divided into 4 blocks.
Block 1, 2, and 3 all define different classes.

Put the three classes in three separate files in the `src` folder, with the names specified in the PDF.
**Leave the __init__.py file untouched**.

Put the **main.py** script **outside** of the `src` folder, in the root of this repo.

Below this line, you can write your report to motivate your design choices.

## Submission

The code should be submitted on GitHub by opening a Pull Request from the branch you were working on to the `submission` branch.

There are automated checks that verify that your submission is correct:

1. Deadline - checks that the last commit in a PR was made before the deadline
2. Reproducibility - downloads libraries included in `requirements.txt` and runs `python3 main.py`. If your code does not throw any errors, it will be marked as reproducible.
3. Style - runs `flake8` on your code to ensure adherence to style guides.

---

## Your report

## Object Oriented Programming - Assignment 1

#### Nynke Terpstra (s4574532) & Sara Huysmans (s4666089)

#### Group 76

## Introduction

The purpose of this report is to detail the design choices that we made while implementing the task in our
code as per the specifications outlined in the assignment. The implementation adheres to object-oriented
programming (OOP) principles.

## 1 multiple linear regression.py

This file defines a Python class MultipleLinearRegression that implements multiple linear regression. The
class includes methods for training the model using the normal equation, making predictions, and retrieving
a dataset (defaulting to the Wine dataset). The normal equation is used to calculate the coefficients for the
linear regression model.

### Class Initialization

class MultipleLinearRegression:
def __init__(self, dataset_name:str=’breast_cancer_data’):
self._intercept = None
self._dataset_name = dataset_name

Motivation

- Dataset Name Parameter: Introduced a dataset name parameter to allow flexibility in choosing
    the dataset. The default value is set to ’breast_cancer_data’.
- Intercept Attribute: Made the intercept attribute private to encapsulate the internal state of the
    class.

### Train Method

def train(self, X:np.ndarray, y:np.ndarray) -> None:

Motivation

- Input Parameters: The method takes input data X and output data y, both as NumPy arrays. This
    design choice allows users to provide datasets in a familiar tabular/matrix form.
- Exception Handling: Implemented a try-except block to catch LinAlg Error when attempting to
    invert the matrix. This ensures robustness by handling cases where the matrix is not invertible.

def predict(self, X:np.ndarray) -> np.ndarray:

Motivation

- Input Parameter:Similar to the train method, the predict method takes input data X as a NumPy
    array.
- Predictions Calculation: Predictions are calculated using the learned coefficients and the input
    data. The method adds a column of ones for the intercept term before making predictions.

### Get Data Method

ddef get_data(self) -> Tuple[np.ndarray, np.ndarray]:

Motivation

- Dataset Retrieval: Introduced a get data method to obtain input data X and output data y using
    the specified dataset name. This allows users to easily retrieve the data for training and testing.

## 2 regression plotter.py

This files defines a class RegressionPlotter for visualizing the results of linear regression. The class provides
methods to plot the linear regression line, plane, or multiple lines based on the number of features in the
dataset.

### Class Initialization

class RegressionPlotter:
def __init__(self):
self._data = None

Motivation

- Data Attribute: Introduced a data attribute to store the input data for plotting. This attribute is
    set using the setdata method before performing any plotting operations.

### Set Data Method

def set_data(self, X:np.ndarray, y:np.ndarray) -> None:

Motivation

- Data Setup: The setdata method allows users to set the input data for plotting. It takes input
    data X and output data y and stores them as a NumPy multidimensional array in the data attribute.

### Plot Linear Regression Line Method

def plot_linear_regression_line(self, model:MultipleLinearRegression, feature:int, target:int) -> None:

Motivation

- Single Feature Plotting: This method plots a linear regression line when asked to plot a model with
    only one feature. It visualizes the relationship between the selected feature and the target variable.


### Plot Linear Regression Plane Method

def plot_linear_regression_plane(self, model:MultipleLinearRegression, features:List[int], target:int) -> None:

Motivation

- Two Feature Plotting: This method plots a linear regression plane when asked to plot a model with
    two features. It visualizes the relationship between the selected features and the target variable in
    three dimensions. The scattering of the features have been specifically set to the color blue to enhance
    visibility.

### Plot Multiple Linear Regression Method

def plot_multiple_linear_regression(self, model:MultipleLinearRegression, features:List[int], target:int) -> None:

Motivation

- Multiple Feature Plotting: This method produces a sequence of 2D plots when asked to plot a
    generic MultipleLinearRegression model with any number of features. Each plot presents one feature
    against the target variable.

### Plot Linear Regression Method

def plot_linear_regression(self, model:MultipleLinearRegression, features:List[int], target:int) -> None:

Motivation

- Behavior Handling: This method determines the appropriate behavior for plotting based on the
    number of features provided. It calls the specific plotting methods accordingly.

### Exception Handling

if self.data is None:
raise ValueError("No data provided for plotting. Set the ’data’ attribute first.")

Motivation

- Data Check: Added a check to ensure that data is set before attempting to plot. Raises a ValueError
    if no data is provided, guiding users to set the ’data’ attribute first.

## 3 model saver.py

This file defines a class ModelSaver that allows saving and loading model parameters in either JSON or CSV
format. It is particularly useful when you want to persist the state of a machine learning model, making it
easier to resume training or deploy the model in different environments.


### Class Initialization

class ModelSaver:
def __init__(self, format_type:str):
self._format_type = format_type

Motivation

- Format Type Parameter:Introduced a format type parameter to specify the desired format for
    saving and loading. This parameter is set at initialization and can be modified by the user.

### Save Model Parameters Method

def save_model_parameters(self, model:any, file_path:str) -> None:

Motivation

- JSON and CSV Support: Implemented support for saving model parameters in both JSON and
    CSV formats. The method checks the specified format type and serializes the model parameters
    accordingly.

### Load Model Parameters Method

def load_model_parameters(self, model:any, file_path:str) -> None:

Motivation

- JSON and CSV Support: Implemented support for loading model parameters from both JSON
    and CSV formats. The method checks the specified format type and deserializes the model parameters
    accordingly.

### Numpy Array Serialization

for key, value in model_dict.items():
if isinstance(value, np.ndarray):
model_dict[key] = value.tolist()

Motivation

- JSON Serialization of Numpy Arrays: Added a conversion step for numpy arrays to lists to
    ensure they are JSON serializable. This is necessary when saving model parameters in JSON format.

### CSV Reader for Loading

with open(file_path, ’r’, newline=’’) as file:
reader = csv.reader(file)
model_parameters = {rows[0]: float(rows[1]) for rows in reader}
model.__dict__.update(model_parameters)


Motivation

- CSV Loading with Reader: Implemented CSV loading using the csv.reader to handle reading
    rows from the CSV file. The loaded parameters are then updated in the model’s dictionary.

### Exception Handling

else:
raise ValueError("Unsupported format type. Choose ’json’ or ’csv’.")

Motivation

- Unsupported Format Handling: Added a check to raise a ValueError if an unsupported format
    type is specified. This ensures that users choose between ’json’ and ’csv’ to maintain compatibility.

## 4 main.py

The main.py file is an implementation of Multiple Linear Regression using Object-Oriented Programming
principles. It uses various functionalities that are implemented in the previous files, including model train-
ing, prediction, comparison with the sklearn implementation, plotting regression lines and planes, and sav-
ing/loading model parameters.

### Importing Modules

from multiple_linear_regression import MultipleLinearRegression
from sklearn.linear_model import LinearRegression
from regression_plotter import RegressionPlotter
from compare_with_sklearn import ModelComparer
from model_saver import ModelSaver

Motivation

- Modular Design: Imported the necessary classes and modules to keep the code modular and orga-
    nized. This allows for easy access to the functionalities of the linear regression model, plotter, model
    comparer, and model saver.

### Instantiating Objects

model = MultipleLinearRegression(dataset_name=’breast_cancer’)
plotter = RegressionPlotter()
saver = ModelSaver(format_type=’json’)

Motivation

- Object Instantiation: Created instances of the MultipleLinearRegression, RegressionPlotter,
    and ModelSaver classes to use their functionalities. The data setname parameter is set for the multiple
    linear regression model, and the formattype parameter is set for the model saver.

### Training the Model

X, y = model.get_data()
model.train(X, y)


Motivation

- Model Training: Used the getdata method to obtain input data and ground truth for training the
    linear regression model. The train method is then called to calculate the model parameters.

### Comparing with sklearn

# compare our model with sklearn
sklearn_model = LinearRegression()
sklearn_model.fit(X, y)
print("Our model coefficients:", (model._intercept))
print("Sklearn model coefficients:", [sklearn_model.intercept_, *sklearn_model.coef_])
our_predictions = model.predict(X)
sklearn_predictions = sklearn_model.predict(X)
print("MSE our model:", mean_squared_error(y, our_predictions))
print("MSE sklearn model:", mean_squared_error(y, sklearn_predictions))

Motivation

- Model Comparison: Compare the custom linear regression model with sklearn, to showcase the
    differences in coefficients and predictions with the sklearnmodel.

### Making Predictions and Printing Results

print(f"Intercept: {model._intercept}")

predictions = model.predict(X)

print("Ground truth:", y)
print("Predicted value:", predictions)

Motivation

- Result Display: Printed the intercept, ground truth, and predicted values to showcase the perfor-
    mance of the trained linear regression model.

### Plotting Regression Lines and Planes

plotter.set_data(X, y)
plotter.plot_linear_regression(model, [1], 1)

plotter.set_data(X, y)
plotter.plot_linear_regression_plane(model, [2,3], 1)

plotter.set_data(X, y)
plotter.plot_linear_regression(model, [3,4,5,6,7,8], 1)


Motivation

- Visualization: Utilized the RegressionPlotter to visualize the linear regression results. Different
    functionalities were demonstrated, including plotting a regression line, a regression plane, and multiple
    regression lines against the dataset.

### Saving and Loading Model Parameters

saver.save_model_parameters(model, ’model_parameters.json’)

saver.load_model_parameters(model, ’model_parameters.json’)

Motivation

- Model Persistence: Utilized the ModelSaver to save the model parameters in JSON format. The
    saved parameters were then loaded back into the model to showcase the persistence of model state.
