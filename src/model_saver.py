'''
This module contains the ModelSaver class, which is used to save and load model.
Either in json or csv format.
'''

import numpy as np
import json
import csv

class ModelSaver:
    def __init__(self, format_type:str):
        self._format_type = format_type
    
    # add type checks
    def save_model_parameters(self, model:any, file_path:str) -> None:
        if self._format_type == 'json':
            model_dict = model.__dict__.copy()
            # Convert numpy arrays to lists to make them JSON serializable
            for key, value in model_dict.items():
                if isinstance(value, np.ndarray):
                    model_dict[key] = value.tolist()
            # Save the model parameters to a file
            with open(file_path, 'w') as file:
                json.dump(model_dict, file)
        elif self._format_type == 'csv':
            # Save the model parameters to a file
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                # Write the model parameters to the file
                for key, value in model.__dict__.items():
                    writer.writerow([key, value])
        else:
            raise ValueError("Unsupported format type. Choose 'json' or 'csv'.")

    def load_model_parameters(self, model:any, file_path:str) -> None:
        if self._format_type == 'json':
            # Load the model parameters from a file
            with open(file_path, 'r') as file:
                model_parameters = json.load(file)
                model.__dict__.update(model_parameters)
        elif self._format_type == 'csv':
            with open(file_path, 'r', newline='') as file:
                # Load the model parameters from a file
                reader = csv.reader(file)
                model_parameters = {rows[0]: float(rows[1]) for rows in reader}
                model.__dict__.update(model_parameters)
        else:
            raise ValueError("Unsupported format type. Choose 'json' or 'csv'.")
