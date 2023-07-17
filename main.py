from fastapi import FastAPI
# from typing import List
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names

app = FastAPI()

# Load the necessary data and models
prod_dow_hour = pd.read_csv("data/prod_dow_hour.csv")

with open('dictionary.pkl', 'rb') as file:
    loaded_dictionary = pickle.load(file)

with open('prod_dict.pkl', 'rb') as file1:
    prod_dictionary = pickle.load(file1)

loaded_model = load_model("savedmodel/")

@app.get("/recommend/{dow}/{hour_of_day}")
def recommend_items(dow: int, hour_of_day: int):
    dict = {
        'product_id': [],
        'order_dow': [],
        'order_hour_of_day': [],
        'aisle_id': [],
        'department_id': []
    }

    for key, (value1, value2) in loaded_dictionary.items():
        dict['product_id'].append(key)
        dict['order_dow'].append(dow)
        dict['order_hour_of_day'].append(hour_of_day)
        dict['aisle_id'].append(value1)
        dict['department_id'].append(value2)

    for key in dict:
        dict[key] = np.array(dict[key]).astype('int64')

    # Use the loaded model for predictions
    pred_ans = loaded_model.predict(dict, batch_size=256)

    # Selecting indices of the items with highest recommendation values
    # Flatten the array to get a 1-dimensional array
    flattened_arr = pred_ans.flatten()

    # Get the indices that would sort the array in descending order
    sorted_indices = np.argsort(flattened_arr)[::-1]

    # Get the top 10 indices
    top_10_indices = sorted_indices[:10]

    # Prepare the recommended items list
    recommendations = []
    for index in top_10_indices:
        if index in prod_dictionary:
            prod_name = prod_dictionary[index]
            recommendations.append(prod_name)

    return recommendations
