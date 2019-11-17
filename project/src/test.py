"""
author: rayhan, ibrahim, toriq
File: testing cifar-10 test images
"""

import numpy as np
import pandas as pd
import os

#project modules
from ..import config
from .import preprocess, my_model

#loading my model
model = my_model.read_model()
label = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

#loading test data
result = []
for part in range(0, 6):
    x_test = preprocess.get_test_data_by_part(part)

    #predicting results
    print("predicting result")
    predictions = model.predict(x_test,
                                batch_size = config.batch_size,
                                verbose = 2)
    

    label_pred = np.argmax(predictions, axis = 1)
    print(label_pred)
    
    result += label_pred.tolist()


