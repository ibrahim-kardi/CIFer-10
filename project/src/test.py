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

