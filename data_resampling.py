import os
import numpy as np
import pandas as pd

# the raw signal was recorded at a 60 Hz. This script downsampled the signal to a frequency of 15 Hz

data_folder_path = "/home/bingyi/Desktop/NCSU/Research/federated_lstm/Data"
data_downsampled_folder_path = "/home/bingyi/Desktop/NCSU/Research/federated_lstm/Data_downsampled"

for activity_folder in os.listdir(data_folder_path):
    # print(activity_folder)
    activity_folder_path = os.path.join(data_folder_path,activity_folder)
    activity_downsampled_folder_path = os.path.join(data_downsampled_folder_path,activity_folder)
    # print(activity_folder_path)
    data_activity = []
    for file in os.listdir(activity_folder_path):
        file_path = os.path.join(activity_folder_path,file)
        print(file_path)
        data = pd.read_csv(file_path)
        clean_data = data.dropna()
        data_downsampled = clean_data.iloc[::4,:]
        file_downsampled = file
        file_downsampled_path = os.path.join(activity_downsampled_folder_path, file_downsampled)
        data_downsampled.to_csv(file_downsampled_path, index=False, header=None)

