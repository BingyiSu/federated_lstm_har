import os
import numpy as np
import pandas as pd

# the raw signal was recorded at a 60 Hz. This script downsampled the signal to a frequency of 15 Hz

data_downsampled_folder_path = "/home/bingyi/Desktop/NCSU/Research/federated_lstm/Data_downsampled"

for activity_folder in os.listdir(data_downsampled_folder_path):
    activity_folder_path = os.path.join(data_downsampled_folder_path,activity_folder)
    
    # segments_ = []
    # print(activity_folder_path)
    for file in os.listdir(activity_folder_path):
        file_path = os.path.join(activity_folder_path,file)
        # print(file_path)
        data =  pd.read_csv(file_path)
        clean_data = data.dropna()
        np.savetxt(file_path,data, delimiter=',')


