import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# This script is used for segmenting the data and normalize the data within each activity/file

data_downsampled_folder_path = "/home/bingyi/Desktop/NCSU/Research/federated_lstm/Data_downsampled"
## calculate the frames of each data recording
# activity_frames = [{}]
# for activity_folder in os.listdir(data_downsampled_folder_path):
#     data_frames = []
#     activity_folder_path = os.path.join(data_downsampled_folder_path,activity_folder)
#     # print(activity_folder_path)
#     for file in os.listdir(activity_folder_path):
#         file_path = os.path.join(activity_folder_path,file)
#         # print(file_path)
#         data =  pd.read_csv(file_path)
#         data_frames.append(data.shape)
#     data_frames = np.array(data_frames)
#     activity_frames_min_value = min(data_frames[:,0])
#     activity_frames_min = {activity_folder:activity_frames_min_value}

#     print(activity_folder)
#     print(list(activity_frames_min.values()))

# Load data from CSV file
def load_data(file_path):
    return pd.read_csv(file_path, header=None)

# Segment the data into chunks
def segment_data(data, length=40, overlap=0):
    num_rows, num_cols = data.shape
    segments = []
    start = 0
    
    while start + length <= num_rows:
        end = start + length
        segment = data.iloc[start:end]
        segments.append(segment.values)
        start += (length - overlap)
    
    # Stack all segments into a higher dimensional array if any segments were created
    if segments:
        result = np.stack(segments)
        return result
    else:
        return np.array([])  # Return an empty array if no segments fit the criteria


sequence_length = 40

for activity_folder in os.listdir(data_downsampled_folder_path):
    print(activity_folder)
    i=0
    data_frames = []
    activity_folder_path = os.path.join(data_downsampled_folder_path,activity_folder)
    segmented_data = None
    segmented_data_norm = None
    
    # segments_ = []
    # print(activity_folder_path)
    for file in os.listdir(activity_folder_path):
        print(file)
        
        file_path = os.path.join(activity_folder_path,file)
        # print(file_path)
        data =  pd.read_csv(file_path)
        segments_ = segment_data(data, sequence_length, 0)
        if segments_.size:
            segments_norm_ = segments_

            # print(segments_.shape)
            # print(segments_norm_.shape)

            if segmented_data is None:
                segmented_data = segments_
            else:
                segmented_data = np.concatenate((segmented_data,segments_),axis=0)

            # normalize the segments
            segments_norm_2d = segments_norm_.reshape(-1,113)  # reshape to to a 2d array
            # create a MinMaxScaler object
            scaler = MinMaxScaler()
            segments_norm_2d[:,2:] = scaler.fit_transform(segments_norm_2d[:,2:])
            segmented_data_norm_3d = segments_norm_2d.reshape(segments_norm_.shape[0],segments_norm_.shape[1],segments_norm_.shape[2])
            
            if segmented_data_norm is None:
                segmented_data_norm = segmented_data_norm_3d
            else:
                segmented_data_norm = np.concatenate((segmented_data_norm,segmented_data_norm_3d),axis=0)
            
            
            # if i == 0:
            #     # print(data)
            #     # print(len(data))
            
            #     segmented_data_norm = segments_
            #     # print(segments_)
            #     # print(len(segments_[1]))
            # else:
            #     segmented_data = np.concatenate((segmented_data,segments_),axis=0)
            #     segments_norm_2d = segments_norm_.reshape(-1,113)  # reshape to to a 2d array
            #     # create a MinMaxScaler object
            #     scaler = MinMaxScaler()
            #     segments_norm_2d[:,2:] = scaler.fit_transform(segments_norm_2d[:,2:])
            #     segmented_data_norm_3d = segments_norm_2d.reshape(segments_norm_.shape[0],segments_norm_.shape[1],segments_norm_.shape[2])
            #     segmented_data_norm = np.concatenate((segmented_data_norm,segmented_data_norm_3d),axis=0)

            # i = i+1 
            # normalize the data using min-max normalization
            

    print(activity_folder,segmented_data.shape)
    print(activity_folder,segmented_data_norm.shape)


    segmented_data_path = os.path.join("/home/bingyi/Desktop/NCSU/Research/federated_lstm/Data_processed", activity_folder)
    segmented_data_norm_path = os.path.join("/home/bingyi/Desktop/NCSU/Research/federated_lstm/Data_processed", activity_folder)

    segmented_data_reshaped = segmented_data.reshape(segmented_data.shape[0], -1)
    segmented_data_norm_reshaped = segmented_data_norm.reshape(segmented_data_norm.shape[0], -1)

    np.savetxt(segmented_data_path + '.csv', segmented_data_reshaped, delimiter=',')
    np.savetxt(segmented_data_path + '_norm.csv', segmented_data_norm_reshaped, delimiter=',')
    # print(segments.shape)
    if np.array_equal( segmented_data_reshaped,segmented_data_norm_reshaped):
        print("yes, both the arrays are the same")
    else:
        print("NO!")
