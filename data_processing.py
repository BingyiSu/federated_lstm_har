import os
import numpy as np
import pandas as pd
import csv

# This script is used to combine all activity data and labeling each of them 

# load the data
Free_body =  np.loadtxt('/home/bingyi/Desktop/NCSU/Research/federated_lstm/Data_processed/Free_body_norm.csv', delimiter=',')
Lifting =  np.loadtxt('/home/bingyi/Desktop/NCSU/Research/federated_lstm/Data_processed/Lifting_norm.csv', delimiter=',')
Lowering =  np.loadtxt('/home/bingyi/Desktop/NCSU/Research/federated_lstm/Data_processed/Lowering_norm.csv', delimiter=',')
Reaching =  np.loadtxt('/home/bingyi/Desktop/NCSU/Research/federated_lstm/Data_processed/Reaching_norm.csv', delimiter=',')
Squating =  np.loadtxt('/home/bingyi/Desktop/NCSU/Research/federated_lstm/Data_processed/Squating_norm.csv', delimiter=',')
Step_up =  np.loadtxt('/home/bingyi/Desktop/NCSU/Research/federated_lstm/Data_processed/Step_up_norm.csv', delimiter=',')
Walking =  np.loadtxt('/home/bingyi/Desktop/NCSU/Research/federated_lstm/Data_processed/Walking_norm.csv', delimiter=',')

print("Free body shape:", Free_body.shape)
print("Lifting shape:", Lifting.shape)
print("Lowering shape:", Lowering.shape)
print("Reaching shape:", Reaching.shape)
print("Squating shape:", Squating.shape)
print("Step up shape:", Step_up.shape)
print("Walking shape:", Walking.shape)


# reshape the data to original sequence data
sequence_length = 40
Free_body_reshaped = Free_body.reshape(Free_body.shape[0],sequence_length, -1)
Lifting_reshaped = Lifting.reshape(Lifting.shape[0],sequence_length, -1)
Lowering_reshaped = Lowering.reshape(Lowering.shape[0],sequence_length, -1)
Reaching_reshaped = Reaching.reshape(Reaching.shape[0],sequence_length, -1)
Squating_reshaped = Squating.reshape(Squating.shape[0],sequence_length, -1)
Step_up_reshaped = Step_up.reshape(Step_up.shape[0],sequence_length, -1)
Walking_reshaped = Walking.reshape(Walking.shape[0],sequence_length, -1)

# Assigning labels to each activity
activity_labels = {
    'Lifting': 0,
    'Lowering': 1,
    'Reaching': 2,
    # 'Squating': 2,
    'Step_up': 3,
    'Walking': 4
}

datasets = [Lifting_reshaped, Lowering_reshaped, Reaching_reshaped, 
            Squating_reshaped, Step_up_reshaped, Walking_reshaped]
all_labels = []
all_data = []

for activity, reshaped_data in [
    ('Lifting',Lifting_reshaped),
    ('Lowering',Lowering_reshaped),
    ('Reaching',Reaching_reshaped),
    # ('Squating',Squating_reshaped),
    ('Step_up',Squating_reshaped),
    ('Walking',Walking_reshaped),
]:
    # number of samples in the dataset
    num_samples = reshaped_data.shape[0]

    # create a labels array for this activity
    labels_array = np.full(num_samples,activity_labels[activity])
    print("shape of leables_array", labels_array.shape)

    all_labels.append(labels_array)
    all_data.append(reshaped_data)



# concatenate all datasets
combined_labels = np.concatenate(all_labels,axis=0)
combined_data = np.concatenate(all_data,axis=0)
combined_data_reshaped = combined_data.reshape(combined_data.shape[0],-1)

print("shape of combined labels:", combined_labels.shape)
print("shape of combined data:", combined_data.shape)

path = '/home/bingyi/Desktop/NCSU/Research/federated_lstm/'

np.savetxt(path + 'combined_data_reshaped.csv', combined_data_reshaped, delimiter=',')
np.savetxt(path + 'combined_labels.csv', combined_labels, delimiter=',')


# simplify the data (37 markers) to a 17 markers data to reduce computation cost


# Define the indices for joints, where each joint has three columns (X, Y, Z)
joint_indices = {
    'H': [0, 1, 2, 3, 4, 5, 6, 7, 8],  # Averaging joints 1, 2, 3 as head
    'H_and_7': [0, 1, 2,21,22,23],  # To be calculated after initial M as neck
    '4': [12, 13, 14],  # left acromion
    '5':[15, 16, 17], # right acromion
    '6':[21, 22, 23], # c7
    '8':[24, 25, 26], # T8
    '11_13': [30, 31, 32, 36, 37, 38],  # Average joints 11 and 13 as left elbow
    '15_17': [42, 43, 44, 48, 49, 50],  # Average joints 15 and 17 as left wrist
    '10_12': [27, 28, 29, 33, 34, 35],  # Average joints 10 and 12 as right elbow
    '14_16': [39, 40, 41, 45, 46, 47],  # Average joints 14 and 16 as right wrist
    '18_20':[51, 52, 53, 57, 58, 59], # right hip
    '19_21':[54, 55, 56, 60, 61, 62], # left hip
    '18_21': [51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62],  # Average joints 18 to 21 as center
    '22_24_26': [63, 64, 65, 69, 70, 71, 75, 76, 77],  # Average joints 22, 24, 26 as right knee
    '23_25_27': [66, 67, 68, 72, 73, 74, 78, 79, 80],  # Average joints 23, 25, 27 as left knee
    '28_30_32_34': [81, 82, 83, 87, 88, 89, 93, 94, 95, 99, 100, 101],  # Average joints 28, 30, 32, 34 as right ankel
    '29_31_33_35': [84, 85, 86, 90, 91, 92, 96, 97, 98, 102, 103, 104]  # Average joints 29, 31, 33, 35 as left ankel
}

# FUnction to average joints separately for X, Y, and Z
def average_xyz(data, indices):
    x_indices = indices[0::3]
    print(x_indices)
    y_indices = indices[1::3]
    z_indices = indices[2::3]
    x_avg = np.mean(data[:,:,x_indices],axis=2,keepdims=True)
    y_avg = np.mean(data[:,:,y_indices],axis=2,keepdims=True)
    z_avg = np.mean(data[:,:,z_indices],axis=2,keepdims=True)
    return np.concatenate([x_avg,y_avg,z_avg],axis=2)

data = combined_data[:,:,2:-6]

new_joint_data = []

for key, indices in joint_indices.items():
    print(indices)
    joint_data = average_xyz(data,indices)
    new_joint_data.append(joint_data)

# concatenate the new joint data along the last axis to form the new data array
simplified_combined_data = np.concatenate(new_joint_data,axis=2)
simplified_combined_data_reshaped = simplified_combined_data.reshape(simplified_combined_data.shape[0],-1)

np.savetxt(path + 'simplified_combined_data_reshaped.csv', simplified_combined_data_reshaped, delimiter=',')
# Print the new shape to verify
print("New shape of combined data:", simplified_combined_data.shape)
print("New shape of combined data reshaped:", simplified_combined_data_reshaped.shape)