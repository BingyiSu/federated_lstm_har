import os
import shutil

main_directory_path = "/home/bingyi/Desktop/NCSU/Research/federated_lstm/working_posture_trc_txt"
new_folder_path_walking = "/home/bingyi/Desktop/NCSU/Research/federated_lstm/Data/Walking"
new_folder_path_reaching = "/home/bingyi/Desktop/NCSU/Research/federated_lstm/Data/Reaching"
new_folder_path_lifting = "/home/bingyi/Desktop/NCSU/Research/federated_lstm/Data/Lifting"
new_folder_path_lowering = "/home/bingyi/Desktop/NCSU/Research/federated_lstm/Data/Lowering"
new_folder_path_squating = "/home/bingyi/Desktop/NCSU/Research/federated_lstm/Data/Squating"
new_folder_path_step_up = "/home/bingyi/Desktop/NCSU/Research/federated_lstm/Data/Step_up"
new_folder_path_free_body = "/home/bingyi/Desktop/NCSU/Research/federated_lstm/Data/Free_body"

def move_files_to_folder(new_folder_path,file_activity_number):
    main_directory_path = "/home/bingyi/Desktop/NCSU/Research/federated_lstm/working_posture_trc_txt"

    # iterate over each sufolder in the main directory
    for subfolder in os.listdir(main_directory_path):
        subfolder_path = os.path.join(main_directory_path,subfolder)

        # check if the current item is indeed a folder
        if os.path.isdir(subfolder_path):
            # iterate over each file in the fubfolder
            for filename in os.listdir(subfolder_path):
                # check if the filename matches the pattern
                if filename.startswith(subfolder) and filename.endswith(file_activity_number):
                    # full path to the file
                    file_path = os.path.join(subfolder_path, filename)
                    # full path to the new location
                    new_file_path = os.path.join(new_folder_path,filename)
                    # move the file
                    shutil.copy2(file_path, new_file_path)

# create the new folder if it does not already exist
activity_path = [new_folder_path_walking,new_folder_path_reaching, new_folder_path_lifting, new_folder_path_lowering, new_folder_path_squating,
            new_folder_path_step_up, new_folder_path_free_body]
for path in activity_path:
    
    if not os.path.exists(path):
        os.makedirs(path)
        
    # walking
    if path == new_folder_path_walking:
        file_activity_number = ["02_01_trc.txt","10_01_trc.txt"]
        for file_number in file_activity_number:
            move_files_to_folder(path, file_number)

    # reaching
    if path == new_folder_path_reaching:
        file_activity_number = ["04_01_trc.txt","04_02_trc.txt","04_03_trc.txt",
                                "05_01_trc.txt","05_02_trc.txt","05_03_trc.txt",
                                "06_01_trc.txt","06_02_trc.txt","06_03_trc.txt"]
        for file_number in file_activity_number:
            move_files_to_folder(path, file_number)

    # squating
    if path == new_folder_path_squating:
        file_activity_number = ["07_01_trc.txt","07_02_trc.txt","07_03_trc.txt","07_04_trc.txt",
                                "08_01_trc.txt","08_02_trc.txt","08_03_trc.txt","08_04_trc.txt"]
        for file_number in file_activity_number:
            move_files_to_folder(path, file_number)

    #step up
    if path == new_folder_path_step_up:
        file_activity_number = ["11_01_trc.txt","11_02_trc.txt","11_03_trc.txt","11_04_trc.txt",
                                "12_01_trc.txt","12_02_trc.txt","12_03_trc.txt","12_04_trc.txt"]
        for file_number in file_activity_number:
            move_files_to_folder(path, file_number)

    # lifting
    if path == new_folder_path_lifting:
        file_activity_number = ["13_01_trc.txt","13_02_trc.txt","13_03_trc.txt",
                                "14_01_trc.txt","14_02_trc.txt","14_03_trc.txt",
                                "15_01_trc.txt","15_02_trc.txt","15_03_trc.txt",
                                "16_01_trc.txt","16_02_trc.txt","16_03_trc.txt",
                                "17_01_trc.txt","17_02_trc.txt","17_03_trc.txt",
                                "18_01_trc.txt","18_02_trc.txt","18_03_trc.txt"]
        for file_number in file_activity_number:
            move_files_to_folder(path, file_number)

    # lowering
    if path == new_folder_path_lowering:
        file_activity_number = ["19_01_trc.txt","19_02_trc.txt","19_03_trc.txt",
                                "20_01_trc.txt","20_02_trc.txt","20_03_trc.txt",
                                "21_01_trc.txt","21_02_trc.txt","21_03_trc.txt",
                                "22_01_trc.txt","22_02_trc.txt","22_03_trc.txt",
                                "23_01_trc.txt","23_02_trc.txt","23_03_trc.txt",
                                "24_01_trc.txt","24_02_trc.txt","24_03_trc.txt"]
        for file_number in file_activity_number:
            move_files_to_folder(path, file_number)

    # free body
    if path == new_folder_path_free_body:
        file_activity_number = ["25_01_trc.txt"]
        for file_number in file_activity_number:
            move_files_to_folder(path, file_number)

# # After moving, list the contents of the new folder to verify
# moved_files = os.listdir(new_folder_path_walking)
# print(moved_files)