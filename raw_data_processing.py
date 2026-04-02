import glob
import re
import os
import pandas as pd
import numpy as np

__author__ = "C. I. Tang"
__copyright__ = "Copyright (C) 2020 C. I. Tang"

"""
Based on work of Tang et al.: https://arxiv.org/abs/2011.11542
Contact: cit27@cl.cam.ac.uk
License: GNU General Public License v3.0

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

def process_motion_sense_accelerometer_files(accelerometer_data_folder_path):
    """
    Preprocess the accelerometer files of the MotionSense dataset into the 'user-list' format
    Data files can be found at https://github.com/mmalekzadeh/motion-sense/tree/master/data

    Parameters:

        accelerometer_data_folder_path (str):
            the path to the folder containing the data files (unzipped)
            e.g. motionSense/B_Accelerometer_data/
            the trial folders should be directly inside it (e.g. motionSense/B_Accelerometer_data/dws_1/)

    Return:
        
        user_datsets (dict of {user_id: [(sensor_values, activity_labels)]})
            the processed dataset in a dictionary, of type {user_id: [(sensor_values, activity_labels)]}
            the keys of the dictionary is the user_id (participant id)
            the values of the dictionary are lists of (sensor_values, activity_labels) pairs
                sensor_values are 2D numpy array of shape (length, channels=3)
                activity_labels are 1D numpy array of shape (length)
                each pair corresponds to a separate trial 
                    (i.e. time is not contiguous between pairs, which is useful for making sliding windows, where it is easy to separate trials)
    """

    # label_set = {}
    user_datasets = {}
    all_trials_folders = sorted(glob.glob(accelerometer_data_folder_path + "/*"))

    # Loop through every trial folder
    for trial_folder in all_trials_folders:
        trial_name = os.path.split(trial_folder)[-1]

        # label of the trial is given in the folder name, separated by underscore
        label = trial_name.split("_")[0]
        # label_set[label] = True
        print(trial_folder)
        
        # Loop through files for every user of the trail
        for trial_user_file in sorted(glob.glob(trial_folder + "/*.csv")):

            # use regex to match the user id
            user_id_match = re.search(r'(?P<user_id>[0-9]+)\.csv', os.path.split(trial_user_file)[-1])
            if user_id_match is not None:
                user_id = int(user_id_match.group('user_id'))

                # Read file
                user_trial_dataset = pd.read_csv(trial_user_file)
                user_trial_dataset.dropna(how = "any", inplace = True)

                # Extract the x, y, z channels
                values = user_trial_dataset[["x", "y", "z"]].values

                # the label is the same during the entire trial, so it is repeated here to pad to the same length as the values
                labels = np.repeat(label, values.shape[0])

                if user_id not in user_datasets:
                    user_datasets[user_id] = []
                user_datasets[user_id].append((values, labels))
            else:
                print("[ERR] User id not found", trial_user_file)
    
    return user_datasets

def process_wisdm_raw_file(raw_file_path):
    """
    Preprocess the WISDM dataset raw text file into the 'user-list' format
    Data format: user_id,activity_label,x-acceleration,y-acceleration,z-acceleration

    Parameters:

        raw_file_path (str):
            path to WISDM_ar_v1.1_raw.txt

    Return:
        
        user_datasets (dict of {user_id: [(sensor_values, activity_labels)]})
            the processed dataset in a dictionary, of type {user_id: [(sensor_values, activity_labels)]}
            the keys of the dictionary is the user_id (participant id)
            the values of the dictionary are lists of (sensor_values, activity_labels) pairs
                sensor_values are 2D numpy array of shape (length, channels=3)
                activity_labels are 1D numpy array of shape (length)
                each activity for a user is treated as a separate trial
    """
    user_datasets = {}
    
    # Read the raw file
    with open(raw_file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    data = []
    for line in lines:
        # Remove trailing semicolons
        line = line.rstrip(';')
        parts = line.split(',')
        if len(parts) != 6:  # Some lines might be malformed
            continue
        try:
            user_id = int(parts[0])
            activity = parts[1]
            x = float(parts[2])
            y = float(parts[3])
            z = float(parts[4])
            data.append((user_id, activity, x, y, z))
        except:
            continue
    
    df = pd.DataFrame(data, columns=['user_id', 'activity', 'x', 'y', 'z'])
    
    # Group by user and activity (each (user, activity) is a separate trial)
    for (user_id, activity), group in df.groupby(['user_id', 'activity']):
        values = group[['x', 'y', 'z']].values
        labels = np.repeat(activity, len(values))
        
        if user_id not in user_datasets:
            user_datasets[user_id] = []
        user_datasets[user_id].append((values, labels))
    
    print(f"Loaded WISDM dataset: {len(user_datasets)} users")
    total_samples = sum(len(v) for user_data in user_datasets.values() for v, _ in user_data)
    print(f"Total windows: {total_samples}")
    
    return user_datasets

def process_uci_har(data_folder_path):
    """
    Preprocess the UCI-HAR dataset into the 'user-list' format
    
    Parameters:

        data_folder_path (str):
            path to UCI HAR Dataset folder (unzipped)

    Return:
        
        user_datasets (dict of {user_id: [(sensor_values, activity_labels)]})
            the processed dataset in a dictionary, of type {user_id: [(sensor_values, activity_labels)]}
            the keys of the dictionary is the user_id (participant id)
            the values of the dictionary are lists of (sensor_values, activity_labels) pairs
                sensor_values are 2D numpy array of shape (length, channels=3) (using body_acc xyz)
                activity_labels are 1D numpy array of shape (length)
                each trial is a separate entry
    """
    # Activity label mapping
    activity_map = {
        1: 'WALKING',
        2: 'WALKING_UPSTAIRS',
        3: 'WALKING_DOWNSTAIRS',
        4: 'SITTING',
        5: 'STANDING',
        6: 'LAYING'
    }
    
    # Read train/test subjects
    train_subjects = set()
    with open(os.path.join(data_folder_path, 'train/subject_train.txt'), 'r') as f:
        for line in f:
            train_subjects.add(int(line.strip()))
    
    test_subjects = set()
    with open(os.path.join(data_folder_path, 'test/subject_test.txt'), 'r') as f:
        for line in f:
            test_subjects.add(int(line.strip()))
    
    all_subjects = sorted(train_subjects | test_subjects)
    
    # Load data
    X_train = np.loadtxt(os.path.join(data_folder_path, 'train/Inertial Signals/body_acc_x_train.txt'))
    X_train = X_train.reshape(-1, 128)
    Y_train = np.loadtxt(os.path.join(data_folder_path, 'train/y_train.txt'))
    
    X_test = np.loadtxt(os.path.join(data_folder_path, 'test/Inertial Signals/body_acc_x_test.txt'))
    X_test = X_test.reshape(-1, 128)
    Y_test = np.loadtxt(os.path.join(data_folder_path, 'test/y_test.txt'))
    
    # Get subject list per split
    subjects_train = []
    with open(os.path.join(data_folder_path, 'train/subject_train.txt'), 'r') as f:
        subjects_train = [int(line.strip()) for line in f]
    subjects_test = []
    with open(os.path.join(data_folder_path, 'test/subject_test.txt'), 'r') as f:
        subjects_test = [int(line.strip()) for line in f]
    
    # Load all data per split
    acc_x_train = np.loadtxt(os.path.join(data_folder_path, 'train/Inertial Signals/body_acc_x_train.txt'))
    acc_y_train = np.loadtxt(os.path.join(data_folder_path, 'train/Inertial Signals/body_acc_y_train.txt'))
    acc_z_train = np.loadtxt(os.path.join(data_folder_path, 'train/Inertial Signals/body_acc_z_train.txt'))
    Y_train = np.loadtxt(os.path.join(data_folder_path, 'train/y_train.txt'))
    
    acc_x_test = np.loadtxt(os.path.join(data_folder_path, 'test/Inertial Signals/body_acc_x_test.txt'))
    acc_y_test = np.loadtxt(os.path.join(data_folder_path, 'test/Inertial Signals/body_acc_y_test.txt'))
    acc_z_test = np.loadtxt(os.path.join(data_folder_path, 'test/Inertial Signals/body_acc_z_test.txt'))
    Y_test = np.loadtxt(os.path.join(data_folder_path, 'test/y_test.txt'))
    
    user_datasets = {}
    
    # Process training split
    for i, subject_id in enumerate(subjects_train):
        if subject_id not in user_datasets:
            user_datasets[subject_id] = []
        
        x = acc_x_train[i]
        y = acc_y_train[i]
        z = acc_z_train[i]
        values = np.stack([x, y, z], axis=1)  # (128, 3)
        activity_label = activity_map[int(Y_train[i])]
        labels = np.repeat(activity_label, values.shape[0])
        user_datasets[subject_id].append((values, labels))
    
    # Process test split
    for i, subject_id in enumerate(subjects_test):
        if subject_id not in user_datasets:
            user_datasets[subject_id] = []
        
        x = acc_x_test[i]
        y = acc_y_test[i]
        z = acc_z_test[i]
        values = np.stack([x, y, z], axis=1)  # (128, 3)
        activity_label = activity_map[int(Y_test[i])]
        labels = np.repeat(activity_label, values.shape[0])
        user_datasets[subject_id].append((values, labels))
    
    print(f"Loaded UCI-HAR dataset: {len(user_datasets)} users")
    total_windows = sum(len(user_data) for user_data in user_datasets.values())
    print(f"Total windows: {total_windows}")
    
    return user_datasets

def process_uci_har_6channels(data_folder_path):
    """
    Preprocess the UCI-HAR dataset into the 'user-list' format with 6 channels (acc + gyro)
    
    Parameters:

        data_folder_path (str):
            path to UCI HAR Dataset folder (unzipped)

    Return:
        
        user_datasets (dict of {user_id: [(sensor_values, activity_labels)]})
            the processed dataset in a dictionary, of type {user_id: [(sensor_values, activity_labels)]}
            the keys of the dictionary is the user_id (participant id)
            the values of the dictionary are lists of (sensor_values, activity_labels) pairs
                sensor_values are 2D numpy array of shape (length, channels=6) (acc xyz + gyro xyz)
                activity_labels are 1D numpy array of shape (length)
                each trial is a separate entry
    """
    # Activity label mapping
    activity_map = {
        1: 'WALKING',
        2: 'WALKING_UPSTAIRS',
        3: 'WALKING_DOWNSTAIRS',
        4: 'SITTING',
        5: 'STANDING',
        6: 'LAYING'
    }
    
    # Get subject list per split
    subjects_train = []
    with open(os.path.join(data_folder_path, 'train/subject_train.txt'), 'r') as f:
        subjects_train = [int(line.strip()) for line in f]
    subjects_test = []
    with open(os.path.join(data_folder_path, 'test/subject_test.txt'), 'r') as f:
        subjects_test = [int(line.strip()) for line in f]
    
    # Load all data per split: acc + gyro
    acc_x_train = np.loadtxt(os.path.join(data_folder_path, 'train/Inertial Signals/body_acc_x_train.txt'))
    acc_y_train = np.loadtxt(os.path.join(data_folder_path, 'train/Inertial Signals/body_acc_y_train.txt'))
    acc_z_train = np.loadtxt(os.path.join(data_folder_path, 'train/Inertial Signals/body_acc_z_train.txt'))
    gyro_x_train = np.loadtxt(os.path.join(data_folder_path, 'train/Inertial Signals/body_gyro_x_train.txt'))
    gyro_y_train = np.loadtxt(os.path.join(data_folder_path, 'train/Inertial Signals/body_gyro_y_train.txt'))
    gyro_z_train = np.loadtxt(os.path.join(data_folder_path, 'train/Inertial Signals/body_gyro_z_train.txt'))
    Y_train = np.loadtxt(os.path.join(data_folder_path, 'train/y_train.txt'))
    
    acc_x_test = np.loadtxt(os.path.join(data_folder_path, 'test/Inertial Signals/body_acc_x_test.txt'))
    acc_y_test = np.loadtxt(os.path.join(data_folder_path, 'test/Inertial Signals/body_acc_y_test.txt'))
    acc_z_test = np.loadtxt(os.path.join(data_folder_path, 'test/Inertial Signals/body_acc_z_test.txt'))
    gyro_x_test = np.loadtxt(os.path.join(data_folder_path, 'test/Inertial Signals/body_gyro_x_test.txt'))
    gyro_y_test = np.loadtxt(os.path.join(data_folder_path, 'test/Inertial Signals/body_gyro_y_test.txt'))
    gyro_z_test = np.loadtxt(os.path.join(data_folder_path, 'test/Inertial Signals/body_gyro_z_test.txt'))
    Y_test = np.loadtxt(os.path.join(data_folder_path, 'test/y_test.txt'))
    
    user_datasets = {}
    
    # Process training split
    for i, subject_id in enumerate(subjects_train):
        if subject_id not in user_datasets:
            user_datasets[subject_id] = []
        
        ax = acc_x_train[i]
        ay = acc_y_train[i]
        az = acc_z_train[i]
        gx = gyro_x_train[i]
        gy = gyro_y_train[i]
        gz = gyro_z_train[i]
        values = np.stack([ax, ay, az, gx, gy, gz], axis=1)  # (128, 6)
        
        activity_label = activity_map[int(Y_train[i])]
        labels = np.repeat(activity_label, values.shape[0])
        user_datasets[subject_id].append((values, labels))
    
    # Process test split
    for i, subject_id in enumerate(subjects_test):
        if subject_id not in user_datasets:
            user_datasets[subject_id] = []
        
        ax = acc_x_test[i]
        ay = acc_y_test[i]
        az = acc_z_test[i]
        gx = gyro_x_test[i]
        gy = gyro_y_test[i]
        gz = gyro_z_test[i]
        values = np.stack([ax, ay, az, gx, gy, gz], axis=1)  # (128, 6)
        
        activity_label = activity_map[int(Y_test[i])]
        labels = np.repeat(activity_label, values.shape[0])
        user_datasets[subject_id].append((values, labels))
    
    print(f"Loaded UCI-HAR dataset (6 channels): {len(user_datasets)} users")
    total_windows = sum(len(user_data) for user_data in user_datasets.values())
    print(f"Total windows: {total_windows}")
    
    return user_datasets

def process_pamap2(data_folder_path):
    """
    Preprocess the PAMAP2 dataset into the 'user-list' format
    
    Parameters:

        data_folder_path (str):
            path to folder containing PAMAP2 *.dat files

    Return:
        
        user_datasets (dict of {user_id: [(sensor_values, activity_labels)]})
            the processed dataset in a dictionary, of type {user_id: [(sensor_values, activity_labels)]}
            the keys of the dictionary is the user_id (participant id)
            the values of the dictionary are lists of (sensor_values, activity_labels) pairs
                sensor_values are 2D numpy array of shape (length, channels=3) (using heart-rate acc hand xyz)
                activity_labels are 1D numpy array of shape (length)
    """
    # PAMAP2 activity label mapping
    activity_map = {
        1: 'lying',
        2: 'sitting',
        3: 'standing',
        4: 'walking',
        5: 'running',
        6: 'cycling',
        7: 'nordic_walking',
        8: 'ascending_stairs',
        9: 'descending_stairs',
        10: 'vacuum_cleaning',
        11: 'ironing',
        12: 'rope_jumping'
    }
    
    # Column indices:
    # 0: timestamp, 1: activity_id, 2: heart_rate
    # 3-5: acc-hand xyz,  13-15: acc-chest xyz, 16-18: acc-ankle xyz
    # We'll use hand accelerometer (3 channels) to keep consistent with other datasets
    
    user_datasets = {}
    
    # Find all dat files
    dat_files = sorted(glob.glob(os.path.join(data_folder_path, '*.dat')))
    
    for dat_file in dat_files:
        filename = os.path.basename(dat_file)
        match = re.search(r'subject(\d+)\.dat', filename)
        if not match:
            continue
        user_id = int(match.group(1))
        
        # Read data
        data = []
        with open(dat_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 18:  # Skip incomplete lines
                    continue
                try:
                    activity_id = int(float(parts[1]))
                    if activity_id == 0:  # transient
                        continue
                    x = float(parts[3])
                    y = float(parts[4])
                    z = float(parts[5])
                    data.append((activity_id, x, y, z))
                except:
                    continue
        
        if not data:
            continue
        
        # Convert to numpy
        data_np = np.array(data)
        activity_ids = data_np[:, 0].astype(int)
        values = data_np[:, 1:]
        
        # Group by activity (each activity is a separate trial)
        for activity_id, group in pd.DataFrame(data_np).groupby(0):
            activity_id = int(activity_id)
            if activity_id not in activity_map:
                continue  # Skip activities not in our mapping (optional activities)
            activity_label = activity_map[activity_id]
            values_activity = group[[1, 2, 3]].values
            labels = np.repeat(activity_label, values_activity.shape[0])
            
            if user_id not in user_datasets:
                user_datasets[user_id] = []
            user_datasets[user_id].append((values_activity, labels))
    
    print(f"Loaded PAMAP2 dataset: {len(user_datasets)} users")
    total_samples = sum(len(v) for user_data in user_datasets.values() for v, _ in user_data)
    print(f"Total samples: {total_samples}")
    
    return user_datasets
    


