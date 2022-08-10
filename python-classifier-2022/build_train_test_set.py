import os
import shutil
import splitfolders
# Clean up to make sure that files are in the same folder with their folder file
from os.path import exists

# Mix classes 

from os import listdir, walk, makedirs
from os.path import isfile, dirname, join

from helper_code import *

DATA_FOLDER = "/media/data/HeartMurmurDetection/physionet.org/files/circor-heart-sound/1.0.3/training_data"
OUTPUT_FOLDER = "/media/data/HeartMurmurDetection/circor-heart-sound"

# Find the patient data files.
patient_files = find_patient_files(DATA_FOLDER)
num_patient_files = len(patient_files)

# Create a folder for the splitted data if one does not already exist
target_output_folder = os.path.join(OUTPUT_FOLDER, 'unsplitted')
os.makedirs(target_output_folder, exist_ok=True)

murmur_classes = ['Present', 'Unknown', 'Absent']
num_murmur_classes = len(murmur_classes)
outcome_classes = ['Abnormal', 'Normal']
num_outcome_classes = len(outcome_classes)

os.makedirs(os.path.join(target_output_folder, 'Present'), exist_ok=True)
os.makedirs(os.path.join(target_output_folder, 'Unknown'), exist_ok=True)
os.makedirs(os.path.join(target_output_folder, 'Absent'), exist_ok=True)

for i in range(num_patient_files):
    print('    {}/{}...'.format(i+1, num_patient_files))

    # Load the current patient data and recordings.
    current_patient_data = load_patient_data(patient_files[i])

    # Extract labels 
    current_murmur = get_murmur(current_patient_data)
    destination_folder = os.path.join(target_output_folder, current_murmur)
    patient_id = get_patient_id(current_patient_data)
    patient_file = os.path.join(DATA_FOLDER, patient_id + '.txt')
    shutil.copy(patient_file, destination_folder)

    num_locations = get_num_locations(current_patient_data)
    recording_information = current_patient_data.split('\n')[1:num_locations+1]

    for j in range(num_locations):
        entries = recording_information[j].split(' ')
        for filename in entries[1:]:
            shutil.copy(os.path.join(DATA_FOLDER, filename), os.path.join(destination_folder, filename))

    import splitfolders

data_folder = os.path.join(OUTPUT_FOLDER, 'unsplitted')
output_folder = os.path.join(OUTPUT_FOLDER, 'splitted')

splitfolders.ratio(data_folder, output=output_folder, seed=1337, ratio=(.8, 0, .2)) 

splitted_data_path = join(OUTPUT_FOLDER, 'splitted')
output_path = join(OUTPUT_FOLDER, "final")

for set in listdir(splitted_data_path):
    set_path = join(output_path, set)
    makedirs(set_path, exist_ok=True)
    for root, dirs, files in walk(join(splitted_data_path, set)):
        for f in files:
            file_path = join(root, f)
            shutil.copy2(file_path, set_path)
    # print(set)

other_set = {
    'train' : 'test',
    'test' : 'train'
}

for set in listdir(output_path):
    for root, dirs, files in walk(join(output_path, set)):
        for f in files:
            if '.txt' not in f: 
                patient_id = f.split('_')[0]
                if not exists(join(root, patient_id+'.txt')):
                    shutil.move(join(root, f), join(output_path, other_set[set]))