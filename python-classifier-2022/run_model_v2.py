#!/usr/bin/env python

# Do *not* edit this script. Changes will be discarded so that we can process the models consistently.

# This file contains functions for running models for the 2022 Challenge. You can run it as follows:
#
#   python run_model.py model data outputs
#
# where 'model' is a folder containing the your trained model, 'data' is a folder containing the Challenge data, and 'outputs' is a
# folder for saving your model's outputs.

import numpy as np, os, sys
import pickle
from tqdm import tqdm

from Const import *
from mmmt_utils import get_sample
from mmmt_model import *
from helper_code import *
from team_code_v4 import load_challenge_model

# Run model.
def run_model(model_folder, data_folder, output_folder, allow_failures, verbose):
    # Load models.
    if verbose >= 1:
        print('Loading Challenge model...')
    model = MMMTClassifier()
    if os.path.exists(last_model_path):
        model.load_state_dict(torch.load(last_model_path))
        tqdm.write(f'Loaded model from {last_model_path}')

    model = model.to(device)
    # Check that it is on Cuda
    next(model.parameters()).device

    # Create a folder for the Challenge outputs if it does not already exist.
    os.makedirs(output_folder, exist_ok=True)

    # Run the team's model on the Challenge data.
    if verbose >= 1:
        print('Running model on Challenge data...')

    # Find the patient data files.
    patient_files = find_patient_files(data_folder)
    num_patient_files = len(patient_files)
    if num_patient_files==0:
        raise Exception('No data was provided.')

    output_files = {}

    with open(filtered_features_file_path, 'rb') as f:
        tqdm.write('Preparing data...')

        data = pickle.load(f)
        patient_ids = data['patient_ids']

        test_patient_ids = []
        test_aud_embeddings = None
        test_cli_features = None
        test_murmurs = []
        test_outcomes = []

        for patient in tqdm(patient_files, desc='Patients'):
            current_patient_data = load_patient_data(patient)
            current_patient_id = get_patient_id(current_patient_data)

            for i in range(len(patient_ids)):
                if current_patient_id == patient_ids[i]:
                    current_sample = get_sample(current_patient_id, data)

                    test_patient_ids = test_patient_ids + [int(current_sample[0])]
                    test_aud_embeddings = current_sample[1] if test_aud_embeddings is None else torch.vstack((test_aud_embeddings, current_sample[1]))
                    test_cli_features = current_sample[2] if test_cli_features is None else torch.vstack((test_cli_features, current_sample[2]))
                    test_murmurs = test_murmurs + [current_sample[3]]
                    test_outcomes = test_outcomes + [current_sample[4]]

            # Save Challenge outputs.
            head, tail = os.path.split(patient)
            root, extension = os.path.splitext(tail)
            output_file = os.path.join(output_folder, root + '.csv')

            output_files[int(current_patient_id)] = output_file

        test_ds = SoundDS(test_patient_ids, test_aud_embeddings, test_cli_features, test_murmurs, test_outcomes)
        # tqdm.write(f'Test set is built, size: {len(test_murmurs)}')
        test_dl = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False)

        inference(model, test_dl, output_files)

        if verbose >= 1:
            print('Done.')

if __name__ == '__main__':
    # Parse the arguments.
    if not (len(sys.argv) == 4 or len(sys.argv) == 5):
        raise Exception('Include the model, data, and output folders as arguments, e.g., python run_model.py model data outputs.')

    # Define the model, data, and output folders.
    model_folder = sys.argv[1]
    data_folder = sys.argv[2]
    output_folder = sys.argv[3]

    # Allow or disallow the model to fail on parts of the data; helpful for debugging.
    allow_failures = False

    # Change the level of verbosity; helpful for debugging.
    if len(sys.argv)==5 and is_integer(sys.argv[4]):
        verbose = int(sys.argv[4])
    else:
        verbose = 1

    run_model(model_folder, data_folder, output_folder, allow_failures, verbose)
