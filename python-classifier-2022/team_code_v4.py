from tqdm import tqdm
import os
import sys
import pickle
import torch
import wandb

from Const import *
from audio_util import AudioUtil
from helper_code import *
from extract_features_utils import *
from mmmt_utils import get_sample
from mmmt_model import *

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    wandb.init(project="MMMT-cvd-detection")
    wandb.config = {
    "learning_rate": LEARNING_RATE,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE
    }

    # Find data files.
    if verbose >= 1:
        print('Finding data files...')

    # Find the patient data files.
    patient_files = find_patient_files(data_folder)
    num_patient_files = len(patient_files)

    if num_patient_files==0:
        raise Exception('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    with open(filtered_features_file_path, 'rb') as f:
        tqdm.write('Preparing data...')

        data = pickle.load(f)
        patient_ids = data['patient_ids']
        train_ids = []

        train_patient_ids = []
        train_aud_embeddings = None
        train_cli_features = None
        train_murmurs = []
        train_outcomes = []

        for patient in tqdm(patient_files, desc='Patients'):
            current_patient_data = load_patient_data(patient)
            current_patient_id = get_patient_id(current_patient_data)

            for i in range(len(patient_ids)):
                if current_patient_id == patient_ids[i]:
                    current_sample = get_sample(current_patient_id, data)

                    train_patient_ids = train_patient_ids + [int(current_sample[0])]
                    train_aud_embeddings = current_sample[1] if train_aud_embeddings is None else torch.vstack((train_aud_embeddings, current_sample[1]))
                    train_cli_features = current_sample[2] if train_cli_features is None else torch.vstack((train_cli_features, current_sample[2]))
                    train_murmurs = train_murmurs + [current_sample[3]]
                    train_outcomes = train_outcomes + [current_sample[4]]

                    train_ids.append(current_patient_id)

        train_ds = SoundDS(train_patient_ids, train_aud_embeddings, train_cli_features, train_murmurs, train_outcomes)
        tqdm.write(f'Training set is built, size: {len(train_murmurs)}')

        # Create training and validation data loaders
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

        # Create the model and put it on the GPU if available
        model = MMMTClassifier()
        if os.path.exists(last_model_path):
            model.load_state_dict(torch.load(last_model_path))
            tqdm.write(f'Loaded model from {last_model_path}')

        model = model.to(device)
        # Check that it is on Cuda
        next(model.parameters()).device

        training(model, train_dl, EPOCHS, LEARNING_RATE)


# Load your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_model(model_folder, verbose):
    model = MMMTClassifier()
    if os.path.exists(last_model_path):
        model.load_state_dict(torch.load(last_model_path))
        tqdm.write(f'Loaded model from {last_model_path}')
    return model

# Save your trained model.
def save_challenge_model(model_folder, imputer, murmur_classes, murmur_classifier, outcome_classes, outcome_classifier):
    d = {'imputer': imputer, 'murmur_classes': murmur_classes, 'murmur_classifier': murmur_classifier, 'outcome_classes': outcome_classes, 'outcome_classifier': outcome_classifier}
    filename = os.path.join(model_folder, 'model_adaboost.sav')
    joblib.dump(d, filename, protocol=0)