import os
import sys
import pickle
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
from torch.nn import init, ZeroPad2d, Module
import torch.nn as nn
from tqdm import tqdm

from Const import *
from audio_util import AudioUtil
from helper_code import *
from extract_features_utils import *
from mmmt_utils import get_sample
from mmmt_model import *


class ClinicalClassifier(nn.Module):
    def __init__(self):
        super(ClinicalClassifier, self).__init__()

        # For clinical features
        self.cli_fc1 = nn.Linear(CLINICAL_FEATURES_DIM, 16)
        self.cli_fc2 = nn.Linear(16, 64)

        # For audio features
        self.aud_fc1 = nn.Linear(AUDIO_FEATURES_DIM, 1024)
        self.aud_fc2 = nn.Linear(1024, 256)

        # Concat layer for the combined feature space for murmur classification
        self.output_murmurt_fc = nn.Linear(64, 3)

        # Concat layer for the combined feature space for outcome classification
        self.output_outcome_fc = nn.Linear(64, 2)

    def forward(self, audio_embeddings, clinical_features):
        cli_out = F.relu(self.cli_fc2(self.cli_fc1(clinical_features)))

        murmur = self.output_murmurt_fc(cli_out)
        outcome = self.output_outcome_fc(cli_out)

        return [murmur, outcome]

# Training
def training(model, train_dl, num_epochs):
    # Loss Function, Optimizer and Scheduler
    murmur_criterion = nn.CrossEntropyLoss()
    outcome_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=2e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-4,
                                                steps_per_epoch=int(len(train_dl)),
                                                epochs=num_epochs,
                                                anneal_strategy='linear')

    # Repeat for each epoch
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_murmur_prediction = 0
        correct_outcome_prediction = 0
        total_prediction = 0

        # Repeat for each batch in the training set
        for i, data in tqdm(enumerate(train_dl), desc='Training'):
            # Get the input features and target labels, and put them on the GPU
            audio_embeddings, clinical_features, murmurs, outcomes = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)
            
            # print(data)
            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            murmur_outputs, outcome_outputs = model(audio_embeddings, clinical_features)

            # print(f'{murmur_outputs} - {murmurs}')
            # print(f'{murmur_outputs.shape} - {murmurs.shape}')
            
            murmur_loss = murmur_criterion(murmur_outputs, murmurs)
            outcome_loss = outcome_criterion(outcome_outputs, outcomes)
            loss = (murmur_loss + outcome_loss) / 2

            # print(f'{loss} - {murmur_loss}/{outcome_loss}')

            loss.backward()
            optimizer.step()
            scheduler.step()

            # Keep stats for Loss and Accuracy
            running_loss += loss.item()

            # Get the predicted class with the highest score
            _, murmur_prediction = torch.max(murmur_outputs,1)
            _, outcome_prediction = torch.max(outcome_outputs,1)
            # Count of predictions that matched the target label
            correct_murmur_prediction += (murmur_prediction == murmurs).sum().item()
            correct_outcome_prediction += (outcome_prediction == outcomes).sum().item()
            
            total_prediction += murmur_prediction.shape[0]

            # print(f'Prediction: {murmur_prediction} - {outcome_prediction}')
            # print(f'{correct_murmur_prediction} / {total_prediction}')

            if i % 10 == 0:    # print every 10 mini-batches
               tqdm.write('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
    
        # Print stats at the end of the epoch
        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        murmur_acc = correct_murmur_prediction/total_prediction
        outcome_acc = correct_outcome_prediction/total_prediction
        tqdm.write(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Murmur accuracy: {murmur_acc:.2f}, Outcome accuracy: {murmur_acc:.2f}')

        torch.save(model.state_dict(), f'{cli_only_model_path}_{epoch}.sav')
        torch.save(model.state_dict(), last_cli_only_model_path)

    tqdm.write('Finished Training')


# Find data files.
# Find the patient data files.
data_folder = "/media/data/minhpv/circor-heart-sound/final/train"
patient_files = find_patient_files(data_folder)
num_patient_files = len(patient_files)

if num_patient_files==0:
    raise Exception('No data was provided.')

# Extract the features and labels.
with open(filtered_features_file_path, 'rb') as f:
    tqdm.write('Preparing data...')

    data = pickle.load(f)
    patient_ids = data['patient_ids']
    train_ids = []

    train_aud_embeddings = None
    train_cli_features = None
    train_murmurs = []
    train_outcomes = []

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

                train_aud_embeddings = current_sample[0] if train_aud_embeddings is None else torch.vstack((train_aud_embeddings, current_sample[0]))
                train_cli_features = current_sample[1] if train_cli_features is None else torch.vstack((train_cli_features, current_sample[1]))
                train_murmurs = train_murmurs + [current_sample[2]]
                train_outcomes = train_outcomes + [current_sample[3]]

                train_ids.append(current_patient_id)

    train_ds = SoundDS(train_aud_embeddings, train_cli_features, train_murmurs, train_outcomes)
    tqdm.write(f'Training set is built, size: {len(train_murmurs)}')

    for i in range((len(patient_ids))):
        if patient_ids[i] not in train_ids:
            current_sample = get_sample(patient_ids[i], data)

            test_aud_embeddings = current_sample[0] if test_aud_embeddings is None else torch.vstack((test_aud_embeddings, current_sample[0]))
            test_cli_features = current_sample[1] if test_cli_features is None else torch.vstack((test_cli_features, current_sample[1]))
            test_murmurs = test_murmurs + [current_sample[2]]
            test_outcomes = test_outcomes + [current_sample[3]]

    test_ds = SoundDS(test_aud_embeddings, test_cli_features, test_murmurs, test_outcomes)
    tqdm.write(f'Test set is built, size: {len(test_murmurs)}')

    # Create training and validation data loaders
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=16, shuffle=False)

    # Create the model and put it on the GPU if available
    model = ClinicalClassifier()
    if os.path.exists(last_cli_only_model_path):
        model.load_state_dict(torch.load(last_cli_only_model_path))
        tqdm.write(f'Loaded model from {last_cli_only_model_path}')

    model = model.to(device)
    # Check that it is on Cuda
    next(model.parameters()).device

    num_epochs = 20
    training(model, train_dl, num_epochs)