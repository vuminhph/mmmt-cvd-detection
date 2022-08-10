import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
from torch.nn import init, ZeroPad2d, Module
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torchmetrics.classification import Accuracy, F1Score
import wandb

from Const import *
from helper_code import save_challenge_outputs
from evaluate_model import compute_weighted_accuracy, enforce_positives

# Define the Dataset
class SoundDS(Dataset):
    def __init__(self, patient_ids, audio_embeddings, clinical_features, murmurs, outcomes):
        self.patient_ids = patient_ids
        self.audio_embeddings = audio_embeddings
        self.clinical_features = clinical_features
        self.murmurs = murmurs
        self.outcomes = outcomes

    def __len__(self):
        return len(self.audio_embeddings)
	
    def __getitem__(self, idx):
        return (self.patient_ids[idx], self.audio_embeddings[idx], self.clinical_features[idx], self.murmurs[idx], self.outcomes[idx])


# Define the Classifier
class MMMTClassifier(nn.Module):
    def __init__(self):
        super(MMMTClassifier, self).__init__()

        # For clinical features
        self.cli_fc1 = nn.Linear(CLINICAL_FEATURES_DIM, 16)
        self.cli_fc2 = nn.Linear(16, 32)
        self.cli_fc3 = nn.Linear(32, 128)
        self.cli_fc4 = nn.Linear(128, 256)

        # For audio features
        self.aud_fc1 = nn.Linear(AUDIO_FEATURES_DIM, 1024 * 4)
        self.aud_fc2 = nn.Linear(1024 * 4, 1024)
        self.aud_fc3 = nn.Linear(1024, 256)

        # Concat layer for the combined feature space for murmur classification
        self.output_murmurt_fc = nn.Linear(512, 3)

        # Concat layer for the combined feature space for outcome classification
        self.output_outcome_fc = nn.Linear(512, 2)

    def forward(self, audio_embeddings, clinical_features):
        cli_out1 = F.relu(self.cli_fc1(clinical_features))
        cli_out2 = F.relu(self.cli_fc2(cli_out1))
        cli_out3 = F.relu(self.cli_fc3(cli_out2))
        cli_out = F.relu(self.cli_fc4(cli_out3))

        aud_out1 = F.relu(self.aud_fc1(audio_embeddings))
        aud_out2 = F.relu(self.aud_fc2(aud_out1))
        aud_out = F.relu(self.aud_fc3(aud_out2))

        combined_inp = torch.cat((cli_out, aud_out), 1)

        murmur = self.output_murmurt_fc(combined_inp)
        outcome = self.output_outcome_fc(combined_inp)

        return [murmur, outcome]


class MMMT_FE1(nn.Module):
    def __init__(self):
        super(MMMT_FE1, self).__init__()

        # For clinical features
        self.cli_fc1 = nn.Linear(CLINICAL_FEATURES_DIM, 16)
        self.cli_fc2 = nn.Linear(16, 64)

    def forward(self, clinical_features):
        cli_out = F.relu(self.cli_fc2(self.cli_fc1(clinical_features)))

        return cli_out

# Training
def training(model, train_dl, num_epochs, learning_rate):
    murmur_Accuracy = Accuracy().to(device)
    murmur_F1Score = F1Score(num_classes=3).to(device)

    outcome_Accuracy = Accuracy().to(device)
    outcome_F1Score = F1Score(num_classes=2).to(device)

    # Loss Function, Optimizer and Scheduler
    murmur_criterion = nn.CrossEntropyLoss()
    outcome_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate,
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
        print(len(train_dl))
        with tqdm(len(train_dl), desc="Training") as pbar:
            for i, data in enumerate(train_dl):
                pbar.update(1)
                # Get the input features and target labels, and put them on the GPU
                patient_ids, audio_embeddings, clinical_features, murmurs, outcomes = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device)
                
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

                wandb.log({"loss": loss})

                # Optional
                wandb.watch(model)

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

                murmur_acc = murmur_Accuracy(murmur_outputs, murmurs)
                murmur_f1 = murmur_F1Score(murmur_outputs, murmurs)
                outcome_acc = outcome_Accuracy(outcome_outputs, outcomes)
                outcome_acc = outcome_F1Score(outcome_outputs, outcomes)
                
                murmur_labels = enforce_positives(murmurs.cpu().detach().numpy(), murmur_classes, 'Present')
                murmur_prediction = enforce_positives(murmur_prediction.cpu().detach().numpy(), murmur_classes, 'Present')

                outcome_labels = enforce_positives(outcomes.cpu().detach().numpy(), outcome_classes, 'Abnormal')
                outcome_prediction = enforce_positives(outcome_prediction.cpu().detach().numpy(), outcome_classes, 'Abnormal')
                

                murmur_weighted_acc = compute_weighted_accuracy(murmur_labels, murmur_prediction, murmur_classes)
                outcome_weighted_acc = compute_weighted_accuracy(outcome_labels, outcome_prediction, outcome_classes)
                
                total_prediction += murmur_prediction.shape[0]

                tqdm.write(f'Murmur - Acc:{murmur_acc}, Weighted Acc:{murmur_weighted_acc}, F1:{murmur_f1}\Outcome - Acc:{outcome_acc}, Weighted Acc:{outcome_weighted_acc}, F1:{outcome_f1}')

                # print(f'Prediction: {murmur_prediction} - {outcome_prediction}')
                # print(f'{correct_murmur_prediction} / {total_prediction}')

                if i % 10 == 0:    # print every 10 mini-batches
                    tqdm.write(f'[{epoch + 1}, %{i + 1}] loss: {round(running_loss / 10, 3)}')
    
        # Print stats at the end of the epoch
        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        murmur_acc = correct_murmur_prediction/total_prediction
        outcome_acc = correct_outcome_prediction/total_prediction
        tqdm.write(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Murmur accuracy: {murmur_acc:.2f}, Outcome accuracy: {murmur_acc:.2f}')

        torch.save(model.state_dict(), f'{model_path}_{epoch}.sav')
        torch.save(model.state_dict(), last_model_path)

    tqdm.write('Finished Training')

def inference (model, val_dl, output_files):
    # Disable gradient updates
    with torch.no_grad():
        for data in val_dl:
            # Get the input features and target labels, and put them on the GPU
            patient_id, audio_embeddings, clinical_features, murmurs, outcomes = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device)

            # forward + backward + optimize
            murmur_outputs, outcome_outputs = model(audio_embeddings, clinical_features)

            # Get the predicted class with the hig# Get the predicted class with the highest score
            _, murmur_prediction = torch.max(murmur_outputs,1)
            _, outcome_prediction = torch.max(outcome_outputs,1)

            murmur_labels = np.zeros(len(murmur_classes), dtype=np.int_)
            murmur_labels[murmur_prediction[0]] = 1
            outcome_labels = np.zeros(len(outcome_classes), dtype=np.int_)
            outcome_labels[outcome_prediction[0]] = 1

            # Concatenate classes, labels, and probabilities.
            classes = murmur_classes + outcome_classes
            labels = np.concatenate((murmur_labels, outcome_labels))
            murmur_probabilities = murmur_outputs[0].cpu().detach().numpy()
            outcome_probabilities = outcome_outputs[0].cpu().detach().numpy()
            probabilities = np.concatenate((murmur_probabilities, outcome_probabilities))

            cur_output_file = output_files[int(patient_id[0])]

            save_challenge_outputs(cur_output_file, patient_id, classes, labels, probabilities)