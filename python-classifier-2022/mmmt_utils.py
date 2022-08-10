import os
import shutil
import pickle
import torch
from tqdm import tqdm

from Const import *

def low_var_filter(audio_embeddings):
    """
    Filter that reduce the number of dimension of audio embeddings by removing the columns that have low variance
    """
    no_rows, no_cols = audio_embeddings.size()

    indexes_to_keep = []

    for i in tqdm(range(no_cols), 'VGGish Embeddings'):
        col_var = torch.var(audio_embeddings[:, i])
        # tqdm.write(str(col_var))
        if col_var > VAR_THRESHOLD:
            indexes_to_keep.append(i)
            # tqdm.write(str(i))

    tqdm.write(f'Feature length: {len(indexes_to_keep)}')

    return indexes_to_keep

def apply_filter(audio_embeddings, indexes_to_keep):
    tqdm.write(f'{audio_embeddings.size()}')
    
    sample_nos = audio_embeddings.size()[0]

    filtered_embeddings = audio_embeddings[:, indexes_to_keep[0]]
    filtered_embeddings= torch.reshape(filtered_embeddings, (sample_nos, 1))
    
    for id in tqdm(indexes_to_keep[1:], desc='Indexes to keep'):
        aud_embedding = audio_embeddings[:, id]
        aud_embedding = torch.reshape(aud_embedding, (sample_nos, 1))

        filtered_embeddings = torch.hstack((filtered_embeddings, aud_embedding))

    tqdm.write(f'Filtered audio feature size: {filtered_embeddings.size()}')
    return filtered_embeddings

def get_sample(cur_patient_id, data):
    patient_ids = data['patient_ids']
    audio_embeddings = data['audio_embeddings']
    clinical_features = data['clinical_features']
    murmurs = data['murmurs']
    outcomes = data['outcomes']

    for i in range(len(patient_ids)):
        if cur_patient_id == patient_ids[i]:
            return (patient_ids[i], audio_embeddings[i, :], clinical_features[i, :], murmurs[i], outcomes[i])

def relabel(labels):
    new_labels = labels
    for i in range(len(new_labels)):
        new_labels[i] -= 1
    return new_labels

if __name__ == '__main__':
    shutil.copyfile(features_file_path, features_file_path + '.bak')
    if os.path.exists(filtered_features_file_path):
                shutil.copy(filtered_features_file_path, filtered_features_file_path + '.bak')
                os.remove(filtered_features_file_path)
                tqdm.write('Deleted existed cycle features')

    with open(features_file_path, 'rb') as f:
        data = pickle.load(f)
        audio_embeddings = torch.nn.functional.normalize(data['audio_embeddings'])
        indexes_to_keep = low_var_filter(audio_embeddings)
        filtered_aud_features = apply_filter(audio_embeddings, indexes_to_keep)

        # clinical_features = data['clinical_features']
        # clinical_features= torch.nan_to_num(clinical_features)
        # print(torch.isnan(clinical_features))
        # print(torch.isnan(data['clinical_features']))

        with open(filtered_features_file_path, 'wb+') as fp:
            new_data = {
                    'patient_ids' : data['patient_ids'],
                    'audio_embeddings' : filtered_aud_features,
                    'clinical_features' : torch.nn.functional.normalize(torch.nan_to_num(data['clinical_features'])),
                    # 'clinical_features' : data['clinical_features'],
                    'murmurs' : relabel(data['murmurs']),
                    'outcomes' : relabel(data['outcomes'])
            }

            pickle.dump(new_data, fp)