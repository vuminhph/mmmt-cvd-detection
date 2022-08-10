import os, numpy as np
import torchaudio
import torch
from torch.nn import ZeroPad2d
from tqdm import tqdm
from torchvggish import vggish, vggish_input
import pickle
import shutil

from Const import *
from helper_code import *
from audio_util import AudioUtil
from extract_features_utils import get_clinical_features


# Initialise model and download weights
embedding_model = vggish()

data_folder = "/media/data/HeartMurmurDetection/physionet.org/files/circor-heart-sound/1.0.3/training_data"
# data_folder = "/media/data/minhpv/circor-heart-sound/final"
# data_folder = "C:/Users/lumin/Desktop/Work/20212/Data/circor-heart-sound/final/test"
recording_locations = ['AV', 'MV', 'PV', 'TV', 'PhC']
patient_files = find_patient_files(data_folder)

if os.path.exists(features_file_path):
    shutil.copy(features_file_path, features_file_path_bak)
    os.remove(features_file_path)
    tqdm.write('Deleted existed cycle features')

for patient in tqdm(patient_files):
    current_patient_data = load_patient_data(patient)
    current_patient_id = get_patient_id(current_patient_data)
    current_recordings_paths, current_recordings = load_recordings(data_folder, current_patient_data, get_paths=True, preprocess=True)
    current_locations = get_locations(current_patient_data)

    # Get audio_embeddings
    cur_patient_cycles_embeddings = torch.zeros((len(recording_locations), 128 * EMBEDDING_ROWS))

    for i in range(len(current_locations)):
            for j in range(len(recording_locations)):
                if compare_strings(current_locations[i], recording_locations[j]) and np.size(current_recordings[i])>0:
                    segmentation_file_path = AudioUtil.get_segmentation_file(current_recordings_paths[i])
                    cardiac_states = AudioUtil.get_cardiac_states(segmentation_file_path)

                    if cardiac_states is None:
                        continue

                    cardiac_cycles = AudioUtil.split_cardiac_cycles(current_recordings[i], cardiac_states)
                    
                    loc_embeddings = None

                    for cycle in cardiac_cycles:
                        cur_cycle = AudioUtil.audio_norm(cycle)
                        cur_cycle = AudioUtil.pad_signal(cur_cycle, MAX_DURATION)

                        example = vggish_input.waveform_to_examples(data=cur_cycle, sample_rate=NEW_SAMPLING_RATE)
                        embeddings = embedding_model.forward(example)
                        
                        loc_embeddings = loc_embeddings = embeddings if loc_embeddings is None else torch.vstack((loc_embeddings, embeddings))

                # Pad embeddings to reach size of (EMBEDDING_ROWS, 128)
                cur_no_rows = loc_embeddings.size()[0]
                pad = ZeroPad2d((0, 0, 0, EMBEDDING_ROWS - cur_no_rows))
                loc_embeddings = pad(loc_embeddings)

                cur_patient_cycles_embeddings[j] = torch.flatten(loc_embeddings)

    cur_patient_cycles_embeddings = torch.flatten(cur_patient_cycles_embeddings)

    # Get clinical features
    cur_clinical_features = get_clinical_features(current_patient_data)

    # Extract labels and assign classes
    current_murmur = murmur_IDs[get_murmur(current_patient_data)]
    current_outcome = outcome_IDs[get_outcome(current_patient_data)]

    # Save the extracted features
    if os.path.exists(features_file_path):
        shutil.copy(features_file_path, features_file_path_bak)
        
        with open(features_file_path_bak, 'rb') as f:
            hist_recording_cycles_embeddings = pickle.load(f)
            
            with open(features_file_path, 'wb') as fp:
                new_data = {
                            'patient_ids' : hist_recording_cycles_embeddings['patient_ids'] + [current_patient_id],
                            'audio_embeddings' : torch.vstack((hist_recording_cycles_embeddings['audio_embeddings'], cur_patient_cycles_embeddings)),
                            'clinical_features' : torch.vstack((hist_recording_cycles_embeddings['clinical_features'], cur_clinical_features)),
                            'murmurs' : hist_recording_cycles_embeddings['murmurs'] + [current_murmur],
                            'outcomes' : hist_recording_cycles_embeddings['outcomes'] + [current_outcome]
                }
                
                tqdm.write(str(new_data['audio_embeddings'].size()))

                pickle.dump(new_data, fp)

    else:
        with open(features_file_path, 'wb+') as fp:
            data = {
                    'patient_ids' : [current_patient_id],
                    'audio_embeddings' : cur_patient_cycles_embeddings,
                    'clinical_features' : cur_clinical_features,
                    'murmurs' : [current_murmur],
                    'outcomes' : [current_outcome]
            }

            pickle.dump(data, fp)
