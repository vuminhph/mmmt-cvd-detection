import torch

# Preprocessing
ORIGINAL_SAMPLING_RATE = 4000
NEW_SAMPLING_RATE = 1000
HIGH_CUT_FREQ = 400
LOW_CUT_FREQ = 25

# Spike Removal
MAX_SWEEPS = 1000

# ML models
RANDOM_STATE = 6789

# Labels IDs
murmur_classes = ['Present', 'Unknown', 'Absent']
num_murmur_classes = len(murmur_classes)
murmur_IDs = {
    'Present' : 1,
    'Unknown' : 2,
    'Absent' : 3
}

outcome_classes = ['Abnormal', 'Normal']
num_outcome_classes = len(outcome_classes)
outcome_IDs = {
    'Abnormal' : 1,
    'Normal' : 2
}

# Low var filter
VAR_THRESHOLD = 0.000007
# VAR_THRESHOLD = 0.00001

# MMMT
MAX_DURATION = 3000 # in ms
MAX_CYCLE_NO = 93
EMBEDDING_ROWS = int(MAX_CYCLE_NO * MAX_DURATION / 1000)
LEARNING_RATE = 0.001
EPOCHS = 5
BATCH_SIZE = 16

# AUDIO_FEATURES_DIM = 4377
AUDIO_FEATURES_DIM = 12210
CLINICAL_FEATURES_DIM = 6

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Paths
data_folder = "/media/data/minhpv/circor-heart-sound/final/train"

features_file_path = './models/embeddings_178k.pickle'
features_file_path_bak = './models/embeddings_178k.pickle.bak'

aud_features_indexes_file_path = './models/audio_features.pickle'

filtered_features_file_path = './models/filtered_features.pickle'

model_path = './models/mmmt'
last_model_path = './models/mmmt_last.sav'

audio_only_model_path = './models/mmmt_aud_only'
last_audio_only_model_path = './models/mmmt_last_aud_only.sav'

cli_only_model_path = './models/mmmt_cli_only'
last_cli_only_model_path = './models/mmmt_last_cli_only.sav'
