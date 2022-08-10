from tqdm.notebook import tqdm
import os
import sys
import joblib
import pickle

from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from concurrent.futures import ProcessPoolExecutor, as_completed

from Const import *
from helper_code import *
from extract_features_utils import *

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
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

    murmur_classes = ['Present', 'Unknown', 'Absent']
    num_murmur_classes = len(murmur_classes)
    outcome_classes = ['Abnormal', 'Normal']
    num_outcome_classes = len(outcome_classes)

    features = list()
    murmurs = list()
    outcomes = list()

    # Create a executor with 4 workers
    executor = ProcessPoolExecutor(max_workers=4)
    inputs = []

    for i in tqdm(range(num_patient_files), desc="Patients", position=0):
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patient_files))

        # Load the current patient data and recordings.
        current_patient_data = load_patient_data(patient_files[i])
        current_recordings_paths, current_recordings = load_recordings(data_folder, current_patient_data, get_paths=True)

        # Extract features.
        current_features = get_full_features(current_patient_data, current_recordings, current_recordings_paths)
        features.append(current_features)

        inputs.append((current_patient_data, current_recordings, current_recordings_paths))
        
        
        # Extract labels and assign classes
        current_murmur = get_murmur(current_patient_data)
        murmurs.append(current_murmur)

        current_outcome = get_outcome(current_patient_data)
        outcomes.append(current_outcome)

    futures = [executor.submit(get_full_features, input[0], input[1], input[2]) for input in inputs]
    for future in as_completed(futures):
        current_features = future.result()
        features.append(current_features)

    features = np.vstack(features)
    murmurs = np.array(murmurs)
    outcomes = np.array(outcomes)


    # Train the model.
    if verbose >= 1:
        print('Training model...')

    # Define parameters
    random_state   = 6789 # Random state; set for reproducibility.

    imputer = SimpleImputer().fit(features)
    features = imputer.transform(features)
    murmur_classifier = AdaBoostClassifier(n_estimators=625, learning_rate=1.01, algorithm='SAMME', random_state=random_state).fit(features, murmurs)
    outcome_classifier = AdaBoostClassifier(n_estimators=625, learning_rate=1.02, algorithm='SAMME', random_state=random_state).fit(features, outcomes)

    # Save the model.
    save_challenge_model(model_folder, imputer, murmur_classes, murmur_classifier, outcome_classes, outcome_classifier)

    if verbose >= 1:
        print('Done.')

# Optimize the model
def optimize_model(data_folder, model_folder, verbose):
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

    murmur_classes = ['Present', 'Unknown', 'Absent']
    num_murmur_classes = len(murmur_classes)
    outcome_classes = ['Abnormal', 'Normal']
    num_outcome_classes = len(outcome_classes)

    features = list()
    murmurs = list()
    outcomes = list()

    # Create a executor with 4 workers
    executor = ProcessPoolExecutor(max_workers=4)
    inputs = []

    for i in tqdm(range(num_patient_files), desc="Patients", position=0):
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patient_files))

        # Load the current patient data and recordings.
        current_patient_data = load_patient_data(patient_files[i])
        current_recordings_paths, current_recordings = load_recordings(data_folder, current_patient_data, get_paths=True)

        # Extract features.
        # current_features = get_full_features(current_patient_data, current_recordings, current_recordings_paths)
        # features.append(current_features)

        inputs.append((current_patient_data, current_recordings, current_recordings_paths))
        
        
        # Extract labels and assign classes
        current_murmur = get_murmur(current_patient_data)
        murmurs.append(current_murmur)

        current_outcome = get_outcome(current_patient_data)
        outcomes.append(current_outcome)

    futures = [executor.submit(get_full_features, input[0], input[1], input[2]) for input in inputs]
    for future in as_completed(futures):
        current_features = future.result()
        features.append(current_features)

    features = np.vstack(features)
    murmurs = np.array(murmurs)
    outcomes = np.array(outcomes)


    # Train the model.
    if verbose >= 1:
        print('Training model...')

    imputer = SimpleImputer().fit(features)
    features = imputer.transform(features)
    murmur_classifier = AdaBoostClassifier(random_state=RANDOM_STATE)
    outcome_classifier = AdaBoostClassifier(random_state=RANDOM_STATE)

    # Define parameters for tuning
    parameters = {
        'n_estimators' : list(range(25, 201, 25)),
        'learning_rate' : [(0.97 + x / 100) for x in range(0, 8)],
        'algorithm' : ['SAMME', 'SAMME.R']
    }

    murmur_clf = GridSearchCV(murmur_classifier, parameters, scoring='balanced_accuracy', cv=5, verbose=verbose, n_jobs=4)
    outcome_clf = GridSearchCV(outcome_classifier, parameters, scoring='balanced_accuracy', cv=5, verbose=verbose, n_jobs=4)

    murmur_clf.fit(features, murmurs)
    outcome_clf.fit(features, outcomes)

    # Save the Grid Search result
    with open('adaboost.pickle', 'wb') as fp:
        d = {'imputer': imputer, 'murmur_classes': murmur_classes, 'murmur_classifier': murmur_clf, 'outcome_classes': outcome_classes, 'outcome_classifier': outcome_clf}
        pickle.dump(d, fp)
    
    print(f"Murmur best estimator: {murmur_clf.best_estimator_}, best_score: {murmur_clf.best_score_}")
    print(f"Outcome best estimator: {outcome_clf.best_estimator_}, best scores: {outcome_clf.best_score_}")

    if verbose >= 1:
        print('Done.')

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_model(model, data, recordings, recording_file_paths, verbose=1):
    imputer = model['imputer']
    murmur_classes = model['murmur_classes']
    murmur_classifier = model['murmur_classifier']
    outcome_classes = model['outcome_classes']
    outcome_classifier = model['outcome_classifier']

    # Load features.
    features = get_full_features(data, recordings, recording_file_paths)

    # Impute missing data.
    features = features.reshape(1, -1)
    features = imputer.transform(features)

    # Get classifier probabilities.
    murmur_probabilities = murmur_classifier.predict_proba(features)[0]
    murmur_probabilities = np.asarray(murmur_probabilities, dtype=np.float32)
    outcome_probabilities = outcome_classifier.predict_proba(features)[0]
    outcome_probabilities = np.asarray(outcome_probabilities, dtype=np.float32)

    # Choose label with highest probability.
    murmur_labels = np.zeros(len(murmur_classes), dtype=np.int_)
    idx = np.argmax(murmur_probabilities)
    murmur_labels[idx] = 1
    outcome_labels = np.zeros(len(outcome_classes), dtype=np.int_)
    idx = np.argmax(outcome_probabilities)
    outcome_labels[idx] = 1

    # Concatenate classes, labels, and probabilities.
    classes = murmur_classes + outcome_classes
    labels = np.concatenate((murmur_labels, outcome_labels))
    probabilities = np.concatenate((murmur_probabilities, outcome_probabilities))

    return classes, labels, probabilities

# Load your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_model(model_folder, verbose):
    filename = os.path.join(model_folder, 'model_adaboost.sav')
    return joblib.load(filename)

# Save your trained model.
def save_challenge_model(model_folder, imputer, murmur_classes, murmur_classifier, outcome_classes, outcome_classifier):
    d = {'imputer': imputer, 'murmur_classes': murmur_classes, 'murmur_classifier': murmur_classifier, 'outcome_classes': outcome_classes, 'outcome_classifier': outcome_classifier}
    filename = os.path.join(model_folder, 'model_adaboost.sav')
    joblib.dump(d, filename, protocol=0)