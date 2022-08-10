import sys
sys.path.append("../python-classifier-2022")

from os.path import join
from os import makedirs
from helper_code import *
from team_code_v2 import run_challenge_model
import streamlit as st
import pandas as pd
import joblib
from st_aggrid import AgGrid, GridOptionsBuilder


DATA_FOLDER = "C:/Users/lumin/Desktop/Work/20212/source-code/streamlit/patients_data"
OUTPUT_FOLDER = "C:/Users/lumin/Desktop/Work/20212/source-code/streamlit/outputs"
MODEL_FOLDER = "C:/Users/lumin/Desktop/Work/20212/source-code/python-classifier-2022/models"

auscultation_loc = {
    'PV': 'Pulmonary Valve',
    'TV': 'Tricuspid Valve',
    'AV': 'Aortic Valve',
    'MV': 'Mitral Valve',
    'Phc': 'Any other Auscultation Location'
}

results = ["Present", "Unknown", "Absent", "Abnormal", "Normal"]

st.title("Patients Database")

def get_display_features(patient_id, data, recordings):
    # Extract the age group and replace with the (approximate) number of months for the middle of the age group.
    age_group = get_age(data)

    if compare_strings(age_group, 'Neonate'):
        age = 0.5
    elif compare_strings(age_group, 'Infant'):
        age = 6
    elif compare_strings(age_group, 'Child'):
        age = 6 * 12
    elif compare_strings(age_group, 'Adolescent'):
        age = 15 * 12
    elif compare_strings(age_group, 'Young Adult'):
        age = 20 * 12
    else:
        age = float('nan')

    # Extract sex. Use one-hot encoding.
    sex = get_sex(data)

    # Extract height and weight.
    height = get_height(data)
    weight = get_weight(data)

    # Extract pregnancy status.
    is_pregnant = get_pregnancy_status(data)

    # Extract recording locations
    locations = get_locations(data)
    locations = ', '.join(locations)

    features = {
        'Name' : patient_id,
        'Age ' : age,
        'Sex' : sex,
        'Height' : height,
        'Weight' : weight,
        'Is Pregnant' : is_pregnant,
        'Recording Locations' : locations
    }
    # features_name = ['Age', 'Sex', 'Height', 'Weight', 'Is Pregnant']
    # features = pd.DataFrame([age, sex, height, weight, is_pregnant], columns=features_name)

    return features

def load_challenge_model(model_folder):
    filename = os.path.join(model_folder, 'model_randomforest.sav')
    return joblib.load(filename)

def get_prediction(patient_file, allow_failures):
    # Load model
    model = load_challenge_model(MODEL_FOLDER)

    # Create a folder for the Challenge outputs if it does not already exist
    makedirs(OUTPUT_FOLDER, exist_ok=True)

    patient_data = load_patient_data(patient_file)
    recordings = load_recordings(DATA_FOLDER, patient_data, get_paths=False)

    # Allow or disallow the model to fail on parts of the data; helpful for debugging.
    try:
        classes, labels, probabilities = run_challenge_model(model, patient_data, recordings, 1) 
    except Exception as e:
        st.write(e)
        if allow_failures:
            classes, labels, probabilities = list(), list(), list()
        else:
            raise

    # Save Challenge outputs.
    head, tail = os.path.split(patient_file)
    root, extension = os.path.splitext(tail)
    output_file = os.path.join(OUTPUT_FOLDER, root + '.csv')
    patient_id = get_patient_id(patient_data)
    save_challenge_outputs(output_file, patient_id, classes, labels, probabilities)

def main():
    # Extract patients features
    patient_files = find_patient_files(DATA_FOLDER)
    num_patient_files = len(patient_files)

    features = list()

    # st.write(num_patient_files)
    for i in range(num_patient_files):
        # Load the current patient data and recordings.
        current_patient_data = load_patient_data(patient_files[i])
        current_recordings = load_recordings(DATA_FOLDER, current_patient_data)

        # Extract features.
        patient_id = patient_files[i].split('\\')[-1].replace('.txt', '')
        current_features = get_display_features(patient_id, current_patient_data, current_recordings)
        features.append(current_features)

    features = pd.DataFrame(features)

    gb = GridOptionsBuilder.from_dataframe(features)
    gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
    gb.configure_side_bar() #Add a sidebar
    gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
    gridOptions = gb.build()

    grid_response = AgGrid(
        features,
        gridOptions=gridOptions,
        data_return_mode='AS_INPUT', 
        update_mode='MODEL_CHANGED', 
        fit_columns_on_grid_load=False,
        theme='blue', #Add theme color to the table
        enable_enterprise_modules=True,
        height=350, 
        width='100%',
        reload_data=True,
    )
    
    # new_df = grid_response['data']
    selected = grid_response['selected_rows'] 
    selected_df = pd.DataFrame(selected) #Pass the selected rows to a new dataframe df

    # Display the selected patient
    if not selected_df.empty:
        st.header('Detailed Patient Info')
        selected_patient = selected_df.to_dict('list')
        
        # Display info
        for k, v in selected_patient.items():
            strip_items = ["'", '"', '[', ']']
            v_str = str(v)
            for i in strip_items:
                v_str = v_str.replace(i, '')

            if k == 'Name':
                selected_patient_id = v_str

            if k == 'Recording Locations':
                recording_locs = [l.strip() for l in v_str.split(',')]

            st.write(f'{k}: {v_str}')

        # Display recordings
        st.subheader('PCG Recordings')
        
        for loc in recording_locs:
            st.write(auscultation_loc[loc])
            recording_file = join(DATA_FOLDER, f"{selected_patient_id}_{loc}.wav")
            st.audio(recording_file, format="audio/wav", start_time=0)

        # Run model on selected patient and display results
        if st.button('Check for Murmur'):
            patient_file = join(DATA_FOLDER, f"{selected_patient_id}.txt")
            get_prediction(patient_file, allow_failures=True)
            st.write("Predicted outcome: ")

            # Load and display predicted outcome
            result_file = join(OUTPUT_FOLDER, f"{selected_patient_id}.csv")
            result_df = pd.read_csv(result_file, skiprows=1)
            
            st.write(result_df)

            for i, j in enumerate(list(result_df.loc[0, :])):
                if (j) == 1:
                    if i < 3:
                        st.write(f"Murmur Status: {results[i]}")
                    else:
                        st.write(f"Outcome Status: {results[i]}")
                    
            

main()