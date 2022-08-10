import os
import streamlit as st
import pandas

DATA_PATH = "C:/Users/lumin/Desktop/Work/20212/source-code/streamlit/patients_data"
FREQUENCY = 4000

st.title("Upload new patient's info")

# Input fields for new patient
# Name
name = st.text_input("Name:").strip()
# Age
age = st.number_input("Age (months old):", 0, 12 * 21, value=6, step=1)
# Sex
sex = st.selectbox("Sex", ["Male", "Female"])
# Height
height = st.number_input("Height (in centimeters)", 0.0, 250.0, value=120.0, step=0.1, format="%.1f")
# Weight
weight = st.number_input("Weight (in kilograms):", 0.0, value=50.0, step=0.1, format="%.1f")
# Pregnancy status"
preg_stat = st.selectbox("Pregnancy status", [False, True])
if sex != "Female" and preg_stat:
    st.markdown("*Not a valid input*")

recordings = st.file_uploader("PCG recordings", accept_multiple_files=True)
num_of_recordings = 0
for recording in recordings:
     bytes_data = recording.read()
     st.markdown(f"*{recording.name} uploaded successfully*")
     num_of_recordings += 1


# Upload patient's info
if st.button("Upload patient's info"):
    # Save patient's info as file
    if name == "":
        st.write("Please enter patient's name")

    else:
        name = '_'.join(name.split(' '))
        new_file = os.path.join(DATA_PATH, name + '.txt')
        if os.path.exists(new_file): # Remove file if already exists
            os.remove(new_file)

        with open(new_file, 'w') as f:
            f.write(f"{name} {num_of_recordings} {FREQUENCY}\n")
            
            # Save recordings
            for recording in recordings:
                position = recording.name.split('.')[0].split('_')[-1]
                new_recording_name = f'{name}_{position}.wav'
                with open(os.path.join(DATA_PATH, new_recording_name),"wb") as recording_f:
                    recording_f.write(recording.getbuffer())                
                
                f.write(f"{position} -1 {new_recording_name} -1\n")

            # Save age 
            if age <= 1:
                age_str = 'Neonate'
            elif age <= 12:
                age_str = 'Infant'
            elif age <= 11 * 12:
                age_str = 'Child'
            elif age_str <= 18 * 12:
                age_str = 'Adolescent'
            elif age_str <= 21 * 12:
                age_str = 'Young Adult'
            f.write(f"#Age: {age_str}\n")

            # Save Sex
            f.write(f"#Sex: {sex}\n")
            # Save height
            f.write(f"#Height: {height}\n")
            # Save weight
            f.write(f"#Weight: {weight}\n")
            # Save Pregnancy status
            f.write(f"#Pregnancy status: {preg_stat}\n")

            

        st.markdown("*Patient's info uploaded*")
