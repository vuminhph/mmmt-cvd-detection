import streamlit as st
from PIL import Image

st.title("Heart Murmur Detection")

st.caption("<Introduction to Murmur related cardiac diseases and PCG murmur screening>")

heart_img = Image.open("C:/Users/lumin/Desktop/Work/20212/Documents/Figures/heart.png")

st.image(heart_img)

st.header("Introduction")

st.write('''
According to WHO, Cardiovascular Diseases (CVD's) continue to be one of the leading causes of deaths globally.\n
To check for any CVD's (abnormalities) in patients' heartbeat sounds, medical practitioners currently use a method known as cardiac auscultation.
This is a process whereby a medical practitioner listens to the heart sound, analyses it and classifies it as normal or abnormal.
Generally it is a difficult skill to acquire considering the complexity of abnormal heart sounds.\n
This is an easily accessible and reliable heartbeat sound classification system that aims to aid the screening and early detection of CVDs.
''')
# upload_info_btn = st.button('Upload new patient info')
# if upload_info_btn:
#     upload_info_btn.disabled = True
#     st.text_input("Name:", key="name")

#     number = st.number_input('Age', 0, 120, 18, 1)