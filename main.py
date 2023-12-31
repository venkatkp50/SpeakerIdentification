import streamlit as st
import numpy as np
import pandas as pd
import tempfile
import os
from PIL import Image
from config import IMAGE_PATH,FILE_TAXONOMY
from data import getSpeakerData, getFileList, getSpeakerLikelihood,populateGraph

banner_image = Image.open(IMAGE_PATH)
temp_dir = tempfile.TemporaryDirectory()
temp_dir_path = temp_dir.name

header = st.container()
with st.sidebar:
    st.title('Training Data')
    st.divider()
    x,y = populateGraph()
    # st.write(x)
    # st.write(y)
    dg = pd.DataFrame({'Speaker':x,'Test Files':y})
    st.bar_chart(data=dg,x='Speaker',y='Test Files')
    

with header:
    st.image(banner_image)  
    st.divider()
    st.header('Input')
    predictedSpeaker = ''
    speakerScore = ''
    fileOption = st.radio('Choose File to Test',['From Folder', 'From Data File'])
    
    if fileOption == 'From Data File':
        flag ='data_file'
        option1 = st.selectbox('Speaker',getSpeakerData())
        option2 = st.selectbox('Files',getFileList(option1))
        # st.write(FILE_TAXONOMY[option1])
        df = getSpeakerLikelihood(option2)
        speakerIdx = np.argmax(df['likelihood'])
        predictedSpeaker = df.loc[speakerIdx][0]
        speakerScore = df.loc[speakerIdx][1]
    else:
        flag = 'browse'
        uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=False,type=(["wav"]))
        if uploaded_files is not None:
            with open(os.path.join(temp_dir_path, uploaded_files.name), "wb") as f:
                audio_bytes = uploaded_files.read()
                f.write(audio_bytes)
            df = getSpeakerLikelihood(os.path.join(temp_dir_path, uploaded_files.name))
            speakerIdx = np.argmax(df['likelihood'])
            predictedSpeaker = df.loc[speakerIdx][0]
            speakerScore = df.loc[speakerIdx][1]
            st.audio(audio_bytes, format='audio/ogg')

    st.divider()
    st.header('Result')
    col1, col2,col3 = st.columns(3,gap='small')
    with col1:
        st.error('Model')
        st.markdown('GMM')
    with col2:
        st.error('Predicted Speaker')
        st.markdown(predictedSpeaker)
    with col3:
        st.error('Speaker Score')
        st.markdown(speakerScore)
        
