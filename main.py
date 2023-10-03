import streamlit as st
import numpy as np
import pandas as pd
import tempfile
import os

from config import IMAGE_PATH,FILE_TAXONOMY
from data import getSpeakerData, getFileList, getSpeakerLikelihood

temp_dir = tempfile.TemporaryDirectory()
temp_dir_path = temp_dir.name

header = st.container()
st.sidebar.title('Settings')
# print('1')
with header:
    # st.image(IMAGE_PATH)  
    st.divider()
    option1 = st.selectbox('Speaker',getSpeakerData())
    option2 = st.selectbox('Files',getFileList(option1))
    # st.write(FILE_TAXONOMY[option1])
    df = getSpeakerLikelihood(option2)
    speakerIdx = np.argmax(df['likelihood'])
    predictedSpeaker = df.loc[speakerIdx][0]

    col1, col2, col3, col4 = st.columns(4,gap='small')
    with col1:
        st.error('Model')
        st.markdown('GMM')
    with col2:
        st.error('Wave File')
        st.markdown(option2)
    with col3:
        st.error('Expected Speaker')
        st.markdown(option1)
    with col4:
        st.error('Predicted Speaker')
        st.markdown(predictedSpeaker)
   
    

