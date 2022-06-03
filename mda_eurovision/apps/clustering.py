from pycaret.clustering import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np

model_clu = load_model('SONGS_clustering')

def predict(model_clu, input_df):
    predictions_df = predict_model(estimator=model_clu, data=input_df)
    predictions = predictions_df['Cluster'][0]
    return predictions

def app():
    from PIL import Image
    image = Image.open('kul.png')
    #image_hospital = Image.open('hospital.jpg')

    #st.image(image, use_column_width=False)

    #add_selectbox = st.sidebar.selectbox(
    #    "What would you like to predict?",
    #    ("Scores (regression)"))


    st.sidebar.info('This app is created to predict Eurovision Score')

    Jury = st.sidebar.slider(label='Jury', min_value=1.0,
                             max_value=300.0,
                             value=1.0,
                             step=1.0)

    Public = st.sidebar.slider(label='Public', min_value=1.0,
                             max_value=300.0,
                             value=1.0,
                             step=1.0)

    Draw = st.sidebar.slider(label='Draw', min_value=1.0,
                             max_value=30.0,
                             value=1.0,
                             step=1.0)

    BPM = st.sidebar.slider(label='BPM', min_value=70.0,
                             max_value=140.0,
                             value=1.0,
                             step=1.0)

    features = {'Jury': Jury, 'Public': Public,
                'Draw': Draw, 'BPM': BPM
                }

    features_df = pd.DataFrame([features])

    st.table(features_df)

    if st.button('Predict'):
        prediction = predict_model(model_clu, features_df)

        st.write(' Based on feature values, your Eurovision score is ' + str(prediction))


    st.sidebar.success('https://eurovision.tv/')

    #st.sidebar.image(image)

    #st.title("Eurovision Score prediction (ECONOMICS)")
    
    #add_selectbox == 'Scores (regression)':
    st.title('Clustering')

    st.write('This is the `clustering` page of the app')

    file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

    if file_upload is not None:
        data = pd.read_csv(file_upload)
        predictions = predict_model(model_clu, data=data)
        st.write(predictions)


if __name__ == '__main__':
    run()