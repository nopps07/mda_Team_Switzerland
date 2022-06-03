from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np

model_reg = load_model('ECONOMICS_regression_after2016')

def predict(model_reg, input_df):
    predictions_df = predict_model(estimator=model_reg, data=input_df)
    predictions = predictions_df['Label'][0]
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
    st.sidebar.success('https://eurovision.tv/')

    st.sidebar.image(image)

    #st.title("Eurovision Score prediction (ECONOMICS)")
    
    #add_selectbox == 'Scores (regression)':
    st.title('Regression')

    st.write('This is the `regression` page of the app')

    file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

    if file_upload is not None:
        data = pd.read_csv(file_upload)
        predictions = predict_model(estimator=model_reg, data=data)
        st.write(predictions)


if __name__ == '__main__':
    run()