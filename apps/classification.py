from pycaret.classification import load_model, predict_model, dashboard
import streamlit as st
import pandas as pd
import numpy as np

model_cla = load_model('ECONOMICS_classification')
def predict_cla(model_cla, input_df):
    predictions_df = predict_model(estimator=model_cla, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def app():
    from PIL import Image
    image = Image.open('kul.png')
    #image_hospital = Image.open('hospital.jpg')

    #st.image(image, use_column_width=False)

    #add_selectbox = st.sidebar.selectbox(
    #    "What would you like to predict?",
    #    ("Scores (classification)"))

    #st.sidebar.info('This app is created to predict Eurovision Score')
    #st.sidebar.success('https://eurovision.tv/')

    st.sidebar.info('This app is created to predict Eurovision Score')
    st.sidebar.success('https://eurovision.tv/')

    st.sidebar.image(image)

    Log_GDP = st.sidebar.slider(label='logarithm_gdppercapita', min_value=3.460,
                                max_value=5.00,
                                value=0.001,
                                step=0.001)

    language = st.sidebar.selectbox("choose your language",
                                    ("Germanic", "Uralic", "Italic",
                                     "Celtic", "Hebrew", "Hellenic/Italic",
                                     "Afro-Asiatic", "Turkic", "Hellenic",
                                     "Balto-Slavic", "Albanian/Balto-Slavic", "Albanian",
                                     "Armenian", "Kartvelian")
                                    )

    pillar_ind = st.sidebar.slider(label='pillar_ind', min_value=1.0,
                                   max_value=3.0,
                                   value=1.0,
                                   step=1.0)
    politicalattitude = st.sidebar.slider(label='politicalattitude', min_value=0.0,
                                          max_value=2.0,
                                          value=1.0,
                                          step=1.0)
    youth_inrate = st.sidebar.slider(label='youth_inrate', min_value=0.00,
                                     max_value=0.050,
                                     value=0.001,
                                     step=0.001)
    culturegoodsexp_inrate = st.sidebar.slider(label='culturegoodsexp_inrate', min_value=0.00,
                                               max_value=0.050,
                                               value=0.001,
                                               step=0.001)

    # st.title("Eurovision Score prediction (ECONOMICS)")

    # add_selectbox == 'Scores (regression)':
    st.title('Classification')

    st.write('This is the `classification` page of the app')

    st.subheader('RANDOM VALUES')
    st.write('You can predict with random values from the sidebar')

    features = {'logarithm_gdppercapita': Log_GDP, 'language': language,
                'pillar_ind': pillar_ind, 'politicalattitude': politicalattitude,
                'youth_inrate': youth_inrate, 'culturegoodsexp_inrate': culturegoodsexp_inrate,
                }
    features_df = pd.DataFrame([features])
    check = predict_model(model_cla, features_df)
    st.write(check)
    st.write('Your predicted `POINTS_GRID` is', check.iloc[0]["Label"])


    st.subheader('TABLE')
    st.write('You can predict with the ready-to-import table')
    #dashboard(model_cla, display_format='streamlit')
    file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

    if file_upload is not None:
        data = pd.read_csv(file_upload)
        predictions = predict_model(estimator=model_cla, data=data)
        st.write(predictions)


if __name__ == '__main__':
    run()