import streamlit as st
from multiapp import MultiApp
from apps import regression, classification, clustering

app = MultiApp()

st.markdown("""
# Eurovision prediction models

You can predict the **Eurovision scores** with the economic factors
- **Regression**
- **Classification**

You can predict the **cluster of the Eurovision songs** with the song features
- **Clustering**
""")

app.add_app("regression", regression.app)
app.add_app("classification", classification.app)
app.add_app("clustering", clustering.app)
# the main app
app.run()