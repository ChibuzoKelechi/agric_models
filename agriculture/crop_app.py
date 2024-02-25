import streamlit as st
import xgboost as xg
from xgboost import Booster
import numpy as np

xgbooster = Booster()


# Model loading and config
crop_model = xg.XGBModel.load_model('model.json')

crops = {
    0: 'rice',
    1: 'Soyabeans',
    2: 'banana',
    3: 'beans',
    4: 'cowpeas',
    5: 'cowpeas',
    6: 'maize',
    7: 'coffee',
    8: 'peas',
    9: 'groundnuts',
    10: 'mango',
    11: 'grapes',
    12: 'watermelon',
    13: 'apple',
    14: 'cotton'
}

# Streamlit app 
st.set_page_config(
    page_title='Agricultural suggestion app'
)

st.header('Crop suggestion with ML')
st.markdown('''
            This machine learning model/app was trained on **agricultural data**, in order to suggest crops to be cultivated,
            based on environmental/soil conditions. The included crops are **rice, Soyabeans, banana beans, cowpeas, orange, maize, coffee, peas, groundnuts, mango, grapes, watermelon, apple,cotton**
''')

st.subheader('Model UI')

nitrogen_level = st.slider(label='Select nitrogen value', min_value=50, max_value=200)
phosphor_level = st.slider(label='Select Phosphorus value', min_value=50, max_value=200)
potassium_level = st.slider(label='Select Potassium value', min_value=50, max_value=200)
temperature =  st.slider(label='Select temperature level', min_value=15, max_value=30)
humidity =  st.slider(label='Select Humidity level', min_value=15, max_value=30)
ph =  st.slider(label='Select ph level', min_value=1, max_value=14)
rainfall =  st.slider(label='Select rainfall level', min_value=50, max_value=300)

data = [nitrogen_level, phosphor_level, potassium_level, temperature, humidity, ph, rainfall]
instance = xg.DMatrix(data)

def make_predictions(instance_data):
    predictions = crop_model.predict_proba(instance_data)
    pred_class = np.argmax(predictions)
    certainty = 100 * np.max(predictions)    
    
    return pred_class, certainty

if st.button('Suggest crop'):
    predicted_class, certainty = make_predictions(instance)
    
    st.markdown(f'''
            <div style="background-color: black; color: white; font-weight: bold; padding: 1rem; border-radius: 10px;">
            <h4>Results</h4>
             Suggested crop => <span style="font-weight: bold;">{crops[predicted_class]} </span> with <span style="font-weight: bold;">{certainty:.2f}% </span>certainty
            </div>
    ''', unsafe_allow_html=True)
    
