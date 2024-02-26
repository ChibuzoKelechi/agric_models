import streamlit as st
import numpy as np
import joblib


# Model loading and config
crop_model = joblib.load('agriculture/model.pkl')

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
    page_title='Crop recommender app'
)

st.header('Crop recommendation with ML')
st.markdown('''
            This machine learning model/app was trained on **agricultural data**, in order to recommend crops to be cultivated,
            with respect to the environmental/soil conditions. The included crops are **rice, Soyabeans, banana beans, cowpeas, orange, maize, coffee, peas, groundnuts, mango, grapes, watermelon, apple,cotton**
''')

st.subheader('Model UI')


N = st.slider(label=' nitrogen value', min_value=50, max_value=200)
P = st.slider(label=' Phosphorus value', min_value=50, max_value=200)
K = st.slider(label=' Potassium value', min_value=50, max_value=200)
temperature =  st.slider(label=' temperature level', min_value=15, max_value=30)
humidity =  st.slider(label=' Humidity level', min_value=15, max_value=30)
ph =  st.slider(label=' ph level', min_value=1, max_value=14)
rainfall =  st.slider(label=' rainfall level', min_value=50, max_value=300)

instance = [[N, P, K, temperature, humidity, ph, rainfall]]


def make_predictions(instance_data):
    predictions = crop_model.predict_proba(instance_data)
    pred_class = np.argmax(predictions)
    certainty = 100 * np.max(predictions)    
    
    return pred_class, certainty

if st.button('Suggest crop'):
    predicted_class, certainty = make_predictions(instance)
    
    st.markdown(f'''
            <div style="background-color: black; color: white; font-weight: normal; padding: 1rem; border-radius: 10px;">
            <h4>Results</h4>
             Suggested crop => <br><span style="font-weight: bold;">{crops[predicted_class]} </span> <br> 
             <span style="font-weight: bold;">Certainty: {certainty:.2f}% </span>
            </div>
    ''', unsafe_allow_html=True)