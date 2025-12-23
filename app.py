import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

st.set_page_config(page_title="Flower AI", page_icon="🌸", layout="wide")

st.title("🌸 Flower Species Predictor AI")
st.write("Live prediction based on 4 flower species.")

# 1. Dataset for 4 species (Ensuring exactly 4 classes)
data_values = [
    [5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], # Setosa
    [7.0, 3.2, 4.7, 1.4], [6.4, 3.2, 4.5, 1.5], # Versicolor
    [6.3, 3.3, 6.0, 2.5], [5.8, 2.7, 5.1, 1.9], # Virginica
    [8.5, 4.2, 8.5, 3.0], [9.5, 4.8, 9.5, 3.8]  # Sunflower
]
target_values = [0, 0, 1, 1, 2, 2, 3, 3]
species_names = ['Setosa', 'Versicolor', 'Virginica', 'Sunflower']

# 2. Train Model
@st.cache_resource
def train_model():
    # n_estimators=100 daddi model eka hondata train wenawa
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(data_values, target_values)
    return clf

model = train_model()

# 3. Sidebar Sliders
st.sidebar.header('📍 Adjust Measurements')
sl = st.sidebar.slider('Sepal Length', 4.0, 10.0, 5.8)
sw = st.sidebar.slider('Sepal Width', 2.0, 5.0, 3.0)
pl = st.sidebar.slider('Petal Length', 1.0, 10.0, 4.3)
pw = st.sidebar.slider('Petal Width', 0.1, 4.0, 1.3)

# 4. Prediction
input_data = np.array([[sl, sw, pl, pw]])
prediction = model.predict(input_data)
result = species_names[prediction[0]]
probabilities = model.predict_proba(input_data)[0]

# Result display
st.success(f"### Current Prediction: **{result}**")

col1, col2 = st.columns([1, 1])

with col1:
    icons = {'Setosa': "💠", 'Versicolor': "🌀", 'Virginica': "⚛️", 'Sunflower': "🌻"}
    st.write(f"<h1 style='font-size: 100px; text-align: center;'>{icons[result]}</h1>", unsafe_allow_html=True)
    
    image_links = {
        'Setosa': "https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg/320px-Kosaciec_szczecinkowaty_Iris_setosa.jpg",
        'Versicolor': "https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Iris_versicolor_3.jpg/320px-Iris_versicolor_3.jpg",
        'Virginica': "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Iris_virginica.jpg/320px-Iris_virginica.jpg",
        'Sunflower': "https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/Sunflower_sky_backdrop.jpg/320px-Sunflower_sky_backdrop.jpg"
    }
    st.image(image_links[result], width=300)

with col2:
    st.write("### 📊 Confidence Levels")
    
    # මෙතන තමයි වැදගත්ම හරිය. 
    # නම් ගණනයි probability ගණනයි හරියටම 4ක් දැයි බලනවා.
    if len(species_names) == len(probabilities):
        prob_df = pd.DataFrame({
            'Species': species_names,
            'Probability (%)': [round(p * 100, 2) for p in probabilities]
        }).sort_values(by='Probability (%)', ascending=False)
        
        st.dataframe(prob_df, use_container_width=True, hide_index=True)
        
        fig = px.bar(prob_df, x='Species', y='Probability (%)', color='Species', 
                     text='Probability (%)', template="plotly_white")
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Data mismatch! Refreshing the app might help.")

st.write("---")
st.caption("Developed by Machan | Fixed Version")