import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pickle

# Load the pre-trained machine learning model
with open('linear_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the fitted One-Hot Encoder
with open('preprocessing_and_model.pkl', 'rb') as encoder_file:
    encoder = pickle.load(encoder_file)


# Define a prediction function
def predict(transformed_data):

    # Make predictions using the pre-trained model
    predictions = model.predict(transformed_data)

    return predictions


# Streamlit UI
st.title("Model Prediction App")

# Input fields for user
st.subheader("User Input")

gender = st.selectbox("Gender", ['female', 'male'])
race_ethnicity = st.selectbox("Race/Ethnicity", ['group B', 'group C', 'group A', 'group D', 'group E'])
parental_education = st.selectbox("Parental Level of Education",
                                  ["bachelor's degree", 'some college', "master's degree", "associate's degree",
                                   'high school', 'some high school'])
lunch = st.selectbox("Lunch", ['standard', 'free/reduced'])
test_preparation = st.selectbox("Test Preparation Course", ['none', 'completed'])
reading_score = st.number_input("Reading Score", min_value=0, max_value=100)
writing_score = st.number_input("Writing Score", min_value=0, max_value=100)

user_input = {
    'gender': gender,
    'race_ethnicity': race_ethnicity,
    'parental_level_of_education': parental_education,
    'lunch': lunch,
    'test_preparation_course': test_preparation,
    'reading_score': reading_score,
    'writing_score': writing_score
}

columns = ["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course","reading_score","writing_score"]
df = pd.DataFrame(user_input, columns=columns, index=[0])
transformed_data = encoder['column_transformer'].transform(df)

if st.button("Predict"):
    result = predict(transformed_data)
    st.subheader("Prediction Result")
    st.write(f"The predicted Math Score is: {result[0]}")

# To run the app, use the following command in your terminal:
# streamlit run your_script.py
