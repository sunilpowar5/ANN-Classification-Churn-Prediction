import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Load the model
model = tf.keras.models.load_model('model.h5')

# Load the pickle files
with open('gender_label_encoder.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('transformer.pkl', 'rb') as file:
    transformer = pickle.load(file)

# Access the OneHotEncoder for the Geography feature
encoder = transformer.transformers_[1][1]  # The second transformer is OneHotEncoder
geography_categories = encoder.categories_[0]  # Accessing the first category list for Geography

# Streamlit app
st.title('Customer Churn Prediction')

# Geography - Access the categories directly
geography = st.selectbox('Geography', geography_categories)

# Gender - Use label encoder to transform the gender selection
gender = st.selectbox('Gender', label_encoder_gender.classes_)

# Other features
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data with Geography column
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
    'Geography': [geography]  # Add the Geography column
})

# Apply the transformer to preprocess the input data
transformed_input = transformer.transform(input_data)

# Get the model prediction
prediction = model.predict(transformed_input)

# Get the churn probability (assuming it's a binary classification with a sigmoid activation)
prediction_prob = prediction[0][0]

# Display the result in the Streamlit app
st.write(f"Predicted Churn Probability: {prediction_prob:.4f}")

if prediction_prob > 0.5:
    st.write("The Customer is Likely to Churn.")
else:
    st.write("The Customer is NOT Likely to Churn.")
