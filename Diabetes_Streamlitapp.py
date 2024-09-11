import streamlit as st
from sklearn.model_selection import train_test_split
import pandas as pd
# import pickle
import joblib

model = joblib.load('K-Nearest Neighborsmodel.pkl')

with open('accuracy.txt', 'r') as file:
  accuracy = file.read()

st.title(f"Model Selection and Real-Time Prediction")
st.write(f"Model {accuracy}")

st.header("Real_Time Prediction")

test_data = pd.read_csv('diabetes.csv')

x_test = test_data.drop(columns='Outcome')
y_test = test_data['Outcome']

input_data = []
for col in x_test.columns:
  input_value = st.number_input(f"Input from {col}", value=0.0)
  input_data.append(input_value)

input_df = pd.DataFrame([input_data], columns = x_test.columns)

if st.button("Predict"):
  prediction = model.predict(input_df)
st.write(f"Accuracy variable content: {accuracy}")

if isinstance(accuracy, (int, float)):
    st.bar_chart([accuracy])
elif ': ' in accuracy:
    try:
        accuracy_value = float(accuracy.split(': ')[1])
        st.bar_chart([accuracy_value])
    except (ValueError, IndexError):
        st.write("Error parsing the accuracy value.")
else:
    st.write("Accuracy value is not in the expected format.")

