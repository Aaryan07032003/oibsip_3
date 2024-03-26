import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import streamlit as st
from io import StringIO

@st.cache
def load_data(file_path):
    return pd.read_csv(file_path)

file_path = 'car data.csv'
car_data = load_data(file_path)

st.title("CAR PRICE PREDICTION ANALYSIS")

st.header("Basic Information")
st.write(car_data.head())

st.header("Summary Statistics")
st.write(car_data.describe())

st.header("Missing Values")
st.write(car_data.isnull().sum())

st.header("Pairplot")
sns.pairplot(car_data)
plt.title("Pairplot")
st.pyplot()

st.header("Box Plot")
plt.figure(figsize=(10, 6))
sns.boxplot(data=car_data.drop(['Car_Name', 'Fuel_Type', 'Selling_type', 'Transmission'], axis=1))
plt.title("Box Plot")
st.pyplot()

# Preprocessing
car_data.fillna(car_data.mean(), inplace=True)

onehot_transformer = ColumnTransformer(
    transformers=[('onehot', OneHotEncoder(), ['Fuel_Type', 'Selling_type', 'Transmission'])],
    remainder='passthrough'
)
X_encoded = onehot_transformer.fit_transform(car_data.drop(['Car_Name', 'Selling_Price'], axis=1))
y = car_data['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.header("Model Evaluation")
st.write("Mean Squared Error:", mse)
st.write("R-squared:", r2)
