import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
data = pd.read_csv("Advertising.csv")

# Remove unnecessary column
data = data.drop("Unnamed: 0", axis=1)

# Display information
print("Column names:")
print(data.columns.tolist())

print("\nFirst 5 rows:")
print(data.head())

print("\nMissing values:")
print(data.isnull().sum())

print("\nBasic statistics:")
print(data.describe())

# TV Advertising vs Sales
plt.scatter(data["TV"], data["sales"])
plt.xlabel("TV Advertising")
plt.ylabel("Sales")
plt.title("TV Advertising vs Sales")
plt.show()

# Radio Advertising vs Sales
plt.scatter(data["radio"], data["sales"])
plt.xlabel("Radio Advertising")
plt.ylabel("Sales")
plt.title("Radio Advertising vs Sales")
plt.show()

# Newspaper Advertising vs Sales
plt.scatter(data["newspaper"], data["sales"])
plt.xlabel("Newspaper Advertising")
plt.ylabel("Sales")
plt.title("Newspaper Advertising vs Sales")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8, 5))

sns.heatmap(data.corr(), annot=True)

plt.title("Correlation Heatmap")
plt.show()

from sklearn.model_selection import train_test_split

# Features
X = data[["TV", "radio", "newspaper"]]

# Target
y = data["sales"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Display predictions
print("Predicted Sales:")
print(y_pred)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R² Score:", r2)

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = mean_squared_error(y_test, rf_pred) ** 0.5
rf_r2 = r2_score(y_test, rf_pred)

print("Random Forest MAE:", rf_mae)
print("Random Forest RMSE:", rf_rmse)
print("Random Forest R² Score:", rf_r2)

importance = rf_model.feature_importances_

print("Feature Importance:")
print("TV:", importance[0])
print("Radio:", importance[1])
print("Newspaper:", importance[2])

residuals = y_test - rf_pred

plt.scatter(rf_pred, residuals)
plt.axhline(y=0, color="red", linestyle="--")

plt.xlabel("Predicted Sales")
plt.ylabel("Residuals")
plt.title("Residual Plot - Random Forest")
plt.show()

# New advertising budget
new_data = [[150, 30, 20]]

# Predict sales
prediction = model.predict(new_data)

print("Predicted Sales:", prediction[0])

import joblib

# Save the trained model
joblib.dump(model, "sales_prediction_model.pkl")

print("Model saved successfully!")

# Step 16: Load the saved model

import joblib

# Load the saved model
loaded_model = joblib.load("sales_prediction_model.pkl")

# New advertising data
new_data = [[150, 30, 20]]

# Make prediction using the loaded model
prediction = loaded_model.predict(new_data)

print("Predicted Sales using loaded model:", prediction[0])

# Step 17: Interactive Sales Prediction

import joblib

# Load the trained model
loaded_model = joblib.load("sales_prediction_model.pkl")

# Get advertising amounts from the user
tv = float(input("Enter TV advertising budget: "))
radio = float(input("Enter Radio advertising budget: "))
newspaper = float(input("Enter Newspaper advertising budget: "))

# Create input data
new_data = [[tv, radio, newspaper]]

# Predict sales
prediction = loaded_model.predict(new_data)

print("Predicted Sales:", prediction[0])

# Step 18: Actual vs Predicted Sales Graph

import matplotlib.pyplot as plt

# Predict sales for the test data
y_pred = model.predict(X_test)

# Create the graph
plt.scatter(y_test, y_pred)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")

plt.show()

# Step 19: Final Model Results

print("\n===== SALES PREDICTION PROJECT RESULTS =====")

print("Linear Regression MAE:", mae)
print("Linear Regression RMSE:", rmse)

print("Random Forest MAE:", rf_mae)
print("Random Forest RMSE:", rf_rmse)

print("\nFeature Importance:")
print("TV:", importance[0])
print("Radio:", importance[1])
print("Newspaper:", importance[2])