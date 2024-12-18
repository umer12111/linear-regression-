import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Function to perform Linear Regression
def perform_linear_regression(df, target_column):
    # Split the dataset into features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Linear Regression model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, X_test, y_test, y_pred, mse, r2

# Streamlit UI for file upload and other inputs
def main():
    st.title("Linear Regression with Streamlit")

    # File Upload
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        # Load the dataset
        df = pd.read_csv(uploaded_file)

        # Display the dataset
        st.subheader("Dataset Overview")
        st.write(df.head())

        # Select target column for regression
        target_column = st.selectbox("Select the target column", df.columns)

        # Perform Linear Regression
        if st.button("Run Linear Regression"):
            model, X_test, y_test, y_pred, mse, r2 = perform_linear_regression(df, target_column)

            # Display model performance
            st.subheader("Model Performance")
            st.write(f"Mean Squared Error (MSE): {mse:.2f}")
            st.write(f"R-squared: {r2:.2f}")

            # Plot the results
            st.subheader("Prediction vs Actual")
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test, y_pred, color='blue', label='Predictions')
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label="Ideal Fit")
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.title("Actual vs Predicted values")
            plt.legend()
            st.pyplot()

            # Show the regression coefficients
            st.subheader("Model Coefficients")
            st.write(f"Intercept: {model.intercept_}")
            st.write(f"Coefficients: {model.coef_}")

if __name__ == "__main__":
    main()
