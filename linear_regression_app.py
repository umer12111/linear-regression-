import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Streamlit App Title
st.title("Linear Regression with CSV Data")

# Step 1: Upload CSV File
st.sidebar.header("Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# Step 2: Allow user to specify libraries (optional)
# If necessary, you can provide a custom text area for users to input additional libraries or settings

# Step 3: Display CSV Data and Select Columns
if uploaded_file is not None:
    # Load the CSV data into a DataFrame
    df = pd.read_csv(uploaded_file)

    # Display the first few rows of the dataframe
    st.write("Dataset Preview:")
    st.write(df.head())

    # Step 4: Select the independent (X) and dependent (y) variables
    st.sidebar.header("Select Columns for Regression")

    # User selects which column to use as X (independent variable) and y (dependent variable)
    columns = df.columns.tolist()

    # Select the column for independent variable (X)
    X_column = st.sidebar.selectbox("Select independent variable (X)", columns)

    # Select the column for dependent variable (y)
    y_column = st.sidebar.selectbox("Select dependent variable (y)", columns)

    if X_column != y_column:
        # Extract the features (X) and target (y) variables
        X = df[[X_column]].values  # Independent variable
        y = df[y_column].values   # Dependent variable

        # Step 5: Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Step 6: Train the Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Step 7: Make Predictions
        y_pred = model.predict(X_test)

        # Step 8: Display Results
        st.write(f"### Model Results")
        st.write(f"**Mean Squared Error**: {mean_squared_error(y_test, y_pred)}")
        st.write(f"**R^2 Score**: {r2_score(y_test, y_pred)}")

        # Step 9: Plot the Regression Line
        st.write(f"### Regression Line Plot")
        plt.figure(figsize=(10, 6))
        plt.scatter(X_test, y_test, color='blue', label='Actual Data')
        plt.plot(X_test, y_pred, color='red', label='Regression Line')
        plt.xlabel(X_column)
        plt.ylabel(y_column)
        plt.title(f"Linear Regression: {X_column} vs {y_column}")
        plt.legend()
        st.pyplot(plt)

    else:
        st.error("Independent and Dependent variables cannot be the same!")
else:
    st.info("Please upload a CSV file to get started.")
