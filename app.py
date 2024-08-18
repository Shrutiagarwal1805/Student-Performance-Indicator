import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import uniform
import traceback

# Load the data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('StudentsPerformance.csv')
        return df
    except FileNotFoundError:
        st.error("Error: 'StudentsPerformance.csv' file not found. Please ensure the file is in the same directory as this script.")
        return None

df = load_data()

if df is not None:
    # Prepare the data
    X = df.drop(columns=['math score'], axis=1)
    Y = df['math score']

    num_features = X.select_dtypes(exclude="object").columns
    cat_features = X.select_dtypes(include="object").columns

    try:
        # Updated OneHotEncoder initialization
        preprocessor = ColumnTransformer(
            [
                ("OneHotEncoder", OneHotEncoder(handle_unknown='ignore'), cat_features),
                ("StandardScaler", StandardScaler(), num_features),        
            ]
        )

        X = preprocessor.fit_transform(X)

        # Train the model with hyperparameter tuning
        @st.cache_resource
        def train_model_with_tuning(X, Y):
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
            
            # Define the hyperparameter space
            param_distributions = {
                'alpha': uniform(0, 10),
                'fit_intercept': [True, False],
                'tol': uniform(1e-5, 1e-3),
                'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
            }
            
            # Create a RandomizedSearchCV object
            random_search = RandomizedSearchCV(
                Ridge(),
                param_distributions=param_distributions,
                n_iter=100,
                cv=5,
                random_state=42,
                n_jobs=-1
            )
            
            # Fit the RandomizedSearchCV object to the data
            random_search.fit(X_train, Y_train)
            
            return random_search.best_estimator_

        # Train the model (this will use caching to avoid retraining on every rerun)
        model = train_model_with_tuning(X, Y)

        # Streamlit app
        st.title('Math Score Prediction')

        st.write('Enter the student information to predict their math score:')

        # Create input fields
        gender = st.selectbox('Gender', df['gender'].unique())
        race_ethnicity = st.selectbox('Race/Ethnicity', df['race/ethnicity'].unique())
        parental_education = st.selectbox('Parental Level of Education', df['parental level of education'].unique())
        lunch = st.selectbox('Lunch', df['lunch'].unique())
        test_preparation = st.selectbox('Test Preparation Course', df['test preparation course'].unique())
        reading_score = st.slider('Reading Score', 0, 100, 50)
        writing_score = st.slider('Writing Score', 0, 100, 50)

        # Create a dataframe from user input
        input_data = pd.DataFrame({
            'gender': [gender],
            'race/ethnicity': [race_ethnicity],
            'parental level of education': [parental_education],
            'lunch': [lunch],
            'test preparation course': [test_preparation],
            'reading score': [reading_score],
            'writing score': [writing_score]
        })

        # Preprocess the input data
        input_processed = preprocessor.transform(input_data)

        # Make prediction
        if st.button('Predict Math Score'):
            prediction = model.predict(input_processed)
            st.write(f'Predicted Math Score: {prediction[0]:.2f}')

        st.write('Note: This prediction is based on a Ridge Regression model with hyperparameter tuning.')

        # Display model details
        st.subheader('Model Details')
        st.write(f'Best hyperparameters: {model.get_params()}')

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.text("Stack Trace:")
        st.text(traceback.format_exc())
else:
    st.error("Cannot proceed without the dataset. Please ensure 'StudentsPerformance.csv' is available.")