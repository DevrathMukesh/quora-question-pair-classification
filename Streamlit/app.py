import streamlit as st
import pickle
import nltk
import helper

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords', quiet=True)  # Quiet mode to suppress output

# Set up the Streamlit app
st.title('Duplicate Question Pair Detector')
st.write(
    "This app helps you determine whether two questions are likely to be duplicates. "
    "Enter two questions below and click 'Find' to get the result."
)

# Input fields for the questions
q1 = st.text_input('Enter Question 1')
q2 = st.text_input('Enter Question 2')

# Check for button click
if st.button('Find'):
    # Validate input fields
    if not q1 or not q2:
        st.warning("Please enter both questions to proceed.")
    else:
        # Specify the exact model path
        model_path = './Streamlit/models/RandomForest_Bow.pkl'

        # Load the trained model
        try:
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
            st.write(f"Model loaded successfully from: {model_path}")
        except Exception as e:
            st.error(f"An error occurred while loading the model: {e}")
            st.stop()

        # Prepare the query for prediction
        try:
            query = helper.query_point_creator(q1, q2)  # Ensure this helper function is defined
            result = model.predict(query)[0]

            # Display the result
            if result:
                st.success("The questions are likely duplicates.")
            else:
                st.info("The questions are not duplicates.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
