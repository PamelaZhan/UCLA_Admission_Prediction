import pandas as pd
import pickle
import streamlit as st

# Set the page title and description
st.title("UCLA Admission Prediction Appication")
st.write("""
This app predicts chances of admission into the University of California, Los Angeles (UCLA) 
based on student's profile.
""")

# Load the pre-trained model
with open("models/MLPmodel.pkl", "rb") as pkl:
    MLP_model = pickle.load(pkl)
# Load the pre-fit scaler 
with open('models/scaler.pkl', 'rb') as f:
    scaler=pickle.load(f)


# Prepare the form to collect user inputs
with st.form("user_inputs"):
    st.subheader("Student Profile")
    
    # GRE Score 
    gre = st.slider("GRE Score", min_value=290, max_value=340)
    
    # TOFEL Score
    tofel = st.slider("TOFEL Score", min_value=90, max_value=120)

    # University Rating
    uni_rating = st.selectbox("University Rating", ["1", "2", "3", "4", "5"])

    # SOP
    sop = st.number_input("Statement of Purpose Strength", min_value=1.0, max_value=5.0, step=0.1)

    # LOR
    lor = st.number_input("Letter of Recommendation Strength", min_value=1.0, max_value=5.0, step=0.1)

    # CGPA score
    cgpa = st.number_input("Undergraduate GPA", min_value=6.0, max_value=10.0, step=0.01)

    #Research Experience
    research = st.selectbox("Have Research Experience?", ["Yes", "No"])
    
    # Submit button
    submitted = st.form_submit_button("Predict Admission Chance")


# Handle the dummy variables to pass to the model
if submitted:
    # convert to number
    GRE_Score = int(gre)
    TOEFL_Score = int(tofel)
    SOP = float(sop)
    LOR = float(lor)
    CGPA = float(cgpa)               

    # deal dummy feature
    University_Rating_1 = 1 if uni_rating == "1" else 0
    University_Rating_2 = 1 if uni_rating == "2" else 0
    University_Rating_3 = 1 if uni_rating == "3" else 0
    University_Rating_4 = 1 if uni_rating == "4" else 0
    University_Rating_5 = 1 if uni_rating == "5" else 0

    Research_0 = 1 if research == "No" else 0
    Research_1 = 1 if research == "Yes" else 0


    # Prepare the input for prediction. This has to go in the same order as it was trained
    prediction_input = pd.DataFrame([[GRE_Score,TOEFL_Score,SOP,LOR,CGPA,
                University_Rating_1,University_Rating_2,University_Rating_3,University_Rating_4,
                University_Rating_5,Research_0,Research_1]]
    )

    # Scale the input
    input_scaled = scaler.transform(prediction_input)
    # Make prediction
    new_prediction = MLP_model.predict(input_scaled)

    # Display result
    st.subheader("Prediction Result:")
    if new_prediction[0] == 1:
        st.write("Congraduation! You got the offer!")
    else:
        st.write("Sorry, you are not eligible.")

st.write(
    """We used a Neural Networks to predict the chance of successful admission."""
)
st.image('confusion_matrix.png')
