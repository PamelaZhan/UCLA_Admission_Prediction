import pandas as pd
import pickle
import streamlit as st
import matplotlib.pyplot as plt
from lime import lime_tabular

# Set the page title and description
st.title("UCLA Admission Prediction")
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

# create explainer and plot the importance
def plot_explainer(scaler, input_data, MLP_model):
    # load training data
    data = pd.read_csv('data/processed/Processed_admissions.csv')
    data = data.drop(['Admit_Chance'], axis=1)

    # scale training data
    train_scaled = scaler.transform(data)

    # create explainer object
    LIMEexplainer = lime_tabular.LimeTabularExplainer(
            training_data=train_scaled,
            class_names=["Not_Admitted", "Admitted"],
            feature_names=['GRE_Score', 'TOEFL_Score', 'University_Rating', 'SOP', 'LOR', 'CGPA', 'Research'],
            mode='classification'
    )

    # Generate explanation instance
    exp = LIMEexplainer.explain_instance(
        data_row=input_data[0],                 
        predict_fn=MLP_model.predict_proba,           # Model's prediction function
        num_features=7                    # Number of features to include in explanation
    )    

    # Convert explanation to a matplotlib figure
    fig = exp.as_pyplot_figure( )  

    # Get feature importance values from the explanation
    importances = [x[1] for x in exp.as_list()]  
    # reverse the order for plot
    importances.reverse()
    plt.title("")
    # Annotate each bar with its corresponding importance value
    for i, importance in enumerate(importances, start=0):
        plt.text(
            importance,  # x-coordinate of the bar (importance value)
            i,  # y-coordinate (corresponding bar)
            f'{importance:.2f}',  # Display importance value 
            ha='center',  # Align text horizontally 
            va='center',  # Align text vertically 
            fontsize=10,  # Font size for the annotation
            color='black'  # Text color
        )
    # return the plot
    return fig, exp.as_list() 


# Prepare the form to collect user inputs
with st.form("user_inputs"):
    st.subheader("Student Profile")
    
    # create 2 columns
    col1, col2 = st.columns(2)
    
    with col1:
        # GRE Score 
        gre = st.slider("GRE Score", min_value=290, max_value=340, value=316)           
        # University Rating
        uni_rating = st.selectbox("University Rating", ["1", "2", "3", "4", "5"])
        # SOP
        sop = st.number_input("Statement of Purpose Strength", min_value=1.0, max_value=5.0, step=0.1, value=3.4)
        #Research Experience
        research = st.selectbox("Have Research Experience?", ["Yes", "No"])

    with col2:
        # TOFEL Score
        tofel = st.slider("TOFEL Score", min_value=90, max_value=120, value=107)        
        # LOR
        lor = st.number_input("Letter of Recommendation Strength", min_value=1.0, max_value=5.0, step=0.1, value=3.5)
        # CGPA score
        cgpa = st.number_input("Undergraduate GPA", min_value=6.0, max_value=10.0, step=0.01, value=8.6)
        
    
    # Submit button
    submitted = st.form_submit_button("Predict Admission Chance")


# Handle the dummy variables to pass to the model
if submitted:
    # convert to numbers
    GRE_Score = int(gre)
    TOEFL_Score = int(tofel)
    University_Rating = int(uni_rating)
    SOP = float(sop)
    LOR = float(lor)
    CGPA = float(cgpa)               
    Research = 1 if research == "Yes" else 0


    # Prepare the input for prediction.Keep same order as it was trained
    prediction_input = pd.DataFrame([[GRE_Score,TOEFL_Score,University_Rating,SOP,LOR,CGPA,Research]])

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

    # call the function to get explaination plot and importance values
    fig, exp=plot_explainer(scaler, input_scaled, MLP_model)
    # Display explanation in Streamlit
    st.subheader("LIME Explanation for Prediction")
    st.pyplot(fig)
    st.subheader("Feature Contributions:")
    st.table(pd.DataFrame(exp, columns=["Feature", "Importance"]))

st.write(
    """A neural network classifier MLP(multi-layer perceptron) model is used to predict the chance of successful admission."""
)
st.image('confusion_matrix.png')
