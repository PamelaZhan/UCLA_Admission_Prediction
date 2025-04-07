# Predicting Chances of Admission at UCLA
This app has been built using Streamlit and deployed with Streamlit community cloud

[Visit the app here](https://ucla-admission-prediction.streamlit.app/)

This application predicts chances of admission into the University of California, Los Angeles (UCLA). The predicted output gives students a fair idea about their chances of getting accepted. Accuracy of 90% and above.

## Features
- User-friendly interface powered by Streamlit.
- Input form to enter details such as GRE Score, TOEFL Score, SOP, LOR, and other relevant factors.
- Real-time prediction of admission chance, Accuracy of 90% and above.
- Accessible via Streamlit Community Cloud.

## Dataset
The application is trained on the **current admission dataset**. It includes features like:
- GRE_Score: (out of 340)
- TOEFL_Score: (out of 120)
- University_Rating: It indicates the Bachelor University ranking (out of 5)
- SOP: Statement of Purpose Strength (out of 5)
- LOR: Letter of Recommendation Strength (out of 5)
- CGPA: Student's Undergraduate GPA(out of 10)
- Research: Whether the student has Research Experience (either 0 or 1)
- Admit_Chance: (ranging from 0 to 1)


## Technologies Used
- **Streamlit**: For building the web application.
- **Scikit-learn**: For model training and evaluation.
- **Pandas** and **NumPy**: For data preprocessing and manipulation.
- **Matplotlib** and **Seaborn**: For exploratory data analysis and visualization (if applicable).

## Model
A neural network classifier MLP(multi-layer perceptron) model is trained and utilized to predict the admit chance.

#### Thank you for using the UCLA Admission Prediction Application! Feel free to share your feedback.
