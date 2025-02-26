
from src.data.load_dataset import load_and_preprocess_data
from src.features.build_features import trans_features
from src.visualization.visualize import plot_loss_curve
from src.models.train_model import train_MLPmodel
from src.models.predict_model import evaluate_model
import pandas as pd

if __name__ == "__main__":
    # Load and preprocess the data
    data_path = "data/raw/Admission.csv"
    df = load_and_preprocess_data(data_path)

    # Transform and create dummy variables 
    x, y = trans_features(df)

    # Train the  Neural Networks model
    MLPmodel, x_test_scaled, y_test = train_MLPmodel(x, y)

    # Evaluate the model
    accuracy, confusion_mat = evaluate_model(MLPmodel, x_test_scaled, y_test)  

    # show result
    if accuracy > 0.9: # achieve the goal, accuracy is over 90%
        print(f"Successful! The accuracy is {accuracy}. The model achieved the goal.")
        print(f"Confusion Matrix:\n{confusion_mat}")
        # Plot
        plot_loss_curve(MLPmodel)
    else:
        print(f"The accuracy is {accuracy}, not good enough.")

 
