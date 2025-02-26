# Import accuracy score
from sklearn.metrics import accuracy_score, confusion_matrix
from ..logging.logging import logging_decorator

@logging_decorator
# # Function to predict and evaluate
def evaluate_model(MLP_model, x_test_scaled, y_test):

    # Make Predictions
    y_pred = MLP_model.predict(x_test_scaled)

    # calculate confusion matrix
    confusion_mat = confusion_matrix(y_test, y_pred)
    # calculate accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)


    return accuracy, confusion_mat