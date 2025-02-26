
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from ..logging.logging import logging_decorator

@logging_decorator
# Plotting loss curve
def plot_loss_curve(MLPmodel):
    
    loss_values = MLPmodel.loss_curve_
    # Plotting the loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values, label='Loss', color='blue')
    plt.title('Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss') 
    plt.grid(True)
    # save the plot to a file
    plt.savefig('loss_curve.png', dpi=300)
    # Show the plot
    plt.show()


@logging_decorator
def plot_feature_importance(model, x):
    """
    Plot a bar chart showing the feature importances.
    
    Args:
        feature_names (list): List of feature names.
        feature_importances (list): List of feature importance values.
    """
    fig, ax = plt.subplots() # a single subplot
    ax = sns.barplot(x=model.feature_importances_, y=x.columns)
    plt.title("Feature importance chart")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    # Save the plot to a file
    fig.savefig("feature_importance.png")
    # Show the plot
    plt.show()


