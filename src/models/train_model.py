from sklearn.model_selection import train_test_split
# import scaler for transforming variables to range [0, 1]
from sklearn.preprocessing import MinMaxScaler
# import MLPClassifier for classification tasks, MLP (Multi-layer Perceptron)
from sklearn.neural_network import MLPClassifier
import pickle
from ..logging.logging import logging_decorator

@logging_decorator
# Function to train the model
def train_MLPmodel(x, y):
    # Splitting the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

    # transform xtrain and xtest using MinMaxScaler, scale data into 0--1
    scaler = MinMaxScaler() 
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # create a instance of MLPClassifier. 2 hidder layers, each layer with 2 neurons
    MLP_model = MLPClassifier(hidden_layer_sizes=(2,2), batch_size=50, max_iter=200)  

    # train the model. 
    MLP_model.fit(x_train_scaled,y_train)
       
    # Save the trained model
    with open('models/MLPmodel.pkl', 'wb') as f:
        pickle.dump(MLP_model, f)

    # Save the scaler to a file
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    return MLP_model, x_test_scaled, y_test


