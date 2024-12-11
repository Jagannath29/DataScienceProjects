import pandas as pd
import pickle


def wrangle(filename):
    df = pd.read_csv(filename)

    return df


def make_prediction(data_file, model_path):
    # Wrangle dataset
    X_test = wrangle(data_file)

    # load_model
    with open(model_path, "rb") as file:
        model = pickle.load(file)
        # print("Expected features:", model.feature_names_in_)


    # Generate prediction
    y_test_pred = model.predict(X_test)
    # Put predictions into Series with name "bankrupt", and same index as X_test
    # y_test_pred = pd.Series(y_test_pred, index=X_test.index, name="Bankrupt?")
    return y_test_pred

