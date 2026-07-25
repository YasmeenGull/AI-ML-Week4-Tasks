# predictor.py

import pandas as pd


def predict_purchase(model, age, salary):
    """
    Predict whether a customer will purchase.
    """

    input_data = pd.DataFrame(
        {
            "Age": [age],
            "Salary": [salary]
        }
    )

    prediction = model.predict(input_data)

    return prediction[0]