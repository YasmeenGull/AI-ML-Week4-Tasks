# predictor.py

import pandas as pd


def predict_result(model, study_hours):
    """
    Predict Pass or Fail using the trained model.
    """

    input_data = pd.DataFrame(
        {"StudyHours": [study_hours]}
    )

    probability = model.predict(input_data, verbose=0)[0][0]

    prediction = 1 if probability >= 0.5 else 0

    return prediction, probability