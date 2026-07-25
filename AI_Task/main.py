# main.py

import os

from data_loader import load_dataset
from preprocessing import preprocess_data
from neural_network import build_model, train_model
from predictor import predict_result
from workflow import show_workflow


def main():

    current_folder = os.path.dirname(os.path.abspath(__file__))

    dataset_path = os.path.join(current_folder, "dataset.csv")

    # Load dataset
    df = load_dataset(dataset_path)

    if df is None:
        return

    # Preprocess
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Build model
    model = build_model()

    # Train model
    model, history = train_model(model, X_train, y_train)

    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    print("========== Neural Network ==========\n")

    hours = float(input("Enter Study Hours: "))

    prediction, probability = predict_result(model, hours)

    print("\n========== Prediction ==========\n")

    if prediction == 1:
        print("Prediction : PASS")
    else:
        print("Prediction : FAIL")

    print(f"Confidence : {probability:.2f}")

    print(f"Model Accuracy : {accuracy * 100:.2f}%")

    show_workflow()


if __name__ == "__main__":
    main()