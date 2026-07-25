# main.py

import os

from sklearn.metrics import accuracy_score

from data_loader import load_dataset
from preprocessing import preprocess
from classification_model import train_model
from predictor import predict_purchase


def main():

    # Dataset Path
    current_folder = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_folder, "dataset.csv")

    # Load Dataset
    df = load_dataset(dataset_path)

    # Split Dataset
    X_train, X_test, y_train, y_test = preprocess(df)

    # Train Model
    model = train_model(X_train, y_train)

    # Accuracy
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("========== Classification Model ==========\n")

    age = int(input("Enter Age: "))
    salary = int(input("Enter Salary: "))

    prediction = predict_purchase(model, age, salary)

    print("\n========== Prediction ==========\n")

    if prediction == 1:
        print("Customer Will Purchase")
    else:
        print("Customer Will Not Purchase")

    print(f"\nAccuracy Score : {accuracy * 100:.2f}%")

    print("\n========== Task Completed ==========")


if __name__ == "__main__":
    main()