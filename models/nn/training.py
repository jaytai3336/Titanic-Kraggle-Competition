import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder

from core.model import Model  # your model file
import tensorflow as tf

def load_data():
    df = pd.read_csv('./data/processed/train_processed.csv')

    categorical_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']
    quantitative_features = ['Age', 'Log_Fare']
    features = categorical_features + quantitative_features
    target = 'Survived'

    # Encode categorical features
    for col in categorical_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    X = df[features].values
    y = df[target].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y

def cross_validate(config, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")

        x_train, x_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Rebuild model for each fold
        model = Model(config)
        model.build_model(input_shape=(X.shape[1],))
        model.train(x_train, y_train, validation_data=(x_val, y_val))
        model.evaluate(x_val, y_val)

        # Optional: store accuracy from each fold
        val_loss, val_acc = model.model.evaluate(x_val, y_val, verbose=0)
        all_scores.append(val_acc)

    print(f"\nAverage Accuracy across {n_splits} folds: {np.mean(all_scores):.4f}")
    print(f"Std Dev Accuracy: {np.std(all_scores):.4f}")

def main():
    config = {
        "model": {
            "save_dir": "saved_models",
            "metrics": ["accuracy"]
        },
        "training": {
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001
        }
    }

    X, y = load_data()
    cross_validate(config, X, y, n_splits=5)

if __name__ == '__main__':
    main()