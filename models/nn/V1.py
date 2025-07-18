# Version 1: we drop cabin, Name, ticket and passenger id
# categorical_features = ['Pclass', 'Sex', 'SibSp 0,1,>=2', 'Parch 0,1,2,>=3', 'Embarked']
# quantitative_features = ['Age', 'Log_Fare']

import numpy as np
import pandas as pd
import os 
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

    def preprocess_data(df):
        df = df.copy()
        df['SibSp'] = df['SibSp'].apply(lambda x: '0' if x == 0 else ('1' if x == 1 else '>=2'))
        df['Parch'] = df['Parch'].apply(lambda x: '0' if x == 0 else ('1' if x == 1 else ('2' if x==2 else '>=3')))
        drop_col = ['Name', 'Ticket']
        df = df.drop(columns=drop_col)
        return df

    df = preprocess_data(df)

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
    print(all_scores)

def cross_validate(config, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_scores = []
    models = []

    best_acc = -np.inf
    best_model = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")

        x_train, x_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Rebuild model for each fold
        model = Model(config)
        model.build_model(input_shape=(X.shape[1],))
        model.train(x_train, y_train, validation_data=(x_val, y_val))
        model.evaluate(x_val, y_val)

        # Evaluate silently for comparison
        results = model.model.evaluate(x_val, y_val, verbose=0)
        metrics = dict(zip(model.model.metrics_names, results))
        val_acc = metrics.get("accuracy", results[1] if len(results) > 1 else None)
        all_scores.append(val_acc)
        models.append(model)

        # Update best model tracker
        if val_acc is not None and val_acc > best_acc:
            best_acc = val_acc
            best_model = model

    # Save best model after all folds
    save_dir = config['model']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    filename = f"acc_{best_acc:.4f}.keras"
    save_path = os.path.join(save_dir, filename)
    best_model.save_model(save_path)
    print(f"\n[CV] Best model saved to: {save_path} (accuracy: {best_acc:.4f})")

    print(f"\nAverage Accuracy across {n_splits} folds: {np.mean(all_scores):.4f}")
    print(f"Std Dev Accuracy: {np.std(all_scores):.4f}")

    predictions = model.predict(X, verbose=0)
    accuracy = np.mean(np.round(predictions) == y.reshape(-1, 1))
    print(f"Accuracy on the entire dataset: {accuracy:.4f}")

    print(f"All fold accuracies: {all_scores}")

def main():
    config = {
        "training": {
            "epochs": 50,
            "batch_size": 10,
            "learning_rate": 0.0075
        },
        "model": {
        "metrics":["accuracy"],
		"save_dir": "Lstm/saved_models",
        }
    }

    X, y = load_data()
    cross_validate(config, X, y, n_splits=5)

def test(): #get accuracy of the model using the whole dataset
    config = {
        "training": {
            "epochs": 20,
            "batch_size": 10,
            "learning_rate": 0.001
        },
        "model": {
        "metrics":["accuracy"],
		"save_dir": "Lstm/saved_models",
        }
    }
    X, y = load_data()
    model = Model(config)
    model.load_model(path='./lstm/saved_models/acc_0.8322.keras')  # loads from config['model']['save_dir'] + '/model.keras'
    predictions = model.predict(X)
    accuracy = np.mean(np.round(predictions) == y.reshape(-1, 1))
    print(f"Accuracy on the entire dataset: {accuracy:.4f}")

if __name__ == '__main__':
    test()

