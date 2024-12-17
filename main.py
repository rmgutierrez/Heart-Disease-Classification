import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class HeartDiseasePredictor:
    def __init__(self, data_path):
        """
        Initialize the Heart Disease Predictor.
        Args:
            data_path (str): Path to the heart disease dataset.
        """
        self.data_path = data_path
        self.models = {
            'Logistic Regression': LogisticRegression(),
            'Support Vector Machine': SVC(),
            'Random Forest': RandomForestClassifier(random_state=42)
        }
        self.hyperparameters = {
            'Logistic Regression': {'C': [0.01, 0.1, 1, 10]},
            'Support Vector Machine': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
            'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
        }
        self.scaler = StandardScaler()
        self.results = {}

    def load_and_preprocess_data(self):
        """
        Load and preprocess the dataset.
        - Standardizes numerical features
        - Splits data into training and testing sets
        """
        # Load data
        self.df = pd.read_csv(self.data_path)
        print("Dataset loaded successfully!\n")

        # Separate features and target
        X = self.df.drop('target', axis=1)
        y = self.df['target']

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42
        )
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}\n")

    def hyperparameter_tuning(self, model_name):
        """
        Perform hyperparameter tuning for a specific model using GridSearchCV.
        Args:
            model_name (str): Name of the model to tune.
        Returns:
            Best estimator after hyperparameter tuning.
        """
        print(f"Tuning hyperparameters for {model_name}...")
        model = self.models[model_name]
        param_grid = self.hyperparameters[model_name]
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)
        print(f"Best parameters for {model_name}: {grid_search.best_params_}\n")
        return grid_search.best_estimator_

    def train_and_evaluate_model(self, model_name):
        """
        Train and evaluate a specific model.
        Args:
            model_name (str): Name of the model to train and evaluate.
        """
        best_model = self.hyperparameter_tuning(model_name)
        best_model.fit(self.X_train, self.y_train)
        y_pred = best_model.predict(self.X_test)

        # Evaluation metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)

        # Store results
        self.results[model_name] = {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': confusion_matrix(self.y_test, y_pred)
        }
        print(f"{model_name} Performance:\n")
        print(f"Accuracy: {accuracy:.2f}\n")
        print(report)

    def evaluate_all_models(self):
        """
        Train and evaluate all models in the pipeline.
        """
        for model_name in self.models.keys():
            print("=" * 50)
            self.train_and_evaluate_model(model_name)

    def plot_confusion_matrices(self):
        """
        Plot confusion matrices for all trained models.
        """
        for model_name, result in self.results.items():
            cm = result['confusion_matrix']
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.title(f"Confusion Matrix - {model_name}")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.show()

# Main execution
if __name__ == "__main__":
    # Instantiate the class
    predictor = HeartDiseasePredictor(data_path='heart.csv')

    # Load and preprocess data
    predictor.load_and_preprocess_data()

    # Train and evaluate all models
    predictor.evaluate_all_models()

    # Plot confusion matrices
    predictor.plot_confusion_matrices()
