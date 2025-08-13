# train_iris_model.py

# Import the libraries we need
import pandas as pd                          # For handling data in tables
from sklearn.datasets import load_iris       # Built-in iris dataset
from sklearn.ensemble import RandomForestClassifier  # Our ML algorithm
from sklearn.model_selection import train_test_split # To split data for testing
from sklearn.metrics import accuracy_score   # To measure how good our model is
import joblib                               # To save our trained model to a file

print("Starting model training...")

# Load the famous iris dataset (150 flowers with measurements)
iris = load_iris()                          # This gives us the data and labels
X = iris.data                               # X = the measurements (4 features per flower)
y = iris.target                             # y = the species labels (0, 1, or 2)

print(f"Loaded {len(X)} flowers with {X.shape[1]} measurements each")
print(f"Species names: {iris.target_names}")  # ['setosa' 'versicolor' 'virginica']

# Split our data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,                                   # Our data and labels
    test_size=0.2,                          # Use 20% for testing
    random_state=42                         # Fixed seed so results are reproducible
)

print(f"Training on {len(X_train)} flowers, testing on {len(X_test)} flowers")

# Create and train our model
model = RandomForestClassifier(
    n_estimators=100,                       # Use 100 decision trees
    random_state=42                         # Fixed seed for reproducible results
)

print("Training the model...")
model.fit(X_train, y_train)                # This is where the learning happens!

# Test how good our model is
y_pred = model.predict(X_test)              # Make predictions on test data
accuracy = accuracy_score(y_test, y_pred)   # Compare predictions to true labels
print(f"Model accuracy: {accuracy:.3f}")   # Show accuracy as percentage

# Save the trained model to a file so our API can use it
joblib.dump(model, "iris_model.pkl")        # Saves model to iris_model.pkl
print("Model saved as 'iris_model.pkl'")
print("Training complete! Ready to build the API.")