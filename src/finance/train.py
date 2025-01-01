from pathlib import Path
from finance.data import get_training_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Path to the preprocessed file
preprocessed_file = Path("data/processed/processed_data.csv")

# Get training and testing data
X_train, X_test, y_train, y_test = get_training_data(preprocessed_file)


# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

# Detailed evaluation
print(classification_report(y_test, y_pred))

