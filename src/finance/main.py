from pathlib import Path
from finance.data import get_training_data
from sklearn.preprocessing import StandardScaler

# Path to the preprocessed file
preprocessed_file = Path("data/processed/processed_data.csv")

# Get training and testing data
X_train, X_test, y_train, y_test = get_training_data(preprocessed_file)
