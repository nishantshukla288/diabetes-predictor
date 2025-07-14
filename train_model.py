from utils.data_utils import load_data
from sklearn.svm import SVC
import joblib

# Load the data
X_train, X_test, y_train, y_test, scaler = load_data()

# Train the model
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Evaluate accuracy
print("Test accuracy:", model.score(X_test, y_test))

# Save model and scaler
joblib.dump(model, 'models/diabetes_model.joblib')
joblib.dump(scaler, 'models/scaler.joblib')
