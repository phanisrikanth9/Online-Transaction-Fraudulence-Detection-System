import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load your dataset
data = pd.read_csv(r'c:\Users\SRIKANTH\Desktop\minidata(5K1).csv')

# Drop non-relevant columns
data.drop(['nameOrig', 'nameDest'], axis=1, inplace=True)

# Encode 'type' column if it's categorical
data['type'] = pd.factorize(data['type'])[0]

# Assuming 'isFraud' is your target variable and other columns are features
X = data.drop('isFraud', axis=1)
y = data['isFraud']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize XGBoost classifier
model = XGBClassifier()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Save the trained model as a pickle file
joblib.dump(model, 'xgboost_model.pkl')
