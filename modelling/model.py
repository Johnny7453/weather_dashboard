import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load data with the correct delimiter
df = pd.read_csv('../data/temperature.csv', delimiter=',')

# Ensure the Date column is treated as a datetime object
df['Date'] = pd.to_datetime(df['Date'])

# Create lag features for temperature
df['Temp_Lag1'] = df['Temperature'].shift(1)
df['Temp_Lag2'] = df['Temperature'].shift(2)

# Drop rows with NaN values created by lag features
df = df.dropna()

# Define features and target
X = df[['Moisture', 'Rain', 'Temp_Lag1', 'Temp_Lag2']]
y = df['Temperature']


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model to a file
joblib.dump(model, 'temperature_model.pkl')


