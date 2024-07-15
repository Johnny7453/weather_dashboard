import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib


SHOW_RESIDUAL_PLOT = False

# Load data with the correct delimiter
df = pd.read_csv('../data/temperature.csv', delimiter=',')

# Ensure the Date column is treated as a datetime object
df['Date'] = pd.to_datetime(df['Date'], yearfirst=True, utc=True, format='ISO8601')

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


# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R²): {r2}')

# Calculate residuals
residuals = y_test - y_pred

# Plot residuals

if(SHOW_RESIDUAL_PLOT):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.show()

# Save the model to a file
joblib.dump(model, 'temperature_model.pkl')

print("TEMPERATURE MODEL FITTED") 
