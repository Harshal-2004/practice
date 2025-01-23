# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Step 1: Load the train and test data
train_data = pd.read_csv('train_data.csv')  # Training data: Lat, Long_, Deaths, Case_Fatality_Ratio
test_data = pd.read_csv('test_data.csv')    # Test data: Lat, Long_

# Step 2: Handle missing data (if any)
# Impute missing values in 'Deaths' and 'Case_Fatality_Ratio'
imputer = SimpleImputer(strategy='median')
train_data[['Deaths', 'Case_Fatality_Ratio']] = imputer.fit_transform(train_data[['Deaths', 'Case_Fatality_Ratio']])

# Step 3: Add ConfirmedCases to the training dataset
# Avoid division by zero by replacing 0 CFR with a small value
train_data['Case_Fatality_Ratio'].replace(0, 1e-6, inplace=True)
train_data['ConfirmedCases'] = train_data['Deaths'] / train_data['Case_Fatality_Ratio']

# Step 4: Prepare the features and targets
X_train = train_data[['Lat', 'Long_']]
y_deaths = train_data['Deaths']
y_cfr = train_data['Case_Fatality_Ratio']
y_cases = train_data['ConfirmedCases']

# Step 5: Data Splitting for Validation
X_train_split, X_val_split, y_train_deaths, y_val_deaths = train_test_split(X_train, y_deaths, test_size=0.2, random_state=42)
_, _, y_train_cfr, y_val_cfr = train_test_split(X_train, y_cfr, test_size=0.2, random_state=42)
_, _, y_train_cases, y_val_cases = train_test_split(X_train, y_cases, test_size=0.2, random_state=42)

# Step 6: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_split)
X_val_scaled = scaler.transform(X_val_split)
X_test_scaled = scaler.transform(test_data[['Lat', 'Long_']])

# Step 7: Train RandomForestRegressor models for each target
# Model for Deaths
deaths_model = RandomForestRegressor(n_estimators=100, random_state=42)
deaths_model.fit(X_train_scaled, y_train_deaths)

# Model for CFR
cfr_model = RandomForestRegressor(n_estimators=100, random_state=42)
cfr_model.fit(X_train_scaled, y_train_cfr)

# Model for Confirmed Cases
cases_model = RandomForestRegressor(n_estimators=100, random_state=42)
cases_model.fit(X_train_scaled, y_train_cases)

# Step 8: Validate the models and evaluate performance using RMSE
# Predictions for Deaths
val_predictions_deaths = deaths_model.predict(X_val_scaled)
rmse_deaths = np.sqrt(mean_squared_error(y_val_deaths, val_predictions_deaths))
print(f'Validation RMSE for Deaths: {rmse_deaths}')

# Predictions for CFR
val_predictions_cfr = cfr_model.predict(X_val_scaled)
rmse_cfr = np.sqrt(mean_squared_error(y_val_cfr, val_predictions_cfr))
print(f'Validation RMSE for CFR: {rmse_cfr}')

# Predictions for Confirmed Cases
val_predictions_cases = cases_model.predict(X_val_scaled)
rmse_cases = np.sqrt(mean_squared_error(y_val_cases, val_predictions_cases))
print(f'Validation RMSE for Confirmed Cases: {rmse_cases}')

# Step 9: Predict the missing values for the test dataset
predicted_deaths = deaths_model.predict(X_test_scaled)
predicted_cfr = cfr_model.predict(X_test_scaled)
predicted_cases = cases_model.predict(X_test_scaled)

# Step 10: Create a DataFrame with predictions for the test dataset
test_predictions_df = pd.DataFrame({
    'Lat': test_data['Lat'],
    'Long_': test_data['Long_'],
    'PredictedDeaths': predicted_deaths,
    'PredictedConfirmedCases': predicted_cases,
    'PredictedCFR': predicted_cfr
})

# Output the predictions for test dataset
print(test_predictions_df)

# Optional: Save the predictions to a CSV file
test_predictions_df.to_csv('test_predictions.csv', index=False)

# Optional: Save the trained models for future use
import joblib
joblib.dump(deaths_model, 'deaths_model.pkl')
joblib.dump(cfr_model, 'cfr_model.pkl')
joblib.dump(cases_model, 'cases_model.pkl')
