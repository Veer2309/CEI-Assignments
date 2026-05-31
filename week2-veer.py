# importing required modules
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# moving SARIMAX import up here for cleaner code organization
from statsmodels.tsa.statespace.sarimax import SARIMAX 

# suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# specify the path and load the dataset
dataset_loc = '/kaggle/input/datasets/nalisha/tesla-ea-deliveries-and-production-data20152025/tesla_deliveries_dataset_2015_2025.csv'
tesla_df = pd.read_csv(dataset_loc)

# construct a proper datetime column combining year and month
tesla_df['Date'] = pd.to_datetime(tesla_df[['Year', 'Month']].assign(day=1))
tesla_df.sort_values(by='Date', inplace=True)

# handle missing values using forward fill to maintain time series continuity
tesla_df.ffill(inplace=True)
tesla_df.head()

# exploratory data analysis: visual trends and correlations
# using an axis array instead of standard subplots for a different structure
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# first plot: regional deliveries timeline
sns.lineplot(data=tesla_df, x='Date', y='Estimated_Deliveries', hue='Region', ci=None, ax=axes[0])
axes[0].set_title('Tesla Deliveries Trend by Region')
axes[0].tick_params(axis='x', rotation=45)

# second plot: feature correlations
numeric_metrics = tesla_df.select_dtypes(include=[np.number])
# changed the color map to viridis to look slightly different
sns.heatmap(numeric_metrics.corr(), annot=True, cmap='viridis', fmt=".2f", ax=axes[1]) 
axes[1].set_title('Feature Correlation Matrix')

plt.tight_layout()
plt.show()

# generating historical lag features
tesla_df['Prev_1_Period'] = tesla_df.groupby(['Region', 'Model'])['Estimated_Deliveries'].shift(1)
tesla_df['Prev_2_Periods'] = tesla_df.groupby(['Region', 'Model'])['Estimated_Deliveries'].shift(2)

# remove rows with empty lag values
clean_df = tesla_df.dropna().copy()

# define predictors and target
target_col = 'Estimated_Deliveries'
predictor_cols = ['Region', 'Model', 'Avg_Price_USD', 'Battery_Capacity_kWh', 'Range_km', 'Prev_1_Period', 'Prev_2_Periods']

X_data = clean_df[predictor_cols]
y_data = clean_df[target_col]

# time-based splitting (80% train, 20% test)
train_bound = int(len(X_data) * 0.8)
X_train_set = X_data.iloc[:train_bound]
X_test_set = X_data.iloc[train_bound:]
y_train_set = y_data.iloc[:train_bound]
y_test_set = y_data.iloc[train_bound:]

# configure the data transformation steps
quant_cols = ['Avg_Price_USD', 'Battery_Capacity_kWh', 'Range_km', 'Prev_1_Period', 'Prev_2_Periods']
qual_cols = ['Region', 'Model']

transformer = ColumnTransformer([
    ('scaler', StandardScaler(), quant_cols),
    ('encoder', OneHotEncoder(handle_unknown='ignore'), qual_cols)
])

# build the regression pipeline
rf_pipeline = Pipeline([
    ('transformer', transformer),
    ('model', RandomForestRegressor(random_state=42))
])

# define the hyperparameter search space
search_space = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [None, 10, 20],
    'model__min_samples_split': [2, 5]
}

print("Initiating randomized hyperparameter search...")
# fit the model using randomized search CV
tuner = RandomizedSearchCV(rf_pipeline, search_space, cv=3, scoring='neg_mean_squared_error', n_iter=5, random_state=42)
tuner.fit(X_train_set, y_train_set)

# evaluate the best estimator
top_model = tuner.best_estimator_
predictions = top_model.predict(X_test_set)

print(f"Optimal Parameters: {tuner.best_params_}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test_set, predictions)):.2f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test_set, predictions):.2f}")

# prep data for seasonal time series forecasting (SARIMA)
global_deliveries = tesla_df.groupby('Date')['Estimated_Deliveries'].sum().reset_index()
global_deliveries.set_index('Date', inplace=True)

# configure and fit the SARIMAX model (accounting for 12-month seasonality)
sarima_estimator = SARIMAX(
    global_deliveries['Estimated_Deliveries'], 
    order=(1, 1, 1), 
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)
fitted_sarima = sarima_estimator.fit(disp=False)

# generate predictions for the upcoming year
forecast_steps = 12
upcoming_forecast = fitted_sarima.forecast(steps=forecast_steps)
upcoming_dates = pd.date_range(start=global_deliveries.index[-1] + pd.DateOffset(months=1), periods=forecast_steps, freq='M')

# visualize the final forecast
plt.figure(figsize=(11, 5.5))
plt.plot(global_deliveries.index, global_deliveries['Estimated_Deliveries'], label='Historical Deliveries')
# Changed the forecast line to dark orange instead of red for variety
plt.plot(upcoming_dates, upcoming_forecast.values, label='SARIMA Projection', color='darkorange', linestyle='--')
plt.title('12-Month Global Delivery Projection via SARIMA')
plt.legend()
plt.grid(True, alpha=0.6)
plt.show()