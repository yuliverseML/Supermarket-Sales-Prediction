# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('ggplot')
sns.set_palette('Set2')

#1. Exploratory Data Analysis (EDA)
# Load the dataset
df = pd.read_csv('supermarket_sales.csv')

# Display basic information
print("Dataset Shape:", df.shape)
print("\nSample data:")
display(df.head())
print("\nData types and null values:")
display(df.info())
print("\nStatistical summary:")
display(df.describe())

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing values in each column:")
print(missing_values)

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")

# Convert date and time columns to datetime with correct format
df['Date'] = pd.to_datetime(df['Date'])
# Use the 12-hour format with seconds and AM/PM indicator
df['Time'] = pd.to_datetime(df['Time'], format='%I:%M:%S %p').dt.time
df['datetime'] = df.apply(lambda row: datetime.combine(row['Date'].date(), row['Time']), axis=1)

# Extract date components
df['day'] = df['Date'].dt.day
df['month'] = df['Date'].dt.month
df['year'] = df['Date'].dt.year
df['day_of_week'] = df['Date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# Extract time components
df['hour'] = df.apply(lambda x: x['Time'].hour, axis=1)
df['time_of_day'] = pd.cut(df['hour'], 
                         bins=[0, 6, 12, 18, 24], 
                         labels=['Night', 'Morning', 'Afternoon', 'Evening'])

#Data Visualization
# Set up the visualization layout
plt.figure(figsize=(20, 15))

# 1. Sales distribution
plt.subplot(3, 3, 1)
sns.histplot(df['Sales'], kde=True)
plt.title('Sales Distribution')
plt.xlabel('Sales')
plt.ylabel('Frequency')

# 2. Sales by Branch
plt.subplot(3, 3, 2)
sns.boxplot(x='Branch', y='Sales', data=df)
plt.title('Sales by Branch')

# 3. Sales by Product Line
plt.subplot(3, 3, 3)
product_sales = df.groupby('Product line')['Sales'].mean().sort_values(ascending=False)
sns.barplot(x=product_sales.index, y=product_sales.values)
plt.title('Average Sales by Product Line')
plt.xticks(rotation=45, ha='right')

# 4. Sales by Customer Type
plt.subplot(3, 3, 4)
sns.boxplot(x='Customer type', y='Sales', data=df)
plt.title('Sales by Customer Type')

# 5. Sales by Gender
plt.subplot(3, 3, 5)
sns.boxplot(x='Gender', y='Sales', data=df)
plt.title('Sales by Gender')

# 6. Sales by Payment Method
plt.subplot(3, 3, 6)
sns.boxplot(x='Payment', y='Sales', data=df)
plt.title('Sales by Payment Method')

# 7. Sales by Day of Week
plt.subplot(3, 3, 7)
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_sales = df.groupby('day_of_week')['Sales'].mean()
sns.barplot(x=day_sales.index, y=day_sales.values)
plt.title('Average Sales by Day of Week')
plt.xticks(ticks=range(7), labels=days, rotation=45)

# 8. Sales by Time of Day
plt.subplot(3, 3, 8)
time_sales = df.groupby('time_of_day')['Sales'].mean()
sns.barplot(x=time_sales.index, y=time_sales.values)
plt.title('Average Sales by Time of Day')

# 9. Sales by Hour
plt.subplot(3, 3, 9)
hour_sales = df.groupby('hour')['Sales'].mean()
sns.lineplot(x=hour_sales.index, y=hour_sales.values)
plt.title('Average Sales by Hour')
plt.xlabel('Hour')
plt.ylabel('Average Sales')

plt.tight_layout()
plt.show()

#Correlation Analysis
# Select numerical columns for correlation analysis
numerical_cols = ['Unit price', 'Quantity', 'Tax 5%', 'Sales', 'cogs', 
                  'gross income', 'Rating', 'day', 'month', 'hour']

# Create correlation matrix
corr_matrix = df[numerical_cols].corr()

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
plt.show()

# Analyze relationship between Sales and other variables
plt.figure(figsize=(10, 6))
for i, col in enumerate(['Unit price', 'Quantity', 'Rating']):
    plt.subplot(1, 3, i+1)
    sns.scatterplot(x=col, y='Sales', data=df, alpha=0.6)
    plt.title(f'Sales vs {col}')
plt.tight_layout()
plt.show()

#Time Series Analysis
# Resample data to daily frequency for time series analysis
daily_sales = df.groupby('Date')['Sales'].sum().reset_index()
daily_sales = daily_sales.set_index('Date')

# Plot time series
plt.figure(figsize=(14, 6))
plt.plot(daily_sales.index, daily_sales['Sales'], marker='o', linestyle='-')
plt.title('Daily Total Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Decompose time series to identify trend, seasonality, and residuals
from statsmodels.tsa.seasonal import seasonal_decompose

# Only if we have enough data points (ideally at least 2 seasons)
if len(daily_sales) >= 14:  # Minimum requirement for weekly seasonality
    decomposition = seasonal_decompose(daily_sales['Sales'], model='additive', period=7)
    
    plt.figure(figsize=(14, 10))
    plt.subplot(4, 1, 1)
    plt.plot(decomposition.observed)
    plt.title('Observed')
    
    plt.subplot(4, 1, 2)
    plt.plot(decomposition.trend)
    plt.title('Trend')
    
    plt.subplot(4, 1, 3)
    plt.plot(decomposition.seasonal)
    plt.title('Seasonality')
    
    plt.subplot(4, 1, 4)
    plt.plot(decomposition.resid)
    plt.title('Residuals')
    
    plt.tight_layout()
    plt.show()
else:
    print("Not enough data points for seasonal decomposition. Need at least 14 days.")

# Monthly sales pattern
monthly_sales = df.groupby('month')['Sales'].sum()
plt.figure(figsize=(10, 6))
sns.barplot(x=monthly_sales.index, y=monthly_sales.values)
plt.title('Total Sales by Month')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.show()

#2. Data Preprocessing and Cleaning
# Function to detect outliers using IQR method
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

# Check for outliers in key numerical columns
numerical_features = ['Unit price', 'Quantity', 'Sales', 'Rating']
for feature in numerical_features:
    outliers, lower, upper = detect_outliers(df, feature)
    print(f"\nOutliers in {feature}: {len(outliers)}")
    print(f"Lower bound: {lower:.2f}, Upper bound: {upper:.2f}")
    
    # Visualize outliers
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[feature])
    plt.title(f'Boxplot of {feature} with Outliers')
    plt.tight_layout()
    plt.show()

# Create a copy of the dataframe for preprocessing
df_processed = df.copy()

# Handle outliers (capping method)
for feature in numerical_features:
    _, lower, upper = detect_outliers(df_processed, feature)
    df_processed[feature] = df_processed[feature].clip(lower=lower, upper=upper)
    
    # Verify outlier handling
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df_processed[feature])
    plt.title(f'Boxplot of {feature} After Outlier Treatment')
    plt.tight_layout()
    plt.show()

#Categorical Feature Encoding
# Identify categorical features
categorical_features = ['Branch', 'City', 'Customer type', 'Gender', 'Product line', 'Payment', 'time_of_day']

# One-hot encoding for categorical variables
df_encoded = pd.get_dummies(df_processed, columns=categorical_features, drop_first=True)

# Display the first few rows of the encoded dataframe
print("DataFrame after encoding categorical features:")
display(df_encoded.head())
print(f"Original shape: {df_processed.shape}, Encoded shape: {df_encoded.shape}")

#Numerical Feature Scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Select features to scale
features_to_scale = ['Unit price', 'Quantity', 'Tax 5%', 'Sales', 'cogs', 
                     'gross income', 'Rating', 'day', 'month', 'hour']

# Create a copy for scaling
df_scaled = df_encoded.copy()

# Apply StandardScaler
scaler = StandardScaler()
df_scaled[features_to_scale] = scaler.fit_transform(df_scaled[features_to_scale])

# Display the first few rows of the scaled dataframe
print("DataFrame after scaling numerical features:")
display(df_scaled.head())

# Visualize the effect of scaling
plt.figure(figsize=(15, 8))
for i, feature in enumerate(features_to_scale[:5]):  # Just show first 5 features
    plt.subplot(2, 3, i+1)
    sns.histplot(df_encoded[feature], kde=True, color='blue', alpha=0.5, label='Before scaling')
    sns.histplot(df_scaled[feature], kde=True, color='red', alpha=0.5, label='After scaling')
    plt.title(f'Distribution of {feature}')
    plt.legend()
plt.tight_layout()
plt.show()

#3. Feature Engineering
# Create a copy for feature engineering
df_features = df_processed.copy()

# Create features based on invoice totals
df_features['items_per_invoice'] = df_features['Quantity']
df_features['avg_item_price'] = df_features['Sales'] / df_features['Quantity']
df_features['sales_per_unit'] = df_features['Sales'] / df_features['Unit price']

# Create time-based features
df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
df_features['day_sin'] = np.sin(2 * np.pi * df_features['day'] / 31)
df_features['day_cos'] = np.cos(2 * np.pi * df_features['day'] / 31)
df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
df_features['day_of_week_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
df_features['day_of_week_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)

# Aggregate features
# Calculate average sales, quantity, and rating by product line
product_line_agg = df_features.groupby('Product line').agg({
    'Sales': 'mean',
    'Quantity': 'mean',
    'Rating': 'mean'
}).reset_index()

product_line_agg.columns = ['Product line', 'avg_sales_by_product', 'avg_quantity_by_product', 'avg_rating_by_product']

# Merge aggregated data back to the main dataframe
df_features = pd.merge(df_features, product_line_agg, on='Product line', how='left')

# Calculate average sales by branch
branch_agg = df_features.groupby('Branch').agg({
    'Sales': 'mean',
    'Quantity': 'mean'
}).reset_index()

branch_agg.columns = ['Branch', 'avg_sales_by_branch', 'avg_quantity_by_branch']

# Merge aggregated data back to the main dataframe
df_features = pd.merge(df_features, branch_agg, on='Branch', how='left')

# Display the first few rows of the dataframe with engineered features
print("DataFrame after feature engineering:")
display(df_features.head())

# Visualize the relationship between new features and sales
plt.figure(figsize=(15, 10))
features_to_plot = ['items_per_invoice', 'avg_item_price', 'avg_sales_by_product', 'avg_sales_by_branch']
for i, feature in enumerate(features_to_plot):
    plt.subplot(2, 2, i+1)
    sns.scatterplot(x=feature, y='Sales', data=df_features, alpha=0.6)
    plt.title(f'Sales vs {feature}')
plt.tight_layout()
plt.show()

#Customer Segmentation (Clusterization)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Select features for customer segmentation
segmentation_features = ['Quantity', 'Unit price', 'Sales', 'Rating']

# Scale the features
segmentation_scaler = StandardScaler()
segmentation_data = segmentation_scaler.fit_transform(df_features[segmentation_features])

# Find optimal number of clusters using the Elbow method
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(segmentation_data)
    inertia.append(kmeans.inertia_)

# Plot the Elbow method results
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.xticks(k_range)
plt.grid(True)
plt.show()

# Apply K-Means clustering with the optimal number of clusters (let's assume it's 3)
optimal_k = 3  # This should be determined from the elbow plot
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df_features['customer_segment'] = kmeans.fit_predict(segmentation_data)

# Analyze the segments
segment_analysis = df_features.groupby('customer_segment').agg({
    'Sales': 'mean',
    'Quantity': 'mean',
    'Unit price': 'mean',
    'Rating': 'mean'
}).reset_index()

print("Customer Segment Analysis:")
display(segment_analysis)

# Visualize the segments
plt.figure(figsize=(12, 8))
for i, feature in enumerate(['Sales', 'Quantity', 'Unit price', 'Rating']):
    plt.subplot(2, 2, i+1)
    sns.boxplot(x='customer_segment', y=feature, data=df_features)
    plt.title(f'{feature} by Customer Segment')
plt.tight_layout()
plt.show()

# Create dummy variables for customer segments
df_features = pd.get_dummies(df_features, columns=['customer_segment'], prefix='segment')

# Prepare the final feature set by encoding categorical variables
df_final = pd.get_dummies(df_features, columns=categorical_features, drop_first=True)

print("Final feature set shape:", df_final.shape)

#4. Model Building and Evaluation
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import time

# Define function to evaluate and compare models
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    # Train the model
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Make predictions
    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Perform cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores.mean())
    
    return {
        'Model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'CV RMSE': cv_rmse,
        'Training Time': training_time,
        'Prediction Time': prediction_time
    }

# Prepare data for modeling
# Drop non-predictive columns and the target variable
X = df_final.drop(['Sales', 'Invoice ID', 'Date', 'Time', 'datetime', 'Tax 5%', 'cogs', 'gross income'], axis=1)
y = df_final['Sales']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Define models to evaluate
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'XGBoost': xgb.XGBRegressor(random_state=42),
    'LightGBM': lgb.LGBMRegressor(random_state=42)
}

# Evaluate each model
results = []
for name, model in models.items():
    result = evaluate_model(model, X_train, X_test, y_train, y_test, name)
    results.append(result)

# Create a dataframe with the results
results_df = pd.DataFrame(results)
print("Model Comparison Results:")
display(results_df.sort_values('RMSE'))

# Visualize model performance
plt.figure(figsize=(12, 8))
# Plot RMSE
plt.subplot(2, 2, 1)
sns.barplot(x='Model', y='RMSE', data=results_df.sort_values('RMSE'))
plt.title('RMSE Comparison')
plt.xticks(rotation=45)

# Plot R2
plt.subplot(2, 2, 2)
sns.barplot(x='Model', y='R2', data=results_df.sort_values('R2', ascending=False))
plt.title('R² Comparison')
plt.xticks(rotation=45)

# Plot training time
plt.subplot(2, 2, 3)
sns.barplot(x='Model', y='Training Time', data=results_df.sort_values('Training Time'))
plt.title('Training Time Comparison (seconds)')
plt.xticks(rotation=45)

# Plot prediction time
plt.subplot(2, 2, 4)
sns.barplot(x='Model', y='Prediction Time', data=results_df.sort_values('Prediction Time'))
plt.title('Prediction Time Comparison (seconds)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

#Fine-tune the Best Model
# Identify the best performing model (let's assume it's XGBoost)
best_model_name = results_df.sort_values('RMSE').iloc[0]['Model']
print(f"The best performing model is: {best_model_name}")

# Fine-tune the best model using GridSearchCV
if best_model_name == 'XGBoost':
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    # Create the model
    model = xgb.XGBRegressor(random_state=42)
    
elif best_model_name == 'Random Forest':
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Create the model
    model = RandomForestRegressor(random_state=42)
    
elif best_model_name == 'LightGBM':
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'num_leaves': [31, 50, 70]
    }
    
    # Create the model
    model = lgb.LGBMRegressor(random_state=42)
    
else:
    # For other models, define appropriate parameter grids
    param_grid = {}
    model = models[best_model_name]

# Perform grid search if we have a parameter grid
if param_grid:
    # Define grid search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    # Get best parameters
    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")
    
    # Evaluate the model with best parameters
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Tuned model RMSE: {rmse:.2f}")
    print(f"Tuned model MAE: {mae:.2f}")
    print(f"Tuned model R²: {r2:.2f}")
    
    # Visualize actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title('Actual vs Predicted Sales')
    plt.xlabel('Actual Sales')
    plt.ylabel('Predicted Sales')
    plt.show()
    
    # Feature importance (for tree-based models)
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
        plt.title('Top 20 Feature Importances')
        plt.show()

#5. Time Series Forecasting (if applicable)
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

# Prepare time series data
# Group by date to get daily sales
ts_data = df.groupby('Date')['Sales'].sum().reset_index()
ts_data = ts_data.set_index('Date')

# Split the time series data
train_size = int(len(ts_data) * 0.8)
train_ts = ts_data[:train_size]
test_ts = ts_data[train_size:]

print(f"Training set size: {len(train_ts)}")
print(f"Testing set size: {len(test_ts)}")

# Plot the time series data
plt.figure(figsize=(12, 6))
plt.plot(train_ts.index, train_ts['Sales'], label='Training Data')
plt.plot(test_ts.index, test_ts['Sales'], label='Testing Data')
plt.title('Time Series Data - Daily Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

# ARIMA Model
try:
    # Fit ARIMA model
    arima_model = ARIMA(train_ts, order=(5, 1, 2))
    arima_results = arima_model.fit()
    
    # Make predictions
    arima_forecast = arima_results.forecast(steps=len(test_ts))
    
    # Calculate metrics
    arima_rmse = np.sqrt(mean_squared_error(test_ts['Sales'], arima_forecast))
    arima_mae = mean_absolute_error(test_ts['Sales'], arima_forecast)
    
    print(f"ARIMA RMSE: {arima_rmse:.2f}")
    print(f"ARIMA MAE: {arima_mae:.2f}")
    
    # Plot ARIMA forecast
    plt.figure(figsize=(12, 6))
    plt.plot(train_ts.index, train_ts['Sales'], label='Training Data')
    plt.plot(test_ts.index, test_ts['Sales'], label='Actual Sales')
    plt.plot(test_ts.index, arima_forecast, label='ARIMA Forecast')
    plt.title('ARIMA Forecast vs Actual Sales')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.show()
except:
    print("ARIMA model could not be fitted. This might be due to insufficient data or convergence issues.")

# Prophet Model
try:
    # Prepare data for Prophet
    prophet_data = pd.DataFrame({
        'ds': ts_data.index,
        'y': ts_data['Sales']
    })
    
    # Split the data
    prophet_train = prophet_data.iloc[:train_size]
    prophet_test = prophet_data.iloc[train_size:]
    
    # Initialize and fit the model
    prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    prophet_model.fit(prophet_train)
    
    # Create a dataframe for future predictions
    future = prophet_model.make_future_dataframe(periods=len(prophet_test))
    
    # Make predictions
    forecast = prophet_model.predict(future)
    
    # Calculate metrics for the test period
    prophet_predictions = forecast.iloc[-len(prophet_test):]['yhat'].values
    prophet_rmse = np.sqrt(mean_squared_error(prophet_test['y'], prophet_predictions))
    prophet_mae = mean_absolute_error(prophet_test['y'], prophet_predictions)
    
    print(f"Prophet RMSE: {prophet_rmse:.2f}")
    print(f"Prophet MAE: {prophet_mae:.2f}")
    
    # Plot Prophet forecast components
    prophet_model.plot_components(forecast)
    plt.show()
    
    # Plot Prophet forecast vs actual
    plt.figure(figsize=(12, 6))
    plt.plot(prophet_train['ds'], prophet_train['y'], label='Training Data')
    plt.plot(prophet_test['ds'], prophet_test['y'], label='Actual Sales')
    plt.plot(prophet_test['ds'], prophet_predictions, label='Prophet Forecast')
    plt.title('Prophet Forecast vs Actual Sales')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.show()
except:
    print("Prophet model could not be fitted. This might be due to insufficient data or other issues.")

#6. Model Deployment Preparation
import pickle
import joblib

# Save the best model
best_model_file = f"{best_model_name.replace(' ', '_').lower()}_sales_prediction_model.pkl"
joblib.dump(best_model, best_model_file)
print(f"Model saved to {best_model_file}")

# Save the preprocessing pipeline components
# Note: In a real implementation, we would create a proper pipeline with sklearn.pipeline
preprocessing = {
    'categorical_features': categorical_features,
    'numerical_features': features_to_scale,
    'scaler': scaler
}
joblib.dump(preprocessing, 'preprocessing_components.pkl')
print("Preprocessing components saved to preprocessing_components.pkl")

# Function to make predictions on new data
def predict_sales(new_data, model_file, preprocessing_file):
    """
    Make sales predictions on new data.
    
    Parameters:
    - new_data: DataFrame with the same structure as the original data
    - model_file: Path to the saved model
    - preprocessing_file: Path to the saved preprocessing components
    
    Returns:
    - Predicted sales values
    """
    # Load model and preprocessing components
    model = joblib.load(model_file)
    preprocessing = joblib.load(preprocessing_file)
    
    # Preprocess the new data (simplified version)
    # In a real implementation, this would be a complete preprocessing pipeline
    # that handles all the transformations we applied during training
    
    # Make predictions
    predictions = model.predict(processed_new_data)
    
    return predictions

# Example of using the prediction function (pseudo-code)
# new_data = pd.read_csv('new_supermarket_data.csv')
# predictions = predict_sales(new_data, best_model_file, 'preprocessing_components.pkl')


