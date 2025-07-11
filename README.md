# Supermarket-Sales-Prediction
Sales forecasting, customer behavior analysis, and performance metrics
# Supermarket Sales Analysis and Prediction

## Overview
This project analyzes supermarket sales data to identify patterns, extract insights, and build predictive models. It includes comprehensive data exploration, preprocessing, feature engineering, and customer segmentation to understand sales drivers and customer behavior.

## Data
The analysis uses [**Supermarket Sales**](https://www.kaggle.com/datasets/faresashraf1001/supermarket-sales) dataset containing transaction records with features such as:
- Branch and city information
- Customer demographics (gender, type)
- Product categories and pricing
- Payment methods
- Ratings and timestamps
- Sales and revenue metrics

## Models Implemented
- Multiple regression models (Linear, Ridge, Lasso)
- Tree-based models (Decision Tree, Random Forest)
- Ensemble methods (Gradient Boosting, XGBoost, LightGBM)
- Customer segmentation using KMeans clustering
- Time series forecasting models

## Features

### Data Exploration
- Dataset overview (shape, data types, statistical summary)
- Missing value and duplicate detection
- Datetime conversion and component extraction
- Distribution analysis of key variables
- Correlation analysis between numerical features
- Time series analysis of sales patterns

### Data Preprocessing
- Outlier detection and handling using IQR method
- Categorical feature encoding using one-hot encoding
- Numerical feature scaling with StandardScaler
- Date and time feature decomposition

### Feature Engineering
- Transaction-based features (items_per_invoice, avg_item_price, sales_per_unit)
- Cyclical time encoding (sin/cos transformations for month, day, hour)
- Aggregated features by product line and branch
- Derived features based on business logic

### Visualization
- Sales distribution analysis
- Comparative analysis across branches, product lines, and customer segments
- Temporal patterns (daily, weekly, monthly sales trends)
- Correlation heatmaps and relationship scatter plots
- Time series decomposition (trend, seasonality, residuals)
- Feature distribution before and after preprocessing

## Results

### Model Comparison

| Model               | RMSE      | MAE       | R2        | CV RMSE    | Training Time | Prediction Time |
|---------------------|-----------|-----------|-----------|------------|--------------|----------------|
| XGBoost             | 9.076215  | 7.035159  | 0.998734  | 11.511004  | 2.270823     | 0.048319       |
| LightGBM            | 12.599371 | 9.120925  | 0.997560  | 15.168484  | 0.108925     | 0.003448       |
| Random Forest       | 13.015120 | 8.768198  | 0.997396  | 16.258552  | 0.978690     | 0.013969       |
| Gradient Boosting   | 18.048330 | 13.739020 | 0.994993  | 16.715247  | 0.326115     | 0.002525       |
| Decision Tree       | 19.545638 | 10.873800 | 0.994128  | 23.525335  | 0.015406     | 0.001775       |
| Lasso Regression    | 65.722535 | 51.251902 | 0.933607  | 65.205810  | 0.031166     | 0.002102       |
| Ridge Regression    | 67.831611 | 52.777647 | 0.929278  | 64.007353  | 0.063455     | 0.002432       |
| Linear Regression   | 68.265567 | 53.136916 | 0.928370  | 64.009134  | 0.017830     | 0.002343       |

### Best Model
XGBoost significantly outperformed other models with the lowest RMSE (9.08) and highest R² (0.998734). After hyperparameter tuning, the model performance improved further:

- **Best parameters:** {'colsample_bytree': 0.9, 'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 300, 'subsample': 0.8}
- **Tuned model RMSE:** 6.64
- **Tuned model MAE:** 4.85
- **Tuned model R²:** 1.00

### Feature Importance
While specific feature importance values aren't provided in the code snippet, the XGBoost model typically identifies key sales drivers such as:
- Product line category
- Time-based features (particularly hour of day and day of week)
- Branch location
- Customer type and payment method

## Outcome

### Best Performing Model
XGBoost demonstrated superior predictive performance for sales forecasting, achieving near-perfect R² score after tuning. The model's exceptional accuracy enables reliable sales predictions while maintaining reasonable computational efficiency.

The analysis reveals valuable insights about:
- Sales patterns across different product lines, with some categories showing significantly higher average sales
- Temporal trends including daily, weekly, and monthly sales patterns
- Customer behavior differences between member and normal customer types
- Potential segmentation of customers based on purchasing behavior

## Future Work
- Implement additional ensemble methods and deep learning approaches
- Develop real-time prediction API for sales forecasting
- Create recommendation systems based on product affinity
- Implement interactive dashboards for real-time sales monitoring
- Deploy models for automated inventory management
- Expand analysis with additional external factors (seasonality, promotions, market conditions)

## Notes
- The dataset shows no missing values, which is uncommon in real-world scenarios
- Outlier treatment was performed using the capping method to preserve data points
- Cyclical features were encoded using trigonometric transformations to preserve their circular nature
- XGBoost's superior performance suggests complex non-linear relationships in the data

## Contributing
1. 

## License
This project is licensed under the MIT License - see the LICENSE file for details.
