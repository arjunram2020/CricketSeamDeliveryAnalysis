import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
df = pd.read_csv('/Users/arjunramakrishnan/Downloads/Analysis Doc.csv')
columns_to_use = df.columns.tolist()[1:]
for col in columns_to_use:
    df[col] = df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True)
for col in columns_to_use:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.dropna(subset=columns_to_use, inplace=True)
df_numeric = df[columns_to_use].copy()
X = df_numeric.drop('Ball Speed', axis=1)
y = df_numeric['Ball Speed']



#MLR LASSO MODEL
# Build a lasso regression model and fit it on X and y
lasso_model = Lasso(alpha=0.1)  # You can adjust the alpha parameter
lasso_model.fit(X, y)
y_pred_lasso = lasso_model.predict(X)
rmse_lasso = np.sqrt(mean_squared_error(y, y_pred_lasso))
r2_lasso = r2_score(y, y_pred_lasso)
print(f'Lasso Regression - RMSE: {rmse_lasso:.2f}, R-squared: {r2_lasso:.2f}')



#MLR RIDGE MODEL
ridge_model = Ridge(alpha=0.1)  # You can adjust the alpha parameter
ridge_model.fit(X, y)
y_pred_ridge = ridge_model.predict(X)
rmse_ridge = np.sqrt(mean_squared_error(y, y_pred_ridge))
r2_ridge = r2_score(y, y_pred_ridge)
print(f'Ridge Regression - RMSE: {rmse_ridge:.2f}, R-squared: {r2_ridge:.2f}')


#MLR ELASTICNET MODEL
elasticnet_model = ElasticNet(alpha=0.1)
elasticnet_model.fit(X, y)
y_pred_elasticnet = elasticnet_model.predict(X)
rmse_elasticnet = np.sqrt(mean_squared_error(y, y_pred_elasticnet))
r2_elasticnet = r2_score(y, y_pred_elasticnet)
print(f'ElasticNet Regression - RMSE: {rmse_elasticnet:.2f}, R-squared: {r2_elasticnet:.2f}')


#GRADIENT BOOSTING MODEL
# Build a Gradient Boosting Regressor model
gb_model = GradientBoostingRegressor(random_state=0)  # You can adjust hyperparameters if needed

# Fit the model
gb_model.fit(X, y)

# Make predictions
y_pred_gb = gb_model.predict(X)

# Calculate and print RMSE and R-squared
rmse_gb = np.sqrt(mean_squared_error(y, y_pred_gb))
r2_gb = r2_score(y, y_pred_gb)
print(f'Gradient Boosting Regressor - RMSE: {rmse_gb:.2f}, R-squared: {r2_gb:.2f}')



#Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
rf_model.fit(X, y)
y_pred_rf = rf_model.predict(X)
rmse_rf = np.sqrt(mean_squared_error(y, y_pred_rf))
r2_rf = r2_score(y, y_pred_rf)
print(f'Random Forest Regressor - RMSE: {rmse_rf:.2f}, R-squared: {r2_rf:.2f}')




# PCA INITIALIZATION
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X)
pca_model = GradientBoostingRegressor(random_state=0)
pca_model.fit(X_pca, y)
y_pred_pca = pca_model.predict(X_pca)
rmse_pca = np.sqrt(mean_squared_error(y, y_pred_pca))
r2_pca = r2_score(y, y_pred_pca)
print(f'PCA with Gradient Boosting - RMSE: {rmse_pca:.2f}, R-squared: {r2_pca:.2f}')





#PLSRegression
pls_model = PLSRegression(n_components=5)
pls_model.fit(X, y)
y_pred_pls = pls_model.predict(X)
rmse_pls = np.sqrt(mean_squared_error(y, y_pred_pls))
r2_pls = r2_score(y, y_pred_pls)
print(f'PLS Regression - RMSE: {rmse_pls:.2f}, R-squared: {r2_pls:.2f}')



#Testing for most important factors of ball speed - Table
importances = gb_model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print(importance_df.round(3).to_markdown(index=False))



#Testing for Direction of values of data points for each variable.
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)
coefficients = model.coef_
coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': coefficients})
coef_df = coef_df.sort_values(by='Coefficient', ascending=False)
print(coef_df.round(3).to_markdown(index=False))


#Optimal Data Points for each variable
features = X.columns
X_mean = X.copy()
X_mean[:] = X_mean.mean()
for feature in features:
    values = np.linspace(X[feature].min(), X[feature].max(), 25)
    X_mean[feature] = values
    predictions = gb_model.predict(X_mean)
    optimal_index = np.argmax(predictions)
    print(f"Feature: {feature}, Optimal Value: {values[optimal_index]:.2f}, Predicted Ball Speed: {predictions[optimal_index]:.2f}")
    X_mean[feature] = X[feature].mean()



    import seaborn as sns
import matplotlib.pyplot as plt

# EDA: Visualize the distribution of the target variable
plt.figure(figsize=(10, 5))
sns.histplot(df['Ball Speed'], bins=30, kde=True)
plt.title('Distribution of Ball Speed')
plt.xlabel('Ball Speed')
plt.ylabel('Frequency')
plt.show()

# EDA: Correlation heatmap
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Heatmap')
plt.show()