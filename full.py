import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from keras.models import Sequential
from keras.layers import Dense

# Load the dataset
data = pd.read_csv('/mnt/data/data_clean.csv')

# Preprocess the data
# Use pd.get_dummies to encode the 'industry' column and any other categorical columns
data = pd.get_dummies(data, columns=['industry','country'], drop_first=True)

# Encoding the 'stage_grouped' column if necessary
if data['stage_grouped'].dtype == 'object':
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    data['stage_grouped'] = label_encoder.fit_transform(data['stage_grouped'])

# Drop any unnecessary columns
columns_to_drop = ['customer_number','stage_grouped','max_closed_date','cloud_revenue','total_opp_amount']

# Feature and target split for classification
X_class = data.drop(columns=[columns_to_drop])
y_class = data['stage_grouped']

# Feature and target split for regression
X_reg = data.drop(columns=[columns_to_drop])
y_reg = data['total_opp_amount']

# Standardize the features
scaler = StandardScaler()
X_class = scaler.fit_transform(X_class)
X_reg = scaler.fit_transform(X_reg)

# Split the data
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

### 1. Correlation Analysis
print("Correlation Analysis:")
correlation_matrix = pd.DataFrame(X_class, columns=data.drop(columns=[columns_to_drop]).columns).corrwith(pd.Series(y_class)).abs()
print(correlation_matrix.sort_values(ascending=False))

# Plot correlation heatmap for top features
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix.values.reshape(-1, 1), annot=True, cmap='coolwarm', cbar=True)
plt.title("Feature Correlation with Target (Classification)")
plt.show()

### 2. Recursive Feature Elimination (RFE)
print("Recursive Feature Elimination (RFE):")
rfe_classifier = RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=10)
X_train_rfe = rfe_classifier.fit_transform(X_train_class, y_train_class)
X_test_rfe = rfe_classifier.transform(X_test_class)
print(f"Selected Features: {rfe_classifier.support_}")

### 3. Feature Importance using Random Forest
print("Feature Importance from Random Forest:")
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train_class, y_train_class)
importances = rf_classifier.feature_importances_

# Plot feature importance
plt.figure(figsize=(10, 8))
sns.barplot(x=importances, y=data.drop(columns=['stage_grouped', 'total_opp_amount']).columns)
plt.title("Feature Importance (Classification)")
plt.show()

### 4. PCA for Dimensionality Reduction
print("Principal Component Analysis (PCA):")
pca = PCA(n_components=0.95)  # Keep 95% of the variance
X_train_pca = pca.fit_transform(X_train_class)
X_test_pca = pca.transform(X_test_class)
print(f"Number of components selected: {pca.n_components_}")

# Hyperparameter Tuning for Classification Models
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

param_grid = {
    'Logistic Regression': {'C': [0.01, 0.1, 1, 10, 100]},
    'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]},
    'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    'K-Nearest Neighbors': {'n_neighbors': [3, 5, 7, 9]}
}

print("\nClassification Results with Hyperparameter Tuning:")
for name, model in classifiers.items():
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid[name], scoring='accuracy', cv=5)
    grid_search.fit(X_train_rfe, y_train_class)  # Using RFE selected features
    best_model = grid_search.best_estimator_
    y_pred_class = best_model.predict(X_test_rfe)
    y_prob_class = best_model.predict_proba(X_test_rfe)[:, 1]
    
    # Calculate AUC
    auc = roc_auc_score(y_test_class, y_prob_class)
    print(f"{name} Best Parameters: {grid_search.best_params_}")
    print(f"{name} Accuracy: {accuracy_score(y_test_class, y_pred_class):.4f}")
    print(f"{name} AUC: {auc:.4f}")
    print(classification_report(y_test_class, y_pred_class))
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test_class, y_pred_class)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Hyperparameter Tuning for Regression Models
regressors = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
    'SVR': SVR(),
    'K-Nearest Neighbors': KNeighborsRegressor()
}

param_grid_reg = {
    'Linear Regression': {},  # No hyperparameters to tune
    'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]},
    'SVR': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    'K-Nearest Neighbors': {'n_neighbors': [3, 5, 7, 9]}
}

print("\nRegression Results with Hyperparameter Tuning:")
for name, model in regressors.items():
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid_reg[name], scoring='r2', cv=5)
    grid_search.fit(X_train_reg, y_train_reg)
    best_model = grid_search.best_estimator_
    y_pred_reg = best_model.predict(X_test_reg)
    print(f"{name} Best Parameters: {grid_search.best_params_}")
    print(f"{name} MSE: {mean_squared_error(y_test_reg, y_pred_reg):.4f}")
    print(f"{name} R2 Score: {r2_score(y_test_reg, y_pred_reg):.4f}")

# Deep Learning Model for Classification
print("\nDeep Learning Classification:")
model_class = Sequential()
model_class.add(Dense(64, input_dim=X_train_rfe.shape[1], activation='relu'))
model_class.add(Dense(32, activation='relu'))
model_class.add(Dense(1, activation='sigmoid'))
model_class.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_class.fit(X_train_rfe, y_train_class, epochs=50, batch_size=10, verbose=1)

# Evaluate the classification model
loss, accuracy = model_class.evaluate(X_test_rfe, y_test_class)
print(f"Deep Learning Classification Accuracy: {accuracy:.4f}")

# Deep Learning Model for Regression
print("\nDeep Learning Regression:")
model_reg = Sequential()
model_reg.add(Dense(64, input_dim=X_train_reg.shape[1], activation='relu'))
model_reg.add(Dense(32, activation='relu'))
model_reg.add(Dense(1))
model_reg.compile(optimizer='adam', loss='mean_squared_error')
model_reg.fit(X_train_reg, y_train_reg, epochs=50, batch_size=10, verbose=1)

# Evaluate the regression model
y_pred_reg_dl = model_reg.predict(X_test_reg).flatten()
print(f"Deep Learning Regression MSE: {mean_squared_error(y_test_reg, y_pred_reg_dl):.4f}")
print(f"Deep Learning Regression R2 Score: {r2_score(y_test_reg, y_pred_reg_dl):.4f}")
