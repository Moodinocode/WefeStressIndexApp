import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

# Define the file name for the preprocessed data
INPUT_FILE = 'Litani_WEFE_Stressed_Data.csv'

def run_linear_regression():
    """
    Loads the WEFE data, performs Linear Regression to predict the 
    wefe_stress_index based on the four input features, evaluates performance,
    and performs 5-fold cross-validation for robustness.
    """
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file '{INPUT_FILE}' not found.")
        print("Please ensure you have run the data processing script to generate this file.")
        return

    # 1. Load the preprocessed data
    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    # 2. Define Features (X) and Target (y)
    feature_cols = ['ndvi', 'ndti', 'rainfall', 'viirs_proxy']
    target_col = 'wefe_stress_index'
    
    X = df[feature_cols]
    y = df[target_col]
    
    print(f"Features (X) selected: {feature_cols}")
    print(f"Target (y) selected: {target_col}")

    # 3. Split Data into Training (80%) and Testing (20%) Sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\nTraining set size: {len(X_train)} samples (80%)")
    print(f"Testing set size: {len(X_test)} samples (20%)")

    # 4. Initialize and Train the Linear Regression Model
    model = LinearRegression()
    print("\nTraining Linear Regression model on training set...")
    model.fit(X_train, y_train)

    # 5. Make Predictions on the Test Set
    y_pred = model.predict(X_test)

    # 6. Evaluate Model Performance on Test Set
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n--- Test Set Performance Evaluation ---")
    print(f"R-squared (Accuracy): {r2:.4f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")

    # 7. 5-Fold Cross-Validation on Full Dataset
    print("\nPerforming 5-Fold Cross-Validation on full dataset for robustness...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2 = cross_val_score(model, X, y, cv=kf, scoring='r2')
    cv_rmse = np.sqrt(-cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error'))

    print("5-Fold CV R² scores:", cv_r2)
    print("Average CV R²:", cv_r2.mean())
    print("5-Fold CV RMSE:", cv_rmse)
    print("Average CV RMSE:", cv_rmse.mean())

    # 8. Print Model Coefficients (Feature Importance)
    print("\nModel Coefficients (Feature Contribution to Stress Index):")
    coefficients = pd.Series(model.coef_, index=feature_cols)
    print(coefficients.sort_values(ascending=False))
    
    


if __name__ == '__main__':
    run_linear_regression()