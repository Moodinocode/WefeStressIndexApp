import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

# Define the file name for the preprocessed data
INPUT_FILE = 'Litani_WEFE_Stressed_Data.csv'

def run_random_forest():
    """
    Loads the WEFE data and performs a Random Forest Regression
    to predict the wefe_stress_index based on the four input features.
    Evaluates performance and prints feature importances.
    """

    # 1. Check if the file exists
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file '{INPUT_FILE}' not found.")
        print("Please ensure you have run the data processing script to generate this file.")
        return

    # 2. Load the data
    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    # 3. Define Features (X) and Target (y)
    feature_cols = ['ndvi', 'ndti', 'rainfall', 'viirs_proxy']
    target_col = 'wefe_stress_index'

    X = df[feature_cols]
    y = df[target_col]

    print(f"Features (X) selected: {feature_cols}")
    print(f"Target (y) selected: {target_col}")

    # 4. Split Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\nTraining set size: {len(X_train)} samples (80%)")
    print(f"Testing set size: {len(X_test)} samples (20%)")

    # 5. Initialize and Train the Random Forest Model
    print("\nTraining Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 6. Predict on Test Set
    y_pred = model.predict(X_test)

    # 7. Evaluate Performance
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n--- Random Forest Performance Evaluation ---")
    print(f"R-squared (Accuracy): {r2:.4f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")

    # 8. Feature Importances
    print("\nFeature Importances (Impact on Stress Index Prediction):")
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    print(importances.sort_values(ascending=False))

    

    # --- 9. 5-Fold Cross-Validation for Random Forest ---
    print("\nPerforming 5-Fold Cross-Validation for robustness...")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_r2 = cross_val_score(model, X, y, cv=kf, scoring='r2')
    cv_rmse = np.sqrt(-cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error'))

    print("5-Fold CV R² scores:", cv_r2)
    print("Average CV R²:", cv_r2.mean())

    print("\n5-Fold CV RMSE scores:", cv_rmse)
    print("Average CV RMSE:", cv_rmse.mean())


if __name__ == '__main__':
    run_random_forest()