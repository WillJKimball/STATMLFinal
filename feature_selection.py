import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_regression, mutual_info_regression, SelectKBest, RFE
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

def exploratory_analysis(X):
    """
    Perform exploratory analysis on the feature set.
    """
    print("\nExploratory Analysis:")
    
    # Check for missing values
    missing_percent = X.isnull().mean() * 100
    print("Missing Values (%):")
    print(missing_percent[missing_percent > 0].sort_values(ascending=False))
    
    # Plot histograms for numeric features
    X.hist(bins=20, figsize=(15, 10))
    plt.suptitle("Feature Distributions")
    plt.show()
    
    # Domain sanity check (manual step)
    print("\nDomain Sanity Check:")
    print("Review features for relevance to basketball knowledge (e.g., adjusted efficiencies, tempo, etc.).")

def univariate_feature_selection(X, y, method='f_regression', k=10):
    """
    Perform univariate feature selection.
    """
    if method == 'f_regression':
        skb = SelectKBest(f_regression, k=k).fit(X, y)
        scores = skb.scores_
    elif method == 'mutual_info':
        scores = mutual_info_regression(X, y)
    else:
        raise ValueError("Invalid method. Choose 'f_regression' or 'mutual_info'.")
    
    feature_scores = pd.DataFrame({
        'Feature': X.columns,
        'Score': scores
    }).sort_values(by='Score', ascending=False)
    
    print("\nTop 15 Features by Importance:")
    print(feature_scores.head(45))
    
    return feature_scores

def recursive_feature_elimination(X, y, n_features=10):
    """
    Perform Recursive Feature Elimination (RFE).
    """
    selector = RFE(Ridge(), n_features_to_select=n_features).fit(X, y)
    selected_features = X.columns[selector.support_]
    print("\nRecursive Feature Elimination:")
    print(f"Selected Features: {list(selected_features)}")
    return selected_features

def tree_based_feature_importance(X, y):
    """
    Use tree-based models to compute feature importance.
    """
    model = XGBRegressor()
    model.fit(X, y)
    importance = model.feature_importances_
    
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    
    print("\nTree-Based Feature Importance:")
    print(feature_importance)
    return feature_importance

def time_aware_cross_validation(X, y, model, n_splits=5):
    """
    Perform time-aware cross-validation.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    errors = []
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        error = mean_squared_error(y_test, y_pred, squared=False)  # RMSE
        errors.append(error)
    
    print("\nTime-Aware Cross-Validation:")
    print(f"Average RMSE: {sum(errors) / len(errors):.2f}")
    return errors

def stability_check(X, y, model, n_repeats=5):
    """
    Check feature stability across multiple CV folds.
    """
    selected_features = []
    for _ in range(n_repeats):
        selected_features.append(recursive_feature_elimination(X, y, n_features=10))
    
    # Count how often each feature is selected
    feature_counts = pd.Series([f for sublist in selected_features for f in sublist]).value_counts()
    print("\nFeature Stability Check:")
    print(feature_counts)
    return feature_counts

class BasketballPredictor:
    def __init__(self):
        self.model = HistGradientBoostingRegressor(
            max_iter=200,         # Number of boosting iterations
            learning_rate=0.1,    # Learning rate
            max_depth=3,          # Maximum depth of trees
            random_state=42       # For reproducibility
        )
        self.scaler = None  # No need for scaling with HistGradientBoostingRegressor
    
    def train(self, X_train, y_train):
        """Train the model"""
        self.model.fit(X_train, y_train)
        
        # Print feature importance
        print("\nFeature Importance:")
        from sklearn.inspection import permutation_importance
        
        result = permutation_importance(self.model, X_train, y_train, n_repeats=10, random_state=42)
        importance_dict = dict(zip(X_train.columns, result.importances_mean))
        print("\nFeature Importance (via Permutation Importance):")
        for col, imp in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True):
            print(f"{col}: {imp:.3f}")
    
    def predict(self, X):
        """Predict margin of victory"""
        return self.model.predict(X)