import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from xgboost import XGBRegressor
from kenpompy.utils import login
from kenpompy import summary
# Ensure the feature_selection module is correctly implemented or replace with actual imports
from feature_selection import (
    univariate_feature_selection
)


class BasketballPredictor:
    def __init__(self):
        self.model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.2,
            max_depth=3,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_alpha=0,
            reg_lambda=0
        )
        
        self.scaler = StandardScaler()
    
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train with fixed parameters"""
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        
        # Print feature importance
        print("\nFeature Importance:")
        importance = self.model.feature_importances_
        importance_dict = dict(zip(X_train.columns, importance))
        for col, imp in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True):
            print(f"{col}: {imp:.3f}")
        
        # Cross-validation
        scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        rmse_scores = (-scores) ** 0.5
        print(f"Cross-Validation RMSE: {rmse_scores.mean():.3f}")
    
    def evaluate(self, X, y):
        """
        Evaluate the model and print performance metrics.
        """
        try:
            # Scale the features
            X_scaled = self.scaler.transform(X)
            y_pred = self.model.predict(X_scaled)
            
            # Calculate R²
            r2 = r2_score(y, y_pred)
            
            # Calculate Adjusted R²
            n = len(y)  # Number of observations
            p = X.shape[1]  # Number of predictors
            if n <= p + 1:
                raise ValueError("Number of observations is too small for the number of predictors.")
            
            # Calculate RSS (Residual Sum of Squares)
            rss = np.sum((y - y_pred) ** 2)
            
            # Calculate AIC
            aic = n * np.log(rss / n) + 2 * p
            
            # Calculate BIC
            bic = n * np.log(rss / n) + p * np.log(n)
            
            # Calculate additional metrics
            mae = mean_absolute_error(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            
        
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            raise
    
    def predict(self, X):
        """
        Predict margin of victory
        Positive number means team1 wins by that many points
        Negative number means team2 wins by that many points
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

def sym_predict(model, x1, x2):
    """
    Symmetrize predictions to enforce antisymmetry.
    f_anti(x1, x2) = 0.5 * [f(x1, x2) - f(x2, x1)]
    """
    # Predict for (team1, team2)
    y_ab = model.predict(x1)
    if len(y_ab) != 1:
        raise ValueError(f"Expected a single prediction for x1, but got {len(y_ab)} predictions.")
    
    # Predict for (team2, team1)
    y_ba = model.predict(x2)
    if len(y_ba) != 1:
        raise ValueError(f"Expected a single prediction for x2, but got {len(y_ba)} predictions.")
    
    # Return the antisymmetric part
    return 0.5 * (y_ab[0] - y_ba[0])

def load_training_data():
    try:
        results = pd.read_csv('2024_season_results.csv')
        browser = login('willkimball8@gmail.com', 'dkzTrGWm1G')
        stats_df = get_all_stats(browser)
        
        # Normalize team names for consistency
        results['home_team'] = results['home_team'].str.lower().str.strip()
        results['away_team'] = results['away_team'].str.lower().str.strip()
        stats_df['Team'] = stats_df['Team'].str.lower().str.strip()
        
        TEAM_NAME_MAPPING = {
            'abil christian': 'abilene christian',
            'alabama state': 'alabama st.',
            'alcorn state': 'alcorn st.',
            'american u': 'american',
            'appalachian st': 'appalachian st.',
            'ar-pine bluff': 'arkansas pine bluff',
            'arizona state': 'arizona st.',
            'arkansas state': 'arkansas st.',
            'ball state': 'ball st.',
            'bethune-cookman': 'bethune cookman',
            'boise state': 'boise st.',
            'boston u': 'boston university',
            'c. carolina': 'coastal carolina',
            'cal': 'california',
            'cent arkansas': 'central arkansas',
            'cent conn st': 'central connecticut',
            'cent michigan': 'central michigan',
            'charleston so': 'charleston southern',
            'chicago state': 'chicago st.',
            'cleveland state': 'cleveland st.',
            'colorado state': 'colorado st.',
            'coppin state': 'coppin st.',
            'csu bakersfield': 'cal st. bakersfield',
            'csu fullerton': 'cal st. fullerton',
            'csu northridge': 'csun',
            'delaware state': 'delaware st.',
            'e illinois': 'eastern illinois',
            'e kentucky': 'eastern kentucky',
            'e michigan': 'eastern michigan',
            'e washington': 'eastern washington',
            'ecu': 'east carolina',
            'etsu': 'east tennessee st.',
            'fair dickinson': 'fairleigh dickinson',
            'fau': 'florida atlantic',
            'fgcu': 'florida gulf coast',
            'fort wayne': 'purdue fort wayne',
            'fresno state': 'fresno st.',
            'fsu': 'florida st.',
            'g washington': 'george washington',
            'ga southern': 'georgia southern',
            'gardner-webb': 'gardner webb',
            'georgia state': 'georgia st.',
            'grambling': 'grambling st.',
            "hawai'i": 'hawaii',
            'houston baptist': 'houston christian',
            'idaho state': 'idaho st.',
            'illinois state': 'illinois st.',
            'indiana state': 'indiana st.',
            'iowa state': 'iowa st.',
            'iupui': 'iu indy',
            'jackson state': 'jackson st.',
            'jacksonville st': 'jacksonville st.',
            'jmu': 'james madison',
            'kansas state': 'kansas st.',
            'kennesaw st': 'kennesaw st.',
            'kent state': 'kent st.',
            'la tech': 'louisiana tech',
            'lbsu': 'long beach st.',
            'loyola (md)': 'loyola md',
            'loyola mary': 'loyola marymount',
            'loyola-chicago': 'loyola chicago',
            'md-e shore': 'maryland eastern shore',
            'miami': 'miami fl',
            'miami (oh)': 'miami oh',
            'michigan state': 'michigan st.',
            'mid tennessee': 'middle tennessee',
            'miss st': 'mississippi st.',
            'miss valley st': 'mississippi valley st.',
            'missouri state': 'missouri st.',
            'montana state': 'montana st.',
            'morehead state': 'morehead st.',
            'morgan state': 'morgan st.',
            "mt st mary's": "mount st. mary's",
            'murray state': 'murray st.',
            'n arizona': 'northern arizona',
            'n colorado': 'northern colorado',
            'n illinois': 'northern illinois',
            'n kentucky': 'northern kentucky',
            'nc a&t': 'north carolina a&t',
            'nc central': 'north carolina central',
            'nc state': 'n.c. state',
            'new mexico st': 'new mexico st.',
            'norfolk state': 'norfolk st.',
            'north dakota st': 'north dakota st.',
            'northwestern st': 'northwestern st.',
            'oklahoma state': 'oklahoma st.',
            'ole miss': 'mississippi',
            'omaha': 'nebraska omaha',
            'oregon state': 'oregon st.',
            'osu': 'ohio st.',
            'penn state': 'penn st.',
            'pitt': 'pittsburgh',
            'portland state': 'portland st.',
            'pv a&m': 'prairie view a&m',
            's carolina st': 'south carolina st.',
            's illinois': 'southern illinois',
            'sacramento st': 'sacramento st.',
            "saint joe's": "saint joseph's",
            'sam houston': 'sam houston st.',
            'san diego state': 'san diego st.',
            'san jose state': 'san jose st.',
            'se louisiana': 'southeastern louisiana',
            'se missouri st': 'southeast missouri',
            'sf austin': 'stephen f. austin',
            'siu ed': 'siue',
            'south dakota st': 'south dakota st.',
            'st bonaventure': 'st. bonaventure',
            'st francis (pa)': 'saint francis',
            "st john's": "st. john's",
            "st peter's": "saint peter's",
            'tenn tech': 'tennessee tech',
            'tennessee st': 'tennessee st.',
            'texas a&m-cc': 'texas a&m corpus chris',
            'texas state': 'texas st.',
            'uconn': 'connecticut',
            'ucsb': 'uc santa barbara',
            'uic': 'illinois chicago',
            'ul lafayette': 'louisiana',
            'ul monroe': 'louisiana monroe',
            'umass': 'massachusetts',
            'umkc': 'kansas city',
            'unc': 'north carolina',
            'uncg': 'unc greensboro',
            'unh': 'new hampshire',
            'uri': 'rhode island',
            'usf': 'south florida',
            'ut martin': 'tennessee martin',
            'ut rio grande': 'ut rio grande valley',
            'utah state': 'utah st.',
            'uva': 'virginia',
            'w carolina': 'western carolina',
            'w illinois': 'western illinois',
            'w kentucky': 'western kentucky',
            'w michigan': 'western michigan',
            'washington st': 'washington st.',
            'weber state': 'weber st.',
            'wichita state': 'wichita st.',
            'wright state': 'wright st.',
            'youngstown st': 'youngstown st.'
        }

        results['home_team'] = results['home_team'].replace(TEAM_NAME_MAPPING)
        results['away_team'] = results['away_team'].replace(TEAM_NAME_MAPPING)
        
        # Merge stats with results
        merged = pd.merge(
            results,
            stats_df,
            left_on='home_team',
            right_on='Team',
            how='left'
        ).merge(
            stats_df,
            left_on='away_team',
            right_on='Team',
            how='left',
            suffixes=('_home', '_away')
        )
        
        # Feature selection

        # Use the actual column names from the data
        numeric_columns = [
            'Off. Efficiency-Adj',        # Overall offensive efficiency (adjusted)
            'Def. Efficiency-Adj',        # Overall defensive efficiency (adjusted)
            'Off. Efficiency-Adj.Rank',   # Ranking perspective on offensive efficiency
            'Def. Efficiency-Adj.Rank',   # Ranking perspective on defensive efficiency
            'Experience',                 # Team experience
            'Def-eFG%',                   # Effective field goal percentage allowed by the defense
            'Off-eFG%',                   # Effective field goal percentage for the offense
            '2P%_y',                      # Two-point shooting percentage for the opponent
            'Blk%_y',                     # Block percentage for the opponent
            'Off-TO%',                    # Turnover percentage for the offense
            'Avg. Poss Length-Defense',   # Average possession length for the defense
            'Continuity',                 # Team continuity
            'Stl%_x',                     # Steal percentage for the offense
            'EffHgt'                      # Effective height     
        ]
        
        for col in numeric_columns:
            stats_df[col] = pd.to_numeric(stats_df[col], errors='coerce')
        
        for col in numeric_columns:
            merged[f'{col}_home'] = pd.to_numeric(merged[f'{col}_home'], errors='coerce')
            merged[f'{col}_away'] = pd.to_numeric(merged[f'{col}_away'], errors='coerce')
        
        # Calculate feature differences
        for col in numeric_columns:
            merged[f'{col}_diff'] = merged[f'{col}_home'] - merged[f'{col}_away']
        
        # Select features and target
        feature_columns = [f'{col}_diff' for col in numeric_columns]
        X = merged[feature_columns]
        y = merged['home_score'] - merged['away_score']
        
        return X, y
        
    except Exception as e:
        print(f"Error in load_training_data: {str(e)}")
        raise


def predict_game(team1, team2, home_team):
    """
    Predict the outcome of a game between two teams using the symmetrizer.
    """
    try:
        browser = login('willkimball8@gmail.com', 'dkzTrGWm1G')
        stats_df = get_all_stats(browser)
        
        # Normalize team names
        stats_df['Team'] = stats_df['Team'].str.lower().str.strip()
        team1 = team1.lower().strip()
        team2 = team2.lower().strip()
        
        # Check if teams exist in stats_df
        if team1 not in stats_df['Team'].values:
            raise ValueError(f"Error: {team1} not found in stats_df.")
        if team2 not in stats_df['Team'].values:
            raise ValueError(f"Error: {team2} not found in stats_df.")
        
        # Get team stats
        team1_stats = stats_df[stats_df['Team'] == team1].iloc[0]
        team2_stats = stats_df[stats_df['Team'] == team2].iloc[0]
        
        # Define numeric_columns with correct column names
        numeric_columns = [
            'Off. Efficiency-Adj',        # Overall offensive efficiency (adjusted)
            'Def. Efficiency-Adj',        # Overall defensive efficiency (adjusted)
            'Off. Efficiency-Adj.Rank',   # Ranking perspective on offensive efficiency
            'Def. Efficiency-Adj.Rank',   # Ranking perspective on defensive efficiency
            'Experience',                 # Team experience
            'Def-eFG%',                   # Effective field goal percentage allowed by the defense
            'Off-eFG%',                   # Effective field goal percentage for the offense
            '2P%_y',                      # Two-point shooting percentage for the opponent
            'Blk%_y',                     # Block percentage for the opponent
            'Off-TO%',                    # Turnover percentage for the offense
            'Avg. Poss Length-Defense',   # Average possession length for the defense
            'Continuity',                 # Team continuity
            'Stl%_x',                     # Steal percentage for the offense
            'EffHgt'                      # Effective height
        ]
        
        # Create features for (team1, team2)
        features_1 = pd.DataFrame([{
            f'{col}_diff': float(team1_stats[col]) - float(team2_stats[col])
            for col in numeric_columns
        }])
        
        # Create features for (team2, team1)
        features_2 = pd.DataFrame([{
            f'{col}_diff': float(team2_stats[col]) - float(team1_stats[col])
            for col in numeric_columns
        }])
        
        # Load training data and train the model
        X, y = load_training_data()
        predictor = BasketballPredictor()
        predictor.train(X, y)
        
        # Use the symmetrizer to make predictions
        prediction = sym_predict(predictor, features_1, features_2)
        
        # Apply home court boost
        home_court_boost = 3  # Example fixed value for home court advantage
        if home_team == "team1":
            prediction += home_court_boost
        elif home_team == "team2":
            prediction -= home_court_boost
            
        return prediction
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise

def get_prediction(team1, team2, venue):
    """
    Get the prediction result for a game between two teams.
    """
    try:
        # Map venue to home_team
        home_team_map = {
            "Team 1 Home": "team1",
            "Team 2 Home": "team2",
            "Neutral Site": None
        }
        home_team = home_team_map[venue]
        
        # Load training data and train the model
        X, y = load_training_data()
        predictor = BasketballPredictor()
        predictor.train(X, y)
        
        # Make prediction
        prediction = predict_game(team1, team2, home_team=home_team)
        
        # Determine favored team and line
        if prediction > 0:
            favored_team = team1
            line = prediction
        else:
            favored_team = team2
            line = -prediction  # Make positive for display
        
        # Return the result
        return {
            "favored_team": favored_team,
            "line": line,
            "venue": venue
        }
    
    except Exception as e:
        raise RuntimeError(f"Error during prediction: {e}")

def prepare_game_data():
    """
    Prepare game data by mapping team IDs to school names and formatting for our model
    """
    # Read the data
    games_df = pd.read_csv('games_men_2024.csv')
    teams_df = pd.read_csv('teams_men.csv')
    
    # Create a dictionary to map team IDs to names
    team_map = dict(zip(teams_df['id'].astype(str), teams_df['name']))
    
    # Create new dataframe with mapped names
    results_df = pd.DataFrame({
        'home_team': games_df['home.id'].astype(str).map(team_map),
        'away_team': games_df['away.id'].astype(str).map(team_map),
        'home_score': games_df['home.score'],
        'away_score': games_df['away.score'],
        'neutral': games_df['neutral']
    })
    
    # Remove any games with missing team names
    results_df = results_df.dropna(subset=['home_team', 'away_team'])
    
    # Save the processed data
    results_df.to_csv('2024_season_results.csv', index=False)
    
    return results_df

def get_all_stats(browser):
    try:
        # Get different stat categories
        efficiency_stats = summary.get_efficiency(browser)
        factors_stats = summary.get_fourfactors(browser).drop(columns=['Conference'])
        pointdist_stats = summary.get_pointdist(browser).drop(columns=['Conference'])
        height_stats = summary.get_height(browser).drop(columns=['Conference'])
        team_stats = summary.get_teamstats(browser, defense=False).drop(columns=['Conference'])
        team_def_stats = summary.get_teamstats(browser, defense=True).drop(columns=['Conference'])
        # Convert relevant columns to numeric
        numeric_columns = [
            'Tempo-Adj', 'Off. Efficiency-Adj', 'Def. Efficiency-Adj',
            'Avg. Poss Length-Offense', 'Avg. Poss Length-Defense'
        ]
        for df in [efficiency_stats, factors_stats, pointdist_stats, height_stats, team_stats, team_def_stats]:
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        # Merge all stats
        all_stats = efficiency_stats.merge(factors_stats, on='Team', how='left')\
                                    .merge(pointdist_stats, on='Team', how='left')\
                                    .merge(height_stats, on='Team', how='left')\
                                    .merge(team_stats, on='Team', how='left')\
                                    .merge(team_def_stats, on='Team', how='left')
        
        return all_stats
        
    except Exception as e:
        print(f"\nError loading additional stats: {str(e)}")
        raise

def main():
    try:
        # Load and split training data
        print("Loading training data...")
        X, y = load_training_data()
        
        # Feature selection
        print("\nPerforming feature selection...")
        feature_scores = univariate_feature_selection(X, y, method='f_regression', k=10)
        print("Univariate feature selection completed.")
        
        # Split data into training and validation sets
        print("\nSplitting data into training and validation sets...")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize and train the model
        print("\nInitializing and training the model...")
        predictor = BasketballPredictor()
        predictor.train(X_train, y_train)
        
        # Test predictions
        print("\nRunning test predictions...")
        team1 = "alabama"  # Replace with a valid team name
        team2 = "auburn"  # Replace with a valid team name
        
        # Test team1 home
        print("\nTesting Team1 Home...")
        prediction_team1_home = predict_game(team1, team2, home_team="team1")
        favored_team1_home = team1 if prediction_team1_home > 0 else team2
        print(f"Prediction (Team1 Home): {prediction_team1_home:.2f} - Favored Team: {favored_team1_home}")
        
        # Test team2 home
        print("\nTesting Team2 Home...")
        prediction_team2_home = predict_game(team1, team2, home_team="team2")
        favored_team2_home = team1 if prediction_team2_home > 0 else team2
        print(f"Prediction (Team2 Home): {prediction_team2_home:.2f} - Favored Team: {favored_team2_home}")
        
        # Test neutral site
        print("\nTesting Neutral Site...")
        prediction_neutral = predict_game(team1, team2, home_team=None)
        favored_neutral = team1 if prediction_neutral > 0 else team2
        print(f"Prediction (Neutral Site): {prediction_neutral:.2f} - Favored Team: {favored_neutral}")
        
        # Calculate and print MSE on validation set
        
        y_pred = predictor.predict(X_val)  # Get predictions for validation set
        mse = mean_squared_error(y_val, y_pred)  # Calculate MSE
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, y_pred)
        
        # Calculate Adjusted R²
        n = len(y_val)  # Number of observations
        p = X_val.shape[1]  # Number of predictors
        adjusted_r2 = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)
        
        # Print performance metrics
        print("\nPerformance Metrics:")
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        print(f"R-Squared (R²): {r2:.2f}")
        print(f"Adjusted R-Squared: {adjusted_r2:.2f}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

