import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from kenpompy.utils import login
from kenpompy import summary


class BasketballPredictor:
    def __init__(self):
        self.model = XGBRegressor(
            n_estimators=200,          # Fewer trees
            learning_rate=0.2,         # Higher learning rate
            max_depth=3,               # Simpler trees
            subsample=1.0,             # Use all data
            colsample_bytree=1.0,      # Use all features
            reg_alpha=0,               # No regularization
            reg_lambda=0               # No regularization
        )
        self.scaler = StandardScaler()
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train with fixed parameters"""
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        
        # Print feature importance with better formatting
        print("\nFeature Importance:")
        importance = self.model.feature_importances_
        importance_dict = dict(zip(X_train.columns, importance))
        # Sort by importance
        for col, imp in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True):
            print(f"{col}: {imp:.3f}")
    
    def predict(self, X):
        """
        Predict margin of victory
        Positive number means team1 wins by that many points
        Negative number means team2 wins by that many points
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

def load_training_data():
    """
    Load and preprocess 2024 season data with additional KenPom metrics
    """
    try:
        results = pd.read_csv('2024_season_results.csv')
        browser = login('willkimball8@gmail.com', 'dkzTrGWm1G')
        stats_df = get_all_stats(browser)
        
        # Use the actual column names from the data
        numeric_columns = [
            'Tempo-Adj',
            'Off. Efficiency-Adj',
            'Def. Efficiency-Adj',
            'Avg. Poss Length-Offense',
            'Avg. Poss Length-Defense'
        ]
        
        # Clean column names
        stats_df.columns = stats_df.columns.str.replace(' ', '_').str.replace('.', '').str.lower()
        numeric_columns = [col.lower().replace(' ', '_').replace('.', '') for col in numeric_columns]
        
        for col in numeric_columns:
            stats_df[col] = pd.to_numeric(stats_df[col], errors='coerce')
        
        # Merge stats with results
        merged = pd.merge(
            results,
            stats_df,
            left_on='home_team',
            right_on='team',
            how='left'
        ).merge(
            stats_df,
            left_on='away_team',
            right_on='team',
            how='left',
            suffixes=('_home', '_away')
        )
        
        # Calculate feature differences
        features = {}
        for col in numeric_columns:
            merged[f'{col}_diff'] = merged[f'{col}_home'] - merged[f'{col}_away']
            
        # Add home court advantage
        merged['home_court_advantage'] = merged['neutral'].map({0: 1, 1: 0})
        
        # Select features and target
        feature_columns = [f'{col}_diff' for col in numeric_columns] + ['home_court_advantage']
        X = merged[feature_columns]
        y = merged['home_score'] - merged['away_score']
        
        return X, y
        
    except Exception as e:
        print(f"Error in load_training_data: {str(e)}")
        raise


def predict_game(team1, team2, home_team):
    """
    Predict the outcome of a game between two teams.
    """
    try:
        browser = login('willkimball8@gmail.com', 'dkzTrGWm1G')
        stats_df = get_all_stats(browser)
        
        # Use the same column names as in load_training_data
        numeric_columns = [
            'Tempo-Adj',
            'Off. Efficiency-Adj',
            'Def. Efficiency-Adj',
            'Avg. Poss Length-Offense',
            'Avg. Poss Length-Defense'
        ]
        
        # Clean column names
        stats_df.columns = stats_df.columns.str.replace(' ', '_').str.replace('.', '').str.lower()
        numeric_columns = [col.lower().replace(' ', '_').replace('.', '') for col in numeric_columns]
        
        # Get team stats
        team1_stats = stats_df[stats_df['team'] == team1].iloc[0]
        team2_stats = stats_df[stats_df['team'] == team2].iloc[0]
        
        # Print stats using correct column names
        print(f"\nKey Stats:")
        print(f"\n{team1}:")
        print(f"Adjusted Tempo: {float(team1_stats['tempo-adj']):.1f}")
        print(f"Adjusted Off. Efficiency: {float(team1_stats['off_efficiency-adj']):.1f}")
        print(f"Adjusted Def. Efficiency: {float(team1_stats['def_efficiency-adj']):.1f}")
        print(f"Off. Possession Length: {float(team1_stats['avg_poss_length-offense']):.1f}")
        print(f"Def. Possession Length: {float(team1_stats['avg_poss_length-defense']):.1f}")
        
        print(f"\n{team2}:")
        print(f"Adjusted Tempo: {float(team2_stats['tempo-adj']):.1f}")
        print(f"Adjusted Off. Efficiency: {float(team2_stats['off_efficiency-adj']):.1f}")
        print(f"Adjusted Def. Efficiency: {float(team2_stats['def_efficiency-adj']):.1f}")
        print(f"Off. Possession Length: {float(team2_stats['avg_poss_length-offense']):.1f}")
        print(f"Def. Possession Length: {float(team2_stats['avg_poss_length-defense']):.1f}")
        
        # Create features DataFrame
        features = pd.DataFrame([{
            f'{col}_diff': float(team1_stats[col]) - float(team2_stats[col])
            for col in numeric_columns
        }])
        
        # Add home court advantage
        if home_team is None:  # Neutral site
            features['home_court_advantage'] = 0
            venue_type = "neutral"
        elif home_team == "team1":
            features['home_court_advantage'] = 1
            venue_type = "team1_home"
        else:
            features['home_court_advantage'] = -1
            venue_type = "team2_home"
        
        # Make prediction
        predictor = BasketballPredictor()
        X, y = load_training_data()
        predictor.train(X, y)
        prediction = predictor.predict(features)[0]
        
        # Show who's favored
        print(f"\nModel Prediction:")
        if prediction > 0:
            favored_team = team1
            line = prediction
        else:
            favored_team = team2
            line = -prediction
            
        print(f"{favored_team} favored by {line:.1f} points")
        
        # Show betting line (from perspective of favored team)
        venue_text = "(neutral)" if venue_type == "neutral" else ""
        print(f"Betting line {venue_text}: {favored_team} -{line:.1f}")
        
        return prediction
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise

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
    print(f"Processed {len(results_df)} games")
    
    return results_df

def get_all_stats(browser):
    """Get comprehensive stats from KenPom"""
    # Get different stat categories
    efficiency_stats = summary.get_efficiency(browser)
    
    try:
        # Get Four Factors stats
        factors_stats = summary.get_fourfactors(browser)  # From stats.php
        print("\nFour Factors columns:")
        print(factors_stats.columns.tolist())
        
        # Get Point Distribution stats
        pointdist_stats = summary.get_pointdist(browser)  # From pointdist.php
        print("\nPoint Distribution columns:")
        print(pointdist_stats.columns.tolist())
        
        # Get Height/Experience stats
        height_stats = summary.get_height(browser)  # From height.php
        print("\nHeight/Experience columns:")
        print(height_stats.columns.tolist())
        
        # Get Team Stats (both offensive and defensive)
        team_stats = summary.get_teamstats(browser, defense=False)  # From teamstats.php
        team_def_stats = summary.get_teamstats(browser, defense=True)
        print("\nTeam Stats columns:")
        print(team_stats.columns.tolist())
        
        # Merge all stats
        all_stats = efficiency_stats.merge(factors_stats, on='Team', how='left')\
                                  .merge(pointdist_stats, on='Team', how='left')\
                                  .merge(height_stats, on='Team', how='left')\
                                  .merge(team_stats, on='Team', how='left')\
                                  .merge(team_def_stats, on='Team', how='left')
        
        return all_stats
        
    except Exception as e:
        print(f"\nError loading additional stats: {str(e)}")
        return efficiency_stats

def main():
    """
    Main function to train and evaluate the model
    """
    try:
        # Prepare the game data first
        # prepare_game_data()  # Commented out since data is already prepared
        
        # Load and split training data
        X, y = load_training_data()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize and train model
        predictor = BasketballPredictor()
        predictor.train(X_train, y_train, X_test, y_test)  # Added validation data
        
        # Evaluate model
        y_pred = predictor.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print("\nModel Evaluation:")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"Average Prediction Error: ±{mae:.1f} points")
        
        # Example prediction
        print("\nExample Prediction:")
        prediction = predict_game(
            team1="Michigan St.",
            team2="Michigan",
            home_team="team1"
        )
        print(f"Predicted margin: {prediction:.1f} points")
        
    except FileNotFoundError:
        print("Error: Could not find season_results.csv. Please ensure the file exists.")
    except ValueError as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main() 
