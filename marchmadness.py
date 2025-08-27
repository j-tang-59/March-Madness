import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    
    team_games = []
    
    for _, game in df.iterrows():
        # Home team perspective
        team_games.append({
            'team': game['home_team'],
            'opponent': game['away_team'],
            'points_for': game['home_score'],
            'points_against': game['away_score'],
            'is_home': 1,
            'won': 1 if game['home_score'] > game['away_score'] else 0,
            'date': game['date']
        })
        
        # Away team perspective  
        team_games.append({
            'team': game['away_team'],
            'opponent': game['home_team'],
            'points_for': game['away_score'],
            'points_against': game['home_score'],
            'is_home': 0,
            'won': 1 if game['away_score'] > game['home_score'] else 0,
            'date': game['date']
        })
    
    team_df = pd.DataFrame(team_games)
    team_df['date'] = pd.to_datetime(team_df['date'])
    team_df = team_df.sort_values(['team', 'date'])
    
    team_df['avg_points_for'] = team_df.groupby('team')['points_for'].rolling(5, min_periods=1).mean().shift(1).values
    team_df['avg_points_against'] = team_df.groupby('team')['points_against'].rolling(5, min_periods=1).mean().shift(1).values
    
    # Fill NaN values for first games
    team_df['avg_points_for'] = team_df['avg_points_for'].fillna(team_df['points_for'])
    team_df['avg_points_against'] = team_df['avg_points_against'].fillna(team_df['points_against'])
    
    return team_df

def create_features(team_df):
    features = []
    targets = []
    
    for _, game in team_df.iterrows():
        if pd.notna(game['avg_points_for']) and pd.notna(game['avg_points_against']):
            feature_vector = [
                game['avg_points_for'],
                game['avg_points_against'], 
                game['is_home']
            ]
            features.append(feature_vector)
            targets.append(game['won'])
    
    return np.array(features), np.array(targets)

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Test the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.3f}")
    
    return model

def predict_game(model, team1_stats, team2_stats, team1_home=True):
    # Team 1 features (home advantage if applicable)
    team1_features = [team1_stats['avg_points_for'], team1_stats['avg_points_against'], 1 if team1_home else 0]
    
    # Team 2 features  
    team2_features = [team2_stats['avg_points_for'], team2_stats['avg_points_against'], 0 if team1_home else 1]
    
    # Get win probabilities
    team1_prob = model.predict_proba([team1_features])[0][1]
    team2_prob = model.predict_proba([team2_features])[0][1]
    
    return team1_prob, team2_prob

def main():
    print("Loading NBA game data...")
    team_df = load_and_prepare_data('nba_games.csv')
    
    print("Creating features...")
    X, y = create_features(team_df)
    
    print("Training model...")
    model = train_model(X, y)
    
    # Save the model
    joblib.dump(model, 'basketball_model.pkl')
    print("Model saved as 'basketball_model.pkl'")
    
    # Example prediction
    print("\nExample prediction:")
    team1_stats = {'avg_points_for': 110, 'avg_points_against': 105}
    team2_stats = {'avg_points_for': 108, 'avg_points_against': 107}
    
    prob1, prob2 = predict_game(model, team1_stats, team2_stats, team1_home=True)
    print(f"Team 1 (home) win probability: {prob1:.3f}")
    print(f"Team 2 (away) win probability: {prob2:.3f}")

if __name__ == "__main__":
    main()
