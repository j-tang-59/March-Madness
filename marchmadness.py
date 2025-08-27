import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import time
import random

datatable = pd.read_csv("data/games.csv")

rows = []

for index, row in datatable.iterrows():
    home_row = {
        "date": row["GAME_DATE_EST"],
        "season": row["SEASON"],
        "team": row["HOME_TEAM_ID"],
        "opp": row["VISITOR_TEAM_ID"],
        "ptdiff": row["PTS_home"] - row["PTS_away"],
        "is_home": 1
    }

    away_row = {
        "date": row["GAME_DATE_EST"],
        "season": row["SEASON"],
        "team": row["VISITOR_TEAM_ID"],
        "opp": row["HOME_TEAM_ID"],
        "ptdiff": row["PTS_away"] - row["PTS_home"],
        "is_home": 0
    }

    rows.append(home_row)
    rows.append(away_row)

teamDataFrame = pd.DataFrame(rows)

teamDataFrame = teamDataFrame.sort_values(["team", "date"])

print(teamDataFrame.head())

# Convert date strings to real datetimes
teamDataFrame["date"] = pd.to_datetime(teamDataFrame["date"])

# Sort properly by team, season, then date
teamDataFrame = teamDataFrame.sort_values(["team", "season", "date"]).reset_index(drop=True)

# Leakage-safe rolling: average point diff over the previous 10 games (min 5)
teamDataFrame["ptdiff_last10"] = (
    teamDataFrame
        .groupby(["team", "season"])["ptdiff"]
        .apply(lambda s: s.shift(1).rolling(window=10, min_periods=5).mean())
        .reset_index(level=[0, 1], drop=True)
)

# Early-season rows won't have enough history; fill those with 0 for now
teamDataFrame["ptdiff_last10"] = teamDataFrame["ptdiff_last10"].fillna(0.0)

print(teamDataFrame.head(15))

print("Creating training pairs...")

# Create a lookup dictionary for team stats by date
team_stats = {}
for _, row in teamDataFrame.iterrows():
    key = (row["team"], row["date"])
    team_stats[key] = {
        "ptdiff_last10": row["ptdiff_last10"]
    }

# Create training data from original games
training_data = []
for _, game in datatable.iterrows():
    game_date = pd.to_datetime(game["GAME_DATE_EST"])
    home_team = game["HOME_TEAM_ID"]
    away_team = game["VISITOR_TEAM_ID"]
    
    # Get team stats for this game date
    home_key = (home_team, game_date)
    away_key = (away_team, game_date)
    
    if home_key in team_stats and away_key in team_stats:
        home_ptdiff = team_stats[home_key]["ptdiff_last10"]
        away_ptdiff = team_stats[away_key]["ptdiff_last10"]
        
        # Features: difference in rolling averages + home court advantage
        feature_diff = home_ptdiff - away_ptdiff
        home_court = 1  # Home team gets +1 advantage
        
        # Target: 1 if home team won, 0 if away team won
        home_won = 1 if game["PTS_home"] > game["PTS_away"] else 0
        
        training_data.append({
            "ptdiff_advantage": feature_diff,
            "home_court": home_court,
            "home_won": home_won,
            "season": game["SEASON"]
        })

training_df = pd.DataFrame(training_data)
print(f"Created {len(training_df)} training examples")

print("\n=== STARTING CONTINUOUS TRAINING ===")
print("The model will train continuously and save improvements...")
print("Press Ctrl+C to stop training\n")

# Features and target
X = training_df[["ptdiff_advantage", "home_court"]]
y = training_df["home_won"]

# Track best performance
best_accuracy = 0
best_model = None
best_params = None
iteration = 0

# Hyperparameter ranges to explore
lr_params = {
    'C': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    'solver': ['liblinear', 'lbfgs'],
    'max_iter': [1000, 2000, 5000]
}

rf_params = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [3, 5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

nn_params = {
    'hidden_layer_sizes': [(25,), (50,), (100,), (50, 25), (100, 50), (100, 50, 25)],
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'learning_rate_init': [0.001, 0.01, 0.1]
}

try:
    while True:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")
        
        # Random train/test split each iteration
        random_state = random.randint(1, 10000)
        test_size = random.uniform(0.15, 0.3)  # Vary test size
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Randomly choose model type and parameters
        model_choice = random.choice(['lr', 'rf', 'nn'])
        
        if model_choice == 'lr':
            params = {k: random.choice(v) for k, v in lr_params.items()}
            model = LogisticRegression(random_state=random_state, **params)
            model_name = "Logistic Regression"
            
        elif model_choice == 'rf':
            params = {k: random.choice(v) for k, v in rf_params.items()}
            model = RandomForestClassifier(random_state=random_state, **params)
            model_name = "Random Forest"
            
        else:  # nn
            params = {k: random.choice(v) for k, v in nn_params.items()}
            model = MLPClassifier(random_state=random_state, max_iter=2000, **params)
            model_name = "Neural Network"
        
        # Add feature scaling for neural networks
        if model_choice == 'nn':
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model.fit(X_train_scaled, y_train)
            test_pred = model.predict(X_test_scaled)
            test_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            test_pred = model.predict(X_test)
            test_prob = model.predict_proba(X_test)[:, 1]
        
        # Evaluate performance
        accuracy = accuracy_score(y_test, test_pred)
        logloss = log_loss(y_test, test_prob)
        
        print(f"Model: {model_name}")
        print(f"Parameters: {params}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Log Loss: {logloss:.4f}")
        
        # Save if this is the best model so far
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_params = params
            
            # Save the best model
            if model_choice == 'nn':
                joblib.dump({'model': model, 'scaler': scaler, 'type': 'nn'}, "best_model.pkl")
            else:
                joblib.dump({'model': model, 'scaler': None, 'type': model_choice}, "best_model.pkl")
            
            print(f"🎉 NEW BEST MODEL! Accuracy: {best_accuracy:.4f}")
            print(f"Saved as 'best_model.pkl'")
            
            # Log the improvement
            with open("training_log.txt", "a") as f:
                f.write(f"Iteration {iteration}: {model_name} - Accuracy: {accuracy:.4f}, Log Loss: {logloss:.4f}\n")
                f.write(f"Parameters: {params}\n\n")
        
        # Sleep briefly to prevent overwhelming the CPU
        time.sleep(0.1)
        
        # Print progress every 10 iterations
        if iteration % 10 == 0:
            print(f"\n📊 Progress Summary:")
            print(f"Iterations completed: {iteration}")
            print(f"Best accuracy so far: {best_accuracy:.4f}")
            print(f"Best model type: {type(best_model).__name__}")

except KeyboardInterrupt:
    print(f"\n\n🛑 Training stopped by user after {iteration} iterations")
    print(f"Best accuracy achieved: {best_accuracy:.4f}")
    print(f"Best model saved as 'best_model.pkl'")

def predict_game_winner(home_team_ptdiff, away_team_ptdiff, model_path="best_model.pkl"):
    """
    Predict the probability that the home team wins using the best trained model
    """
    model_data = joblib.load(model_path)
    model = model_data['model']
    scaler = model_data['scaler']
    
    feature_diff = home_team_ptdiff - away_team_ptdiff
    home_court = 1
    
    features = np.array([[feature_diff, home_court]])
    
    if scaler is not None:  # Neural network model
        features = scaler.transform(features)
    
    win_prob = model.predict_proba(features)[0][1]
    return win_prob

print("\nExample prediction with best model:")
if best_model is not None:
    home_prob = predict_game_winner(5.2, -2.1)
    print(f"Home team win probability: {home_prob:.3f}")
