import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, roc_auc_score, classification_report, brier_score_loss
import os
import joblib

def load_and_preprocess_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, low_memory=False)
    
    # 1. Basic Cleaning
    print("Feature Engineering: Context & Situation...")
    # Fill specific NAs
    df['line'] = df['line'].fillna('Unknown')
    df['length'] = df['length'].fillna('Unknown')
    df['batruns'] = df['batruns'].fillna(0)
    df['bowlruns'] = df['bowlruns'].fillna(0)
    
    # New Context Features Handling
    if 'cur_bat_bf' in df.columns:
        df['cur_bat_bf'] = df['cur_bat_bf'].fillna(0)
    if 'inns_rr' in df.columns:
        df['inns_rr'] = df['inns_rr'].fillna(0)
    # Required Run Rate (Crucial for 2nd Inns Chasing Pressure)
    if 'inns_rrr' in df.columns:
        df['inns_rrr'] = df['inns_rrr'].fillna(-1.0) # -1.0 implies No Target (1st Inns)


    # Bowling runs conceded (includes extras)
    if 'bowl_runs_conceded' not in df.columns:
        # Standard: BatRuns + Wides + NoBalls
        df['bowl_runs_conceded'] = df['batruns'] + df['wide'] + df['noball']
    
    # 2. Phase Definition (Updated: PP 1-6, MO 7-12, Slog 13-16, Death 17-20)
    # df['over'] is 0-indexed (0-19)
    conditions = [
        (df['over'] < 6),                  # 1-6 (Indices 0-5)
        (df['over'] >= 6) & (df['over'] < 12), # 7-12 (Indices 6-11)
        (df['over'] >= 12) & (df['over'] < 16),# 13-16 (Indices 12-15)
        (df['over'] >= 16)                 # 17-20 (Indices 16-19)
    ]
    choices = ['PP', 'MO', 'Slog', 'Death']
    df['phase'] = np.select(conditions, choices, default='MO')
    
    # 3. Balls Remaining & Wickets Down
    df['inns_wkts'] = df['inns_wkts'].fillna(0)
    df['wickets_in_hand'] = 10 - df['inns_wkts']
    if 'inns_balls_rem' not in df.columns:
        df['inns_balls_rem'] = 120 - (df['over']*6 + df['ball'])
        
    # 4. Rolling Venue Average (Context for Ground Scoring)
    # Important for normalizing "High Scoring Ground" vs "Low Scoring Ground"
    print("Calculating Rolling Venue Features...")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values(['date', 'p_match'])
    
    match_ground_stats = df.groupby(['ground', 'p_match', 'date']).agg({
        'batruns': 'sum',
        'ballfaced': 'sum'
    }).reset_index()
    
    def calculate_rolling_stats(group):
        # Rolling window of last 10 matches at this venue
        roll_runs = group['batruns'].rolling(window=10, min_periods=1).sum()
        roll_balls = group['ballfaced'].rolling(window=10, min_periods=1).sum()
        # Avg scoring rate
        roll_avg = roll_runs / roll_balls
        # Shift 1 to use ONLY historical data
        return roll_avg.shift(1)

    match_ground_stats['rolling_venue_avg'] = match_ground_stats.groupby('ground').apply(calculate_rolling_stats, include_groups=False).reset_index(level=0, drop=True)
    
    # Map back
    p_match_to_avg = dict(zip(match_ground_stats['p_match'], match_ground_stats['rolling_venue_avg']))
    df['rolling_venue_avg'] = df['p_match'].map(p_match_to_avg)
    
    # Fallback to global mean
    global_avg = df['batruns'].sum() / df['ballfaced'].sum()
    df['rolling_venue_avg'] = df['rolling_venue_avg'].fillna(global_avg)
    
    # 5. Label Encoding (Line, Length, Ground, and New Features)
    print("Encoding Categorical Features...")
    le_line = LabelEncoder()
    df['line_id'] = le_line.fit_transform(df['line'].astype(str))
    
    le_length = LabelEncoder()
    df['length_id'] = le_length.fit_transform(df['length'].astype(str))
    
    le_phase = LabelEncoder()
    df['phase_id'] = le_phase.fit_transform(df['phase'].astype(str))

    # New Features Encoding
    df['bat_hand'] = df['bat_hand'].fillna('Right') # Default to Right
    le_hand = LabelEncoder()
    df['bat_hand_id'] = le_hand.fit_transform(df['bat_hand'].astype(str))

    df['bowl_style'] = df['bowl_style'].fillna('Unknown')
    le_style = LabelEncoder()
    df['bowl_style_id'] = le_style.fit_transform(df['bowl_style'].astype(str))

    df['bowl_kind'] = df['bowl_kind'].fillna('Unknown')
    le_kind = LabelEncoder()
    df['bowl_kind_id'] = le_kind.fit_transform(df['bowl_kind'].astype(str))

    # Interaction Feature: Matchups (Hand vs Style)
    # e.g. "Right_OB" vs "Left_SLA" captures specific matchup advantages
    df['matchup_str'] = df['bat_hand'].astype(str) + "_" + df['bowl_style'].astype(str)
    le_matchup = LabelEncoder()
    df['matchup_id'] = le_matchup.fit_transform(df['matchup_str'])
    
    # Feature Enhancement: Run Rate Gap
    # If chasing (inns=2), rr_gap = Required RR - Current RR
    # If first innings (inns=1), we set inns_rrr = -1, so rr_gap is not directly comparable.
    # Logic: inns_rrr is filled with -1 for 1st innings.
    # For 1st innings, gap = 0 (neutral) or Current RR - Target (but no target).
    # Let's set dist_to_par? No, simpler:
    df['rr_gap'] = np.where(df['inns'] == 2, df['inns_rrr'] - df['inns_rr'], 0.0)
    
    return df

def apply_bayesian_shrinkage(df, k=100):
    """
    Applies Bayesian Shrinkage to RAA metrics to reduce noise from small sample sizes.
    Formula: (Sum_RAA + Prior_Mean * K) / (Balls + K)
    
    Returns df with 'RAA_bat_adj_rate' and 'RAA_bowl_adj_rate'.
    """
    print(f"Applying Bayesian Shrinkage (K={k})...")
    
    # --- Batting ---
    if 'RAA_bat' in df.columns:
        # 1. Global Prior (Weighted Mean RAA/Ball)
        prior_bat = df['RAA_bat'].sum() / len(df)
        print(f"  > Batting Prior (RAA/ball): {prior_bat:.4f}")
        
        # 2. Player Aggregates
        bat_agg = df.groupby('bat').agg(
            sum_raa=('RAA_bat', 'sum'),
            balls=('ball', 'count')
        )
        
        # 3. Calculate Adjusted Rating (Per Ball)
        bat_agg['adj_rate'] = (bat_agg['sum_raa'] + (prior_bat * k)) / (bat_agg['balls'] + k)
        
        # 4. Map back to DF
        df['RAA_bat_adj_rate'] = df['bat'].map(bat_agg['adj_rate'])
        
    # --- Bowling ---
    if 'RAA_bowl' in df.columns:
        # 1. Global Prior (Weighted Mean RAA/Ball)
        prior_bowl = df['RAA_bowl'].sum() / len(df)
        print(f"  > Bowling Prior (RAA/ball): {prior_bowl:.4f}")
        
        # 2. Player Aggregates
        bowl_agg = df.groupby('bowl').agg(
            sum_raa=('RAA_bowl', 'sum'),
            balls=('ball', 'count')
        )
        
        # 3. Calculate Adjusted Rating (Per Ball)
        bowl_agg['adj_rate'] = (bowl_agg['sum_raa'] + (prior_bowl * k)) / (bowl_agg['balls'] + k)
        
        # 4. Map back to DF
        df['RAA_bowl_adj_rate'] = df['bowl'].map(bowl_agg['adj_rate'])

    return df
# We store trained models here to expose prediction API
TRAINED_MODELS = {}

def train_and_save_pipeline(df):
    """
    Trains models on the FULL dataset provided and saves them for API usage.
    """
    print("Training Global Models (For Prediction API)...")
    
    features = [
        'wickets_in_hand',
        'inns_balls_rem',
        'cur_bat_bf',
        'inns_rr',
        'inns_rrr',
        'rr_gap',         # [NEW]
        'rolling_venue_avg',
        'phase_id',
        'line_id',
        'length_id',
        'over',
        'bat_hand_id',
        'bowl_style_id',
        'bowl_kind_id',
        'matchup_id'
    ]
    
    # Store features for consistency
    TRAINED_MODELS['features'] = features
    
    X = df[features]
    
    # Monotone Constraints: 
    # wickets_in_hand (+1), inns_balls_rem (+1)
    # Features list order matches constraints list. 
    # We need to map constraints to feature names or use column indices if passing numpy array.
    # LGBM sklearn API accepts monotone_constraints="name:value,name:value" string or list.
    
    monotone_constraints = [
        1 if f in ['wickets_in_hand', 'inns_balls_rem'] else 0 
        for f in features
    ]
    
    # xR Batting
    y_bat = df['batruns']
    model_bat = lgb.LGBMRegressor(
        objective='regression',
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        monotone_constraints=monotone_constraints,
        n_jobs=-1,
        random_state=42,
        verbose=-1
    )
    model_bat.fit(X, y_bat)
    TRAINED_MODELS['xr_bat'] = model_bat
    
    # xW Wicket
    y_wicket = df['out'].astype(int)
    model_xw = lgb.LGBMRegressor(
        objective='binary',
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        n_jobs=-1,
        random_state=42,
        verbose=-1
    )
    model_xw.fit(X, y_wicket)
    TRAINED_MODELS['xw'] = model_xw
    
    print("Models trained and registered.")

def predict_expected_runs(X: pd.DataFrame) -> np.ndarray:
    """Predicted Expected Runs (xR) for given features."""
    if 'xr_bat' not in TRAINED_MODELS:
        raise ValueError("Model not trained. Run train_and_save_pipeline or load_model first.")
    
    # Ensure feature order if possible, or assume caller provides correct X
    # In production, we'd realign X[TRAINED_MODELS['features']]
    return TRAINED_MODELS['xr_bat'].predict(X)

def predict_wicket_prob(X: pd.DataFrame) -> np.ndarray:
    """Predict Wicket Probability (xW) for given features."""
    if 'xw' not in TRAINED_MODELS:
        raise ValueError("Model not trained.")
    return TRAINED_MODELS['xw'].predict(X)

def train_model(df):
    """
    Trains models on the FULL dataset provided and saves them for API usage.
    """
    print("Training Global Models (For Prediction API)...")
    
    features = [
        'wickets_in_hand',
        'inns_balls_rem',
        'cur_bat_bf',
        'inns_rr',
        'inns_rrr',
        'rr_gap',         # [NEW]
        'rolling_venue_avg',
        'phase_id',
        'line_id',
        'length_id',
        'over',
        'bat_hand_id',
        'bowl_style_id',
        'bowl_kind_id',
        'matchup_id'
    ]
    
    X = df[features]
    
    # Monotone Constraints: 
    # wickets_in_hand (+1), inns_balls_rem (+1)
    # Features list order matches constraints list. 
    # We need to map constraints to feature names or use column indices if passing numpy array.
    # LGBM sklearn API accepts monotone_constraints="name:value,name:value" string or list.
    
    monotone_constraints = [
        1 if f in ['wickets_in_hand', 'inns_balls_rem'] else 0 
        for f in features
    ]
    
    # xR Batting
    y_bat = df['batruns']
    model_bat = lgb.LGBMRegressor(
        objective='regression',
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        monotone_constraints=monotone_constraints,
        n_jobs=-1,
        random_state=42,
        verbose=-1
    )
    model_bat.fit(X, y_bat)
    TRAINED_MODELS['xr_bat'] = model_bat
    
    # xW Wicket
    y_wicket = df['out'].astype(int)
    model_xw = lgb.LGBMRegressor(
        objective='binary',
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        n_jobs=-1,
        random_state=42,
        verbose=-1
    )
    model_xw.fit(X, y_wicket)
    TRAINED_MODELS['xw'] = model_xw
    
    print("Models trained and registered.")

def predict_expected_runs(X: pd.DataFrame) -> np.ndarray:
    """Predicted Expected Runs (xR) for given features."""
    if 'xr_bat' not in TRAINED_MODELS:
        raise ValueError("Model not trained. Run train_and_save_pipeline or load_model first.")
    
    # Ensure feature order matches
    model = TRAINED_MODELS['xr_bat']
    # Check if we need to align columns? XGBoost handles by name in recent versions/sklearn wrapper? 
    # Safest to enforce list
    return model.predict(X)

def predict_wicket_prob(X: pd.DataFrame) -> np.ndarray:
    """Predict Wicket Probability (xW) for given features."""
    if 'xw' not in TRAINED_MODELS:
        raise ValueError("Model not trained.")
    return TRAINED_MODELS['xw'].predict(X)

def train_model(df):
    """
    Trains XGBoost using contextual features.
    """
    print("Training XGBoost Model (Tuned)...")
    
    features = [
        'wickets_in_hand',
        'inns_balls_rem',
        # Player State Features (Critical for Context)
        'cur_bat_bf',     # How "Set" is the batter? (0 balls vs 30 balls)
        'inns_rr',        # Scoreboard Pressure (Momentum)
        'inns_rrr',       # Chasing Pressure (Required Run Rate)
        'rr_gap',         # [NEW] Comparison of RRR vs Current RR
        'rolling_venue_avg',
        'phase_id',
        'line_id',
        'length_id',
        # New Features
        'over',           # Captures non-linear trends (e.g. 7th over dip)
        'bat_hand_id',    # Matchups (RHB vs LHB)
        'bowl_style_id',  # Granular style (OB vs SLA)
        'bowl_kind_id',    # Pace vs Spin distinction
        'matchup_id'      # Explicit Interaction (Hand x Style)
    ]
    
    X = df[features]
    
    # Common Hyperparams for Tuning (Prevent Overfitting, Learn Slower)
    # Monotone constraints
    monotone_constraints = [
        1 if f in ['wickets_in_hand', 'inns_balls_rem'] else 0 
        for f in features
    ]
    
    model_params = {
        'objective': 'regression',
        'n_estimators': 500,   # Increased for LGBM
        'learning_rate': 0.05,
        'max_depth': 5,
        'colsample_bytree': 0.8,
        'monotone_constraints': monotone_constraints,
        'n_jobs': -1,
        'random_state': 42,
        'verbose': -1
    }

    # --- Batting xR ---
    print("  > Batting xR (Target: Bat Runs)...")
    y_bat = df['batruns']
    model_bat = lgb.LGBMRegressor(**model_params)
    # Use GroupKFold for CV to prevent leakage if match_id is available
    if 'match_id' in df.columns:
        gkf = GroupKFold(n_splits=5)
        # cross_val_predict doesn't support groups directly in this signature easily in all versions without iterator
        # We will use simple CV for now or manual if strict.
        # Given "use match-level CV", let's try to pass groups via cv arg if iterator.
        # Standard cross_val_predict(estimator, X, y, cv=gkf, groups=groups) works in newer sklearn.
        df['xR_bat'] = cross_val_predict(model_bat, X, y_bat, cv=gkf, groups=df['match_id'], n_jobs=-1)
    else:
        df['xR_bat'] = cross_val_predict(model_bat, X, y_bat, cv=3, n_jobs=-1)
    
    # Eval Batting
    mse_bat = mean_squared_error(y_bat, df['xR_bat'])
    rmse_bat = np.sqrt(mse_bat)
    print("\n  [METRICS] Batting xR Model:")
    print(f"    RMSE: {rmse_bat:.4f} (Target: < 1.2)")
    print(f"    R2 Score: {r2_score(y_bat, df['xR_bat']):.4f}")
    
    # --- Bowling xR ---
    print("  > Bowling xR (Target: Runs Conceded)...")
    y_bowl = df['bowl_runs_conceded']
    model_bowl = lgb.LGBMRegressor(**model_params)
    
    if 'match_id' in df.columns:
        gkf = GroupKFold(n_splits=5)
        df['xR_bowl'] = cross_val_predict(model_bowl, X, y_bowl, cv=gkf, groups=df['match_id'], n_jobs=-1)
    else:
        df['xR_bowl'] = cross_val_predict(model_bowl, X, y_bowl, cv=3, n_jobs=-1)

    # Eval Bowling
    mse_bowl = mean_squared_error(y_bowl, df['xR_bowl'])
    rmse_bowl = np.sqrt(mse_bowl)
    print("\n  [METRICS] Bowling xR Model:")
    print(f"    RMSE: {rmse_bowl:.4f} (Target: < 1.2)")
    print(f"    R2 Score: {r2_score(y_bowl, df['xR_bowl']):.4f}")
    
    # --- Wickets (xW) ---
    print("  > Wicket Probability (xW)...")
    df['is_wicket'] = df['out'].astype(int)
    y_wicket = df['is_wicket']
    
    # Remove reg constraints? No, wickets_in_hand -> more wickets? Maybe less careful?
    # Actually, more wickets in hand -> play safer? No, play riskier -> more wickets.
    # So wickets_in_hand constraint for xW should be Positive?
    # Balls remaining: More balls -> less pressure -> fewer wickets? Or more time to build?
    # Let's keep constraints off for xW for now, or careful.
    # User only mentioned expected runs model for constraints context usually.
    # But generally: Wickets In Hand -> Higher Score (xR).
    # Wickets In Hand -> Higher Wicket Prob? (Risk taking).
    # Let's stick to xR constraints for now.
    
    model_xw = lgb.LGBMRegressor( 
        objective='binary',
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        n_jobs=-1,
        random_state=42,
        verbose=-1
    )
    
    if 'match_id' in df.columns:
        gkf = GroupKFold(n_splits=5)
        df['xW'] = cross_val_predict(model_xw, X, y_wicket, cv=gkf, groups=df['match_id'], n_jobs=-1)
    else:
        df['xW'] = cross_val_predict(model_xw, X, y_wicket, cv=3, n_jobs=-1)

    # Eval Wickets
    print("\n  [METRICS] xW (Wicket) Model:")
    try:
        roc = roc_auc_score(y_wicket, df['xW'])
        brier = brier_score_loss(y_wicket, df['xW'])
        print(f"    ROC-AUC: {roc:.4f} (Target: > 0.8)")
        print(f"    Brier Score: {brier:.4f} (Closer to 0 is better)")
        
        # Phase-wise Diagnostics
        print("\n  [DIAGNOSTICS] Phase-wise Performance:")
        for ph in sorted(df['phase'].unique()):
            mask = df['phase'] == ph
            if mask.sum() > 0:
                ph_rmse = np.sqrt(mean_squared_error(y_bat[mask], df.loc[mask, 'xR_bat']))
                ph_roc = roc_auc_score(y_wicket[mask], df.loc[mask, 'xW'])
                print(f"    Phase {ph}: xR RMSE = {ph_rmse:.3f}, xW AUC = {ph_roc:.3f}")

    except Exception as e:
        print(f"    Metric Error: {e}")
        
    # Feature Importance (Fit single model to get importances)
    print("\n  [DIAGNOSTICS] Feature Importance (Batting xR):")
    model_bat.fit(X, y_bat)
    imps = pd.DataFrame({'Feature': features, 'Importance': model_bat.feature_importances_})
    print(imps.sort_values('Importance', ascending=False).head(5).to_string(index=False))
    
    return df

def calculate_advanced_metrics(df):
    """
    Calculates True Strike Rate (True SR) and Wicket Advancement Average (WAA).
    """
    print("Calculating True SR and WAA...")

    # Calculate True SR (Difference Method: Actual - Expected)
    # Per ball contribution to True SR = (Runs - xR) * 100
    # Mean of this column = Actual SR - Expected SR
    df['True_SR'] = (df['batruns'] - df['xR_bat']) * 100

    # WAA (Wicket Advancement Average)
    df['WAA_bat'] = df['xW'] - df['is_wicket']
    
    # For bowlers, Positive WAA means Taking MORE wickets than expected.
    df['WAA_bowl'] = df['is_wicket'] - df['xW']

    # --- IMPACT METRIC (RAA + Wicket Value) ---
    # Formula: Impact = RunsSaved + (WicketsTakenAboveExp * WicketValue)
    WICKET_VALUE = 22.0 
    
    # Bowling Impact: RAA (Economy) + WAA (Strike Power)
    df['Impact_bowl'] = df['RAA_bowl'] + (df['WAA_bowl'] * WICKET_VALUE)
    
    # Batting Impact: RAA (Runs) + WAA (Survival) 
    df['Impact_bat'] = df['RAA_bat'] + (df['WAA_bat'] * WICKET_VALUE)

    return df

def main():
    import os
    filepath = os.environ.get('T20_BBB', 't20_bbb.csv')
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    # 1. Pipeline
    print("--- Starting RAAR pipeline ---")
    df = load_and_preprocess_data(filepath)
    df = train_model(df)
    
    # 2. RAA Calculation (Standard)
    # RAA = Actual - Expected (or vice versa depending on perspective)
    # Batting RAA: Actual Runs - Expected Runs
    df['RAA_bat'] = df['batruns'] - df['xR_bat']
    
    # Bowling RAA: Expected Runs - Actual Runs Conceded
    # (Positive means they conceded LESS than expected, i.e., Good)
    df['RAA_bowl'] = df['xR_bowl'] - df['bowl_runs_conceded']
    
    # 3. Advanced Metrics (True SR, WAA, Impact)
    df = calculate_advanced_metrics(df)
    
    # 3.5. Bayesian Shrinkage (Reduce Noise)
    df = apply_bayesian_shrinkage(df, k=100)
    
    # Debug: Show Shrinkage Impact
    print("\n[Shrinkage Check] Top Difference in RAA Rate (Raw vs Adj):")
    # Quick groupby to check rates
    if 'RAA_bat_adj_rate' in df.columns:
        chk = df.groupby('bat').agg(
            balls=('ball', 'count'),
            raw_rate=('RAA_bat', 'mean'), # RAA_bat is total? No df['RAA_bat'] is per ball deviation?
            # Wait, step 2 definition: df['RAA_bat'] = df['batruns'] - df['xR_bat']. Yes, per ball.
            adj_rate=('RAA_bat_adj_rate', 'mean') # Constant per player
        ).reset_index()
        chk['diff'] = abs(chk['raw_rate'] - chk['adj_rate'])
        print(chk.sort_values('diff', ascending=False).head(5).to_string(index=False))

    # 4. Save
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "processed_data.parquet")
    print(f"Saving results to {output_path}...")
    df.to_parquet(output_path, index=False)
    
    # 5. Validation Check (IPL 2025 & 2023 for Benchmarking)
    print("Years in data:", df['year'].unique())
    
    # IPL 2023 Validation (Liam Livingstone Benchmark)
    if 2023 in df['year'].unique():
        print("\n--- Validation: IPL 2023 (Liam Livingstone Check) ---")
        df_23 = df[(df['year'] == 2023) & (df['competition'] == 'IPL')]
        
        # Group by Batter
        qv_23 = df_23.groupby('bat').agg({
            'batruns': 'sum',
            'xR_bat': 'sum',
            'ball': 'count',
            'is_wicket': 'sum',
            'True_SR': 'mean' # Mean of (Runs - xR)*100 = Actual SR - Exp SR
        }).reset_index()
        
        # Calculate True Avg
        # True Avg = Actual Avg - Exp Avg
        qv_23['Avg'] = qv_23['batruns'] / qv_23['is_wicket']
        qv_23['xAvg'] = qv_23['xR_bat'] / qv_23['is_wicket']
        qv_23['True_Avg'] = qv_23['Avg'] - qv_23['xAvg']
        
        target_player = 'Liam Livingstone'
        if target_player in qv_23['bat'].values:
            p_data = qv_23[qv_23['bat'] == target_player]
            print(p_data[['bat', 'batruns', 'Avg', 'True_SR', 'True_Avg']].to_string(index=False))
        else:
            print(f"{target_player} not found in 2023 data.")

    # IPL 2021 Validation (Harshal Patel Benchmark)
    if 2021 in df['year'].unique():
        print("\n--- Validation: IPL 2021 (Harshal Patel Check) ---")
        df_21 = df[(df['year'] == 2021) & (df['competition'] == 'IPL')]
        
        # Group by Bowler
        qv_21 = df_21.groupby('bowl').agg({
            'bowl_runs_conceded': 'sum',
            'xR_bowl': 'sum',
            'ball': 'count',
            'is_wicket': 'sum',
            'xW': 'sum'
        }).reset_index()
        
        # Calculate Metrics
        # True Wickets = Actual - Expected
        qv_21['True_Wkts'] = qv_21['is_wicket'] - qv_21['xW']
        
        # True Economy = Actual Econ - Expected Econ
        # Econ = (Runs / Balls) * 6
        qv_21['Actual_Econ'] = (qv_21['bowl_runs_conceded'] / qv_21['ball']) * 6
        qv_21['Exp_Econ'] = (qv_21['xR_bowl'] / qv_21['ball']) * 6
        qv_21['True_Econ'] = qv_21['Actual_Econ'] - qv_21['Exp_Econ']
        
        target_bowler = 'Harshal Patel'
        if target_bowler in qv_21['bowl'].values:
            p_data = qv_21[qv_21['bowl'] == target_bowler]
            print(p_data[['bowl', 'is_wicket', 'True_Wkts', 'Actual_Econ', 'True_Econ']].to_string(index=False))
        else:
             print(f"{target_bowler} not found in 2021 data.")

    if 2025 in df['year'].unique():
        df_ipl_25 = df[(df['year'] == 2025) & (df['competition'] == 'IPL')]
        
        print("\n--- Validation: Top 10 Batters (2025 IPL) by Impact ---")
        if not df_ipl_25.empty:
            leaderboard_25 = df_ipl_25.groupby('bat')[['RAA_bat', 'Impact_bat']].sum().sort_values('Impact_bat', ascending=False).head(10)
            print(leaderboard_25)

        print("\n--- Validation: Top 15 Bowlers (2025 IPL, Min 100 Balls) by IMPACT ---")
        if not df_ipl_25.empty:
            # Impact_bowl: Positive is good
            bowl_stats = df_ipl_25.groupby('bowl').agg({
                'RAA_bowl': 'sum',
                'Impact_bowl': 'sum',
                'ball': 'count'
            }).reset_index()
            
            # Filter Min 100 Balls
            bowl_stats = bowl_stats[bowl_stats['ball'] >= 100].sort_values('Impact_bowl', ascending=False)
            
            # Format and Print
            bowl_stats = bowl_stats.round(2)
            print(bowl_stats.head(15).to_string(index=False))
        else:
            print("(No 2025 IPL data found for bowlers)")

if __name__ == "__main__":
    main()
