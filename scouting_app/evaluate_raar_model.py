import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb # Keep for backward compat if needed, or remove
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
import raar_model
import os

# --- Configuration ---
DATA_PATH = os.environ.get("T20_BBB", "t20_bbb.csv")
OUTPUT_DIR = "evaluation_results"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def rolling_origin_evaluation(df):
    """
    Performs Time-Series Split (Rolling Origin) Evaluation.
    Train on [Start...Year N], Test on [Year N+1].
    """
    print("\n--- Starting Rolling-Origin Evaluation (Time-Series Split) ---")
    
    years = sorted(df['year'].unique())
    # Start checking from 2018 to ensure enough training data (2015-2017)
    start_eval_year = 2018 
    
    results = []
    
    for split_year in range(start_eval_year, max(years) + 1):
        if split_year not in years: continue
        
        print(f"  > Eval Window: Test Year {split_year} (Train: {min(years)}-{split_year-1})")
        
        # Split Data
        train_mask = df['year'] < split_year
        test_mask = df['year'] == split_year
        
        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()
        
        if len(train_df) < 1000 or len(test_df) < 1000:
            print("    Skipping: Insufficient data.")
            continue
            
        # Train Models using raar_model's logic (but locally to avoid overwriting global)
        # We reuse the logic but need to fit fresh models
        features = [
            'wickets_in_hand', 'inns_balls_rem', 'cur_bat_bf', 'inns_rr', 'inns_rrr', 'rr_gap',
            'rolling_venue_avg', 'phase_id', 'line_id', 'length_id', 'over', 
            'bat_hand_id', 'bowl_style_id', 'bowl_kind_id', 'matchup_id'
        ]
        
        X_train = train_df[features]
        y_bat_train = train_df['batruns']
        y_wkt_train = train_df['out'].astype(int)
        
        X_test = test_df[features]
        y_bat_test = test_df['batruns']
        y_wkt_test = test_df['out'].astype(int)
        
        # Monotone Constraints for Batting
        monotone_constraints = [
            1 if f in ['wickets_in_hand', 'inns_balls_rem'] else 0 
            for f in features
        ]
        
        # Train Batting (xR) with LightGBM
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
        model_bat.fit(X_train, y_bat_train)
        preds_bat = model_bat.predict(X_test)
        
        # Train Bowling (xW) with LightGBM
        model_xw = lgb.LGBMRegressor(
            objective='binary', 
            n_estimators=200, 
            learning_rate=0.05, 
            max_depth=5, 
            n_jobs=-1, 
            random_state=42,
            verbose=-1
        )
        model_xw.fit(X_train, y_wkt_train)
        preds_xw = model_xw.predict(X_test)
        
        # Calculate Metrics (Global)
        rmse = np.sqrt(mean_squared_error(y_bat_test, preds_bat))
        auc = roc_auc_score(y_wkt_test, preds_xw)
        brier = brier_score_loss(y_wkt_test, preds_xw)
        
        print(f"    Result: RMSE={rmse:.4f}, AUC={auc:.4f}, Brier={brier:.4f}")
        
        # --- Phase-wise Diagnostics ---
        # Add phase info to test df for grouping
        test_df['xR_bat_pred'] = preds_bat
        test_df['xW_pred'] = preds_xw
        
        print("    [Phase Breakdown]")
        phases = {1: 'Powerplay', 2: 'Middle', 3: 'Death'} # Assuming 1,2,3 logic. 
        # Actually raar_model uses: 1 (PP), 2 (MO), 3 (Death) usually? 
        # Let's verify phase_id values or assume standard. 
        # If unknown, just iterate unique.
        
        for ph in sorted(test_df['phase_id'].unique()):
            mask = test_df['phase_id'] == ph
            if mask.sum() > 50:
                y_true_ph = y_bat_test[mask]
                y_pred_ph = preds_bat[mask]
                y_wkt_true_ph = y_wkt_test[mask]
                y_wkt_pred_ph = preds_xw[mask]
                
                ph_rmse = np.sqrt(mean_squared_error(y_true_ph, y_pred_ph))
                try:
                    ph_auc = roc_auc_score(y_wkt_true_ph, y_wkt_pred_ph)
                except ValueError:
                    ph_auc = 0.5 # Single class present
                
                ph_name = phases.get(ph, f"Phase {ph}")
                print(f"      {ph_name:<10}: RMSE={ph_rmse:.3f}, AUC={ph_auc:.3f}")
        
        results.append({
            'Test_Year': split_year,
            'Train_Size': len(train_df),
            'Test_Size': len(test_df),
            'RMSE_xR': rmse,
            'AUC_xW': auc,
            'Brier_xW': brier
        })
        
        # Save Calibration Data for the FINAL year (most recent check)
        if split_year == max(years):
            plot_calibration_curve(y_wkt_test, preds_xw, split_year)

    # Save Aggregate Results
    res_df = pd.DataFrame(results)
    print("\n--- Rolling Origin Performance Summary ---")
    print(res_df.to_string(index=False))
    res_df.to_csv(f"{OUTPUT_DIR}/rolling_origin_metrics.csv", index=False)
    
    return res_df

def plot_calibration_curve(y_true, y_prob, year):
    """
    Plots calibration curve for xW model.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
    
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='xW Model', color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated', color='gray')
    plt.xlabel('Predicted Probability (xW)')
    plt.ylabel('Actual Wicket Frequency')
    plt.title(f'Calibration Plot (Test Year {year})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/calibration_plot_{year}.png")
    print(f"Saved calibration plot to {OUTPUT_DIR}/calibration_plot_{year}.png")

def bayesian_shrinkage_demo(df):
    """
    Demonstrates shrinkage impact on top/bottom players.
    Shrunken Estimate = (Sum + Prior*K) / (N + K)
    """
    print("\n--- Bayesian Shrinkage Demo ---")
    # Using simple RAA
    
    # 1. Calculate Raw Stats
    agg = df.groupby('bat').agg(
        runs=('batruns', 'sum'),
        balls=('ball', 'count'),
        raa=('RAA_bat', 'sum')
    ).reset_index()
    
    agg['raa_per_ball_raw'] = agg['raa'] / agg['balls']
    
    # 2. Calculate Prior (Global Mean RAA/Ball)
    # Theoretically 0, but let's check
    prior_mean = agg['raa'].sum() / agg['balls'].sum() # Weighted mean
    print(f"Global Prior Mean (RAA/ball): {prior_mean:.4f}")
    
    # K factor (Strength of prior). User suggested CV, we use heuristic K=100 balls
    K = 100 
    
    # 3. Apply Shrinkage
    # Formula: (Sum_RAA + Prior * K) / (Balls + K)
    agg['raa_per_ball_shrunk'] = (agg['raa'] + (prior_mean * K)) / (agg['balls'] + K)
    
    # Show Top movers (High Variance -> Shrunk)
    agg['diff'] = abs(agg['raa_per_ball_raw'] - agg['raa_per_ball_shrunk'])
    
    print(f"\nTop 5 Players Most Affected by Shrinkage (K={K}):")
    print(agg.sort_values('diff', ascending=False).head(5)[['bat', 'balls', 'raa_per_ball_raw', 'raa_per_ball_shrunk']].to_string(index=False))
    
    # Save Full
    agg.to_csv(f"{OUTPUT_DIR}/shrinkage_impact.csv", index=False)

def main():
    # 1. Load Data
    # Use global function from raar
    df = raar_model.load_and_preprocess_data(DATA_PATH)
    
    # Ensure features created (preprocess might need train_model to finish? No, separate steps)
    # Actually load_and_preprocess misses some feature steps inside train_model like encoding 
    # WE MUST RUN raar_model.train_model(df) to populate all columns? 
    # Wait, train_model modifies df in place and returns it. We can use that.
    # But train_model does the CV predict. We just want feature engineering.
    # Let's verify if raar_model does feat engineering inside train_model. 
    # Yes: encoding labels.
    # So we should call a prep function.
    # Ideally raar_model should expose a 'prepare_features(df)' function.
    # For now, I will manually replicate the LabelEncoding or call the function and ignore the CV.
    
    print("Preparing features...")
    # Hack: Run train_model but ignore output just to get encoded columns in DF
    # Actually, train_model inside raar_model does: le.fit_transform.
    # If we call it, it adds cols.
    df = raar_model.train_model(df) 
    
    # 2. Run Rolling Eval
    rolling_origin_evaluation(df)
    
    # 3. Run Shrinkage Demo
    # Need to calculate RAA globally first (using models trained on full data or the last split?
    # Shrinkage is typically applied to the 'current' season or full dataset stats.
    # Let's use the full DF with predictions from the *Full Global Model* (which we trained in raar_model?)
    # Wait, raar_model.train_model(df) returns df with xR columns populated by Cross-Validation?
    # No, raar_model.train_model(df) does 3-fold CV and adds 'xR_bat'.
    # So 'RAA_bat' logic needs to be run.
    
    # Calculate RAA columns if missing
    if 'RAA_bat' not in df.columns:
        # We need predictions first.
        # raar_model.train_model(df) was called in main, so df SHOULD have xR_bat if train_model does it.
        # Let's heck raar_model values. 
        # Actually my refactored train_model might NOT be returning the DF with cols if I didn't assign it properly?
        # Check raar_model.train_model logic... it does `df['xR_bat'] = ... return df`.
        # So df has xR_bat.
        df['RAA_bat'] = df['batruns'] - df['xR_bat']
    
    bayesian_shrinkage_demo(df)

if __name__ == "__main__":
    main()
