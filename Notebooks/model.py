"""
model.py - Fully Synchronized Inference Pipeline
"""

import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.isotonic import IsotonicRegression

DEFAULT_CONFIG = {
    'TRAIN_PATH':         '../data/train_final.pkl',
    'VAL_PATH':           '../data/val_final.pkl',
    'INTERNAL_TEST_PATH': '../data/test_final.pkl',
    'TEST_PATH':          '../data/real_test_final.pkl',
    'FEATURES_PATH':      '../data/final_features.json',
    'OUTPUT_PATH':        '../data/test_predictions.pkl',
    'N_RUNS':             5
}

def run_inference_pipeline(config=DEFAULT_CONFIG):
    print("--- 1. Loading Data & Features ---")
    train_df = pd.read_pickle(config['TRAIN_PATH'])
    val_df   = pd.read_pickle(config['VAL_PATH'])
    int_df   = pd.read_pickle(config['INTERNAL_TEST_PATH'])
    test_df  = pd.read_pickle(config['TEST_PATH'])
    
    with open(config['FEATURES_PATH']) as f:
        f_dict = json.load(f)
    clf_feat, reg_feat = f_dict['clf_features'], f_dict['reg_features']

    # Gộp toàn bộ dữ liệu đã biết nhãn để retrain tầng cuối
    full_df = pd.concat([train_df, val_df, int_df]).reset_index(drop=True)
    y_full_buy = (full_df['target_revenue'] > 0).astype(int)
    
    # --- STAGE 1: CLASSIFICATION ---
    print(f"--- 2. Stage 1 ({config['N_RUNS']} classifiers) ---")
    spw = np.sqrt((y_full_buy == 0).sum() / (y_full_buy == 1).sum())
    p_sum = np.zeros(len(test_df))
    
    for s in range(config['N_RUNS']):
        clf = XGBClassifier(
            max_depth=4, learning_rate=0.05, scale_pos_weight=spw, 
            subsample=0.8, colsample_bytree=0.8,
            eval_metric='aucpr', n_estimators=1000, early_stopping_rounds=30, 
            random_state=s, n_jobs=-1
        )
        clf.fit(train_df[clf_feat], (train_df['target_revenue']>0).astype(int), 
                eval_set=[(val_df[clf_feat], (val_df['target_revenue']>0).astype(int))], verbose=False)
        
        # Calibration
        iso = IsotonicRegression(out_of_bounds='clip').fit(clf.predict_proba(val_df[clf_feat])[:, 1], (val_df['target_revenue']>0).astype(int))
        
        # Retrain full
        clf_full = XGBClassifier(
            max_depth=4, learning_rate=0.05, scale_pos_weight=spw, 
            subsample=0.8, colsample_bytree=0.8,
            n_estimators=clf.best_iteration, random_state=s, n_jobs=-1
        )
        clf_full.fit(full_df[clf_feat], y_full_buy, verbose=False)
        
        p_sum += iso.predict(clf_full.predict_proba(test_df[clf_feat])[:, 1])
        print(f"   > Seed {s} classifier done (iter={clf.best_iteration}).")

    test_df['p_buy'] = p_sum / config['N_RUNS']

    # --- STAGE 2: REGRESSION ---
    print(f"--- 3. Stage 2 ({config['N_RUNS']} regressors) ---")
    tr_b, val_b = train_df[train_df['target_revenue']>0], val_df[val_df['target_revenue']>0]
    f_buy = full_df[full_df['target_revenue'] > 0]
    p99 = f_buy['target_log_revenue'].quantile(0.99)
    e_sum = np.zeros(len(test_df))
    
    for s in range(config['N_RUNS']):
        # Tìm best_iteration cho Regressor
        reg_cal = XGBRegressor(
            n_estimators=2000, max_depth=3, learning_rate=0.05, 
            subsample=0.8, colsample_bytree=0.8, 
            min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0, # Đồng nhất tham số
            eval_metric='rmse', early_stopping_rounds=50, random_state=s, n_jobs=-1
        )
        reg_cal.fit(tr_b[reg_feat], tr_b['target_log_revenue'], eval_set=[(val_b[reg_feat], val_b['target_log_revenue'])], verbose=False)
        
        # Retrain full
        reg_full = XGBRegressor(
            n_estimators=reg_cal.best_iteration, max_depth=3, learning_rate=0.05, 
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0, # Đồng nhất tham số
            random_state=s, n_jobs=-1
        )
        reg_full.fit(f_buy[reg_feat], f_buy['target_log_revenue'], verbose=False)
        
        e_sum += reg_full.predict(test_df[reg_feat]).clip(0, p99)
        print(f"   > Seed {s} regressor done (iter={reg_cal.best_iteration}).")

    test_df['e_revenue'] = e_sum / config['N_RUNS']
    test_df['final_score'] = test_df['p_buy'] * test_df['e_revenue']
    
    test_df.to_pickle(config['OUTPUT_PATH'])
    print(f"\n✅ Pipeline complete! Result saved to: {config['OUTPUT_PATH']}")
    return test_df

if __name__ == "__main__":
    run_inference_pipeline()
