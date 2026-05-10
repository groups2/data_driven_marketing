import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, mean_squared_error

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

    # ── Load ──────────────────────────────────────────────
    train_df = pd.read_pickle(config['TRAIN_PATH'])
    val_df   = pd.read_pickle(config['VAL_PATH'])
    int_df   = pd.read_pickle(config['INTERNAL_TEST_PATH'])
    test_df  = pd.read_pickle(config['TEST_PATH'])

    with open(config['FEATURES_PATH']) as f:
        f_dict = json.load(f)
    clf_feat, reg_feat = f_dict['clf_features'], f_dict['reg_features']

    full_df     = pd.concat([train_df, val_df, int_df]).reset_index(drop=True)
    y_full_buy  = (full_df['target_revenue'] > 0).astype(int)
    y_train_buy = (train_df['target_revenue'] > 0).astype(int)
    y_val_buy   = (val_df['target_revenue'] > 0).astype(int)

    neg = (y_full_buy == 0).sum()
    pos = (y_full_buy == 1).sum()
    spw = np.sqrt(neg / pos)

    print(f"Negative: {neg:,} | Positive: {pos:,} | SPW: {spw:.2f}x")
    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | "
          f"Internal test: {len(int_df):,} | Kaggle test: {len(test_df):,}")
    print(f"Clf features: {len(clf_feat)} | Reg features: {len(reg_feat)}")


    # ── Stage 1: Classification ───────────────────────────
    print(f"\n{'='*55}")
    print(f"STAGE 1 — CLASSIFICATION")
    print(f"{'='*55}")

    true_rate    = y_val_buy.mean()
    baseline_auc = true_rate
    print(f"Buyers: {y_val_buy.sum()} / {len(y_val_buy):,} | true rate: {true_rate:.6f}")
    print(f"Baseline PR-AUC: {baseline_auc:.6f}\n")

    p_sum    = np.zeros(len(test_df))
    auc_list = []

    for s in range(config['N_RUNS']):
        clf = XGBClassifier(
            max_depth=4, learning_rate=0.05, scale_pos_weight=spw,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric='aucpr', n_estimators=1000, early_stopping_rounds=30,
            random_state=s, n_jobs=-1
        )
        clf.fit(
            train_df[clf_feat], y_train_buy,
            eval_set=[(val_df[clf_feat], y_val_buy)],
            verbose=False
        )

        p_val_raw = clf.predict_proba(val_df[clf_feat])[:, 1]
        auc_pr    = average_precision_score(y_val_buy, p_val_raw)
        auc_list.append(auc_pr)

        iso = IsotonicRegression(out_of_bounds='clip').fit(p_val_raw, y_val_buy)

        clf_full = XGBClassifier(
            max_depth=4, learning_rate=0.05, scale_pos_weight=spw,
            subsample=0.8, colsample_bytree=0.8, eval_metric='aucpr',
            n_estimators=clf.best_iteration, random_state=s, n_jobs=-1
        )
        clf_full.fit(full_df[clf_feat], y_full_buy, verbose=False)
        p_sum += iso.predict(clf_full.predict_proba(test_df[clf_feat])[:, 1])

        print(f"  seed={s} | PR-AUC={auc_pr:.4f} | iter={clf.best_iteration}")

    p_avg            = p_sum / config['N_RUNS']
    test_df['p_buy'] = p_avg
    calib_ratio      = p_avg.mean() / true_rate

    print(f"\nEnsemble PR-AUC: {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}")
    print(f"Baseline PR-AUC: {baseline_auc:.6f}")
    print(f"Mean p_buy: {p_avg.mean():.6f} | Calibration: {calib_ratio:.2f}x "
          f"{'✅' if 0.5 <= calib_ratio <= 2.0 else '⚠️'}")


    # ── Stage 2: Regression ───────────────────────────────
    print(f"\n{'='*55}")
    print(f"STAGE 2 — REGRESSION")
    print(f"{'='*55}")

    tr_b  = train_df[train_df['target_revenue'] > 0]
    val_b = val_df[val_df['target_revenue'] > 0]
    f_buy = full_df[full_df['target_revenue'] > 0]
    p99   = f_buy['target_log_revenue'].quantile(0.99)

    baseline_reg = np.sqrt(mean_squared_error(
        val_b['target_log_revenue'], np.zeros(len(val_b))
    ))
    print(f"Train buyers: {len(tr_b):,} | Val buyers: {len(val_b):,} | "
          f"Full buyers: {len(f_buy):,}")
    print(f"p99_log: {p99:.4f} | Baseline RMSE: {baseline_reg:.4f}\n")

    e_sum     = np.zeros(len(test_df))
    rmse_list = []

    for s in range(config['N_RUNS']):
        reg_cal = XGBRegressor(
            n_estimators=2000, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0,
            eval_metric='rmse', early_stopping_rounds=50,
            random_state=s, n_jobs=-1
        )
        reg_cal.fit(
            tr_b[reg_feat], tr_b['target_log_revenue'],
            eval_set=[(val_b[reg_feat], val_b['target_log_revenue'])],
            verbose=False
        )

        p_val_b = reg_cal.predict(val_b[reg_feat]).clip(0, p99)
        rmse    = np.sqrt(mean_squared_error(val_b['target_log_revenue'], p_val_b))
        rmse_list.append(rmse)

        reg_full = XGBRegressor(
            n_estimators=reg_cal.best_iteration, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0,
            random_state=s, n_jobs=-1
        )
        reg_full.fit(f_buy[reg_feat], f_buy['target_log_revenue'], verbose=False)
        e_sum += reg_full.predict(test_df[reg_feat]).clip(0, p99)

        print(f"  seed={s} | RMSE={rmse:.4f} | iter={reg_cal.best_iteration}")

    test_df['e_revenue']   = e_sum / config['N_RUNS']
    test_df['final_score'] = test_df['p_buy'] * test_df['e_revenue']

    print(f"\nEnsemble RMSE: {np.mean(rmse_list):.4f} ± {np.std(rmse_list):.4f}")
    print(f"Baseline RMSE: {baseline_reg:.4f}")


    # ── Combine ───────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"COMBINE")
    print(f"{'='*55}")

    top_k      = int(len(test_df) * 0.10)
    top_idx    = test_df['final_score'].nlargest(top_k).index
    lift_proxy = test_df.loc[top_idx, 'p_buy'].mean() / test_df['p_buy'].mean()

    print(f"Users: {len(test_df):,} | Top 10%: {top_k:,}")
    print(f"Lift@10%: {lift_proxy:.2f}x")

    test_df.to_pickle(config['OUTPUT_PATH'])
    print(f"\nSaved → {config['OUTPUT_PATH']}")
    return test_df


if __name__ == "__main__":
    run_inference_pipeline()