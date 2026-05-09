import os
import pickle
import argparse
import numpy as np
import pandas as pd
from datetime import timedelta
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data


# ════════════════════════════════════════════════════════════
# 0. CONFIG
# ════════════════════════════════════════════════════════════
# Đường dẫn input
TRAIN_INPUT_PATH       = '../data/full_df_preprocessed.pkl'     # Kaggle train đã preprocess
KAGGLE_TEST_INPUT_PATH = '../data/kaggle_test_preprocessed.pkl' # Kaggle test đã preprocess

# Đường dẫn output — train mode (gộp toàn bộ)
TRAIN_OUTPUT_PATH = '../data/train_final.pkl'  # Toàn bộ train+val+test đã FE

# Đường dẫn output — kaggle_test mode
KAGGLE_TEST_OUTPUT_PATH = '../data/kaggle_test_final.pkl'

# Artifacts fit từ train → dùng lại cho kaggle test
ARTIFACTS_PATH = '../data/fe_artifacts.pkl'

# Fold config
FEATURE_DAYS = 168
TARGET_DAYS  = 92
SLIDE_DAYS   = 60

# Categorical columns cần top-N grouping
CAT_COLS = [
    'geoNetwork_country', 'geoNetwork_region', 'geoNetwork_city',
    'geoNetwork_metro', 'geoNetwork_subContinent',
    'trafficSource_source', 'trafficSource_campaign',
    'device_browser', 'device_operatingSystem',
]

# Categorical columns cần target encode
CAT_ENCODE_COLS = [
    'geoNetwork_city', 'geoNetwork_country', 'geoNetwork_region',
    'geoNetwork_metro', 'geoNetwork_subContinent',
    'trafficSource_source', 'trafficSource_campaign',
    'device_browser', 'device_operatingSystem',
    'channelGrouping', 'trafficSource_medium', 'device_deviceCategory',
]

DIVERSITY_COLS = [
    'geoNetwork_city', 'geoNetwork_country', 'trafficSource_source',
    'trafficSource_medium', 'channelGrouping',
    'device_browser', 'device_operatingSystem',
]


# ════════════════════════════════════════════════════════════
# 1. TIME FOLD GENERATOR
# ════════════════════════════════════════════════════════════
def generate_time_folds(df, date_col='date',
                        feature_days=FEATURE_DAYS,
                        target_days=TARGET_DAYS,
                        slide_days=SLIDE_DAYS):
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    print(f"Dữ liệu gốc: Từ {min_date.date()} đến {max_date.date()}\n")

    folds = []
    current_start = min_date
    fold_idx = 1

    while True:
        feature_end  = current_start + timedelta(days=feature_days)
        target_start = feature_end
        target_end   = target_start + timedelta(days=target_days)

        if target_end >= max_date:
            break

        folds.append({
            'Fold':         fold_idx,
            'Feature_Start': current_start,
            'Feature_End':   feature_end,
            'Target_Start':  target_start,
            'Target_End':    target_end,
        })
        current_start += timedelta(days=slide_days)
        fold_idx += 1

    if len(folds) < 3:
        raise ValueError(f"Chỉ có {len(folds)} fold, cần ít nhất 3 (train/val/test)")

    return pd.DataFrame(folds)


def assign_split_type(fold_id, total_folds):
    if fold_id == total_folds:   return 'TEST'
    if fold_id == total_folds-1: return 'VAL'
    return 'TRAIN'


# ════════════════════════════════════════════════════════════
# 2. FIT ENCODERS (chỉ chạy trên train sessions)
# ════════════════════════════════════════════════════════════
def find_elbow_n(series, max_n=50):
    counts = series.value_counts(normalize=True)
    max_n  = min(max_n, len(counts))
    records = []
    for n in range(1, max_n + 1):
        top_probs  = counts.iloc[:n]
        other_prob = 1 - top_probs.sum()
        all_probs  = list(top_probs) + ([other_prob] if other_prob > 1e-9 else [])
        gini = 1 - sum(p**2 for p in all_probs)
        records.append({'n': n, 'gini': gini})
    df_res = pd.DataFrame(records)
    df_res['marginal'] = df_res['gini'].diff().abs().fillna(0)
    gini_range = df_res['gini'].max() - df_res['gini'].min()
    threshold  = gini_range * 0.01
    elbow_n = max_n
    for i, row in df_res.iterrows():
        if row['n'] >= 2 and df_res.loc[i:, 'marginal'].max() < threshold:
            elbow_n = int(row['n'])
            break
    return elbow_n


def fit_encoders(df_train_sessions):
    """
    Fit tất cả encoders và models từ train sessions.
    Trả về artifacts dict để dùng lại cho kaggle test.
    """
    print("=" * 55)
    print("FIT ENCODERS (train sessions only)")
    print("=" * 55)

    # ── Top-N categorical mapping ──────────────────────────
    print("\n[1] Top-N categorical mapping...")
    fitted_cat_mappings = {}
    for col in CAT_COLS:
        n = find_elbow_n(df_train_sessions[col], max_n=50)
        top_cats = df_train_sessions[col].value_counts(dropna=False).nlargest(n).index
        fitted_cat_mappings[col] = set(top_cats)
        print(f"  [{col:<35}]: N={n}")

    # ── Target encoders ────────────────────────────────────
    print("\n[2] Target encoders...")
    global_mean_rev = df_train_sessions['totals_transactionRevenue'].mean()
    target_enc_map  = {}
    for col in CAT_ENCODE_COLS:
        target_enc_map[col] = (
            df_train_sessions.groupby(col)['totals_transactionRevenue']
            .mean().to_dict()
        )
        print(f"  [{col:<35}]: {len(target_enc_map[col])} categories")
    print(f"  global_mean_rev = {global_mean_rev:.4f}")

    # ── BG/NBD + Gamma-Gamma ───────────────────────────────
    print("\n[3] BG/NBD + Gamma-Gamma...")
    train_transactions = df_train_sessions[
        df_train_sessions['totals_transactionRevenue'] > 0
    ][['fullVisitorId', 'date', 'totals_transactionRevenue']].copy()

    train_summary = summary_data_from_transaction_data(
        train_transactions,
        customer_id_col='fullVisitorId',
        datetime_col='date',
        monetary_value_col='totals_transactionRevenue',
        observation_period_end=df_train_sessions['date'].max(),
        freq='D',
    )

    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(train_summary['frequency'], train_summary['recency'], train_summary['T'])
    print(f"  BG/NBD fitted. Buyers: {len(train_summary):,}")

    gg_data = train_summary[train_summary['frequency'] > 0]
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(gg_data['frequency'], gg_data['monetary_value'])
    print(f"  Gamma-Gamma fitted. Repeat buyers: {len(gg_data):,}")

    # Hibernation threshold (từ train data)
    visit_stats_train = df_train_sessions.groupby('fullVisitorId').agg(
        last_visit_date =('date', 'max'),
        first_visit_date=('date', 'min'),
        visit_count     =('visitId', 'count'),
    ).reset_index()
    visit_stats_train['lifetime_days'] = (
        visit_stats_train['last_visit_date'] - visit_stats_train['first_visit_date']
    ).dt.days
    repeat_users = visit_stats_train[visit_stats_train['visit_count'] > 1].copy()
    repeat_users['inter_visit_time'] = (
        repeat_users['lifetime_days'] / (repeat_users['visit_count'] - 1)
    )
    ivt_mean = repeat_users['inter_visit_time'].mean()
    ivt_std  = repeat_users['inter_visit_time'].std()
    hibernate_threshold = ivt_mean + 2 * ivt_std
    print(f"\n  Hibernate threshold: {hibernate_threshold:.1f} days")

    artifacts = {
        'fitted_cat_mappings':  fitted_cat_mappings,
        'target_enc_map':       target_enc_map,
        'global_mean_rev':      global_mean_rev,
        'bgf':                  bgf,
        'ggf':                  ggf,
        'hibernate_threshold':  hibernate_threshold,
    }

    # Lưu artifacts để dùng lại cho kaggle test
    with open(ARTIFACTS_PATH, 'wb') as f:
        pickle.dump(artifacts, f)
    print(f"\n✅ Artifacts saved: {ARTIFACTS_PATH}")

    return artifacts


# ════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING CHO MỘT FEATURE WINDOW
#    Dùng chung cho cả train folds và kaggle test window
# ════════════════════════════════════════════════════════════
def engineer_features(df_feat, feature_end_date, artifacts, feature_window_days):
    """
    Tính toàn bộ features cho một feature window.
    df_feat: session-level data trong window
    feature_end_date: mốc thời gian cuối window (để tính recency)
    artifacts: dict chứa fitted encoders
    feature_window_days: số ngày của window (để fillna days_since_purchase)
    """
    fitted_cat_mappings = artifacts['fitted_cat_mappings']
    target_enc_map      = artifacts['target_enc_map']
    global_mean_rev     = artifacts['global_mean_rev']
    hibernate_threshold = artifacts['hibernate_threshold']

    # Khôi phục BTYD models (hỗ trợ cả dạng object và dạng params)
    if 'bgf' in artifacts:
        bgf = artifacts['bgf']
    else:
        bgf = BetaGeoFitter()
        bgf.params_ = artifacts['bgf_params']

    if 'ggf' in artifacts:
        ggf = artifacts['ggf']
    else:
        ggf = GammaGammaFitter()
        ggf.params_ = artifacts['ggf_params']

    # ── B. Apply top-N grouping ────────────────────────────
    for col, keep_set in fitted_cat_mappings.items():
        df_feat[col] = np.where(df_feat[col].isin(keep_set), df_feat[col], 'Other')

    # ── B2. Target encode session-level ───────────────────
    for col, enc_map in target_enc_map.items():
        df_feat[f'{col}_enc'] = df_feat[col].map(enc_map).fillna(global_mean_rev)

    # ── B3. Diversity features ─────────────────────────────
    diversity_df = (
        df_feat.groupby('fullVisitorId')
        .agg({col: pd.Series.nunique for col in DIVERSITY_COLS})
        .rename(columns={
            'geoNetwork_city':       'city_nunique',
            'geoNetwork_country':    'country_nunique',
            'trafficSource_source':  'source_nunique',
            'trafficSource_medium':  'medium_nunique',
            'channelGrouping':       'channel_nunique',
            'device_browser':        'browser_nunique',
            'device_operatingSystem':'os_nunique',
        })
        .reset_index()
    )
    diversity_df['diversity_score'] = diversity_df[[
        'city_nunique','country_nunique','source_nunique',
        'medium_nunique','channel_nunique','browser_nunique','os_nunique',
    ]].sum(axis=1)

    # ── C. Aggregate → user-level ──────────────────────────
    agg_dict = {
        'visitId':                      'count',
        'visitNumber':                  'max',
        'totals_hits':                  ['sum', 'mean'],
        'totals_pageviews':             ['sum', 'mean'],
        'totals_timeOnSite':            ['sum', 'mean'],
        'totals_bounces':               'mean',
        'totals_sessionQualityDim':     ['mean', 'max'],
        'totals_transactionRevenue':    ['sum', 'mean', 'max'],
        'totals_transactions':          'sum',
        'device_isMobile':              'mean',
        'device_deviceCategory_enc':    ['mean', 'last'],
        'geoNetwork_city_enc':          ['mean', 'max'],
        'geoNetwork_country_enc':       ['mean', 'max'],
        'geoNetwork_region_enc':        'mean',
        'geoNetwork_metro_enc':         'mean',
        'geoNetwork_subContinent_enc':  'mean',
        'channelGrouping_enc':          ['mean', 'max', 'last'],
        'trafficSource_medium_enc':     ['mean', 'max', 'last'],
        'trafficSource_source_enc':     ['mean', 'max', 'last'],
        'trafficSource_campaign_enc':   ['mean', 'max'],
        'device_browser_enc':           ['mean', 'last'],
        'device_operatingSystem_enc':   ['mean', 'last'],
    }
    named_aggs = {
        'pct_weekend': pd.NamedAgg(
            column='Date_Dayofweek',
            aggfunc=lambda x: (x >= 5).mean()
        ),
        'pct_evening': pd.NamedAgg(
            column='Date_Hour',
            aggfunc=lambda x: (x >= 20).mean()
        ),
    }

    X_user = df_feat.groupby('fullVisitorId').agg({**agg_dict})
    X_user.columns = ['_'.join(str(c) for c in col).strip('_') for col in X_user.columns]
    X_user = X_user.reset_index()

    X_named = df_feat.groupby('fullVisitorId').agg(**named_aggs).reset_index()
    X_user  = X_user.merge(X_named, on='fullVisitorId', how='left')
    X_user  = X_user.rename(columns={'totals_hits_mean': 'funnel_depth'})
    X_user['has_purchased'] = (X_user['totals_transactionRevenue_sum'] > 0).astype(int)
    X_user = X_user.merge(diversity_df, on='fullVisitorId', how='left')

    # ── D. Recency, Lifetime, Hibernation, Velocity ────────
    visit_stats = df_feat.groupby('fullVisitorId').agg(
        last_visit_date =('date', 'max'),
        first_visit_date=('date', 'min'),
        visit_count     =('visitId', 'count'),
    ).reset_index()
    visit_stats['recency_days']  = (feature_end_date - visit_stats['last_visit_date']).dt.days
    visit_stats['lifetime_days'] = (
        visit_stats['last_visit_date'] - visit_stats['first_visit_date']
    ).dt.days
    X_user = X_user.merge(
        visit_stats[['fullVisitorId', 'recency_days', 'lifetime_days']],
        on='fullVisitorId', how='left'
    )

    # days_since_purchase
    last_purchase = (
        df_feat[df_feat['totals_transactionRevenue'] > 0]
        .groupby('fullVisitorId')['date'].max().reset_index()
        .rename(columns={'date': 'last_purchase_date'})
    )
    last_purchase['days_since_purchase'] = (
        feature_end_date - last_purchase['last_purchase_date']
    ).dt.days
    X_user = X_user.merge(
        last_purchase[['fullVisitorId', 'days_since_purchase']],
        on='fullVisitorId', how='left'
    )
    X_user['days_since_purchase'] = X_user['days_since_purchase'].fillna(feature_window_days + 1)

    # Hibernation flag — dùng threshold đã fit từ train
    X_user['hibernation_flag'] = (X_user['recency_days'] > hibernate_threshold).astype(int)

    # Velocity
    date_30d_ago = feature_end_date - pd.Timedelta(days=30)
    date_90d_ago = feature_end_date - pd.Timedelta(days=90)

    v30 = df_feat[df_feat['date'] >= date_30d_ago].groupby('fullVisitorId').agg(
        pv_last30      =('totals_pageviews', 'mean'),
        hits_last30    =('totals_hits',      'mean'),
        sessions_last30=('visitId',          'count'),
    ).reset_index()
    v90 = df_feat[
        (df_feat['date'] >= date_90d_ago) & (df_feat['date'] < date_30d_ago)
    ].groupby('fullVisitorId').agg(
        pv_prev90      =('totals_pageviews', 'mean'),
        hits_prev90    =('totals_hits',      'mean'),
        sessions_prev90=('visitId',          'count'),
    ).reset_index()

    vel = pd.merge(v90, v30, on='fullVisitorId', how='outer')
    vel = vel[vel['fullVisitorId'].isin(X_user['fullVisitorId'])].fillna(0)
    vel['vel_pageviews'] = (vel['pv_last30']       + 1) / (vel['pv_prev90']       + 1)
    vel['vel_hits']      = (vel['hits_last30']     + 1) / (vel['hits_prev90']     + 1)
    vel['vel_sessions']  = (vel['sessions_last30'] + 1) / (vel['sessions_prev90'] + 1)
    X_user = X_user.merge(
        vel[['fullVisitorId', 'vel_pageviews', 'vel_hits', 'vel_sessions']],
        on='fullVisitorId', how='left'
    )
    X_user[['vel_pageviews', 'vel_hits', 'vel_sessions']] = (
        X_user[['vel_pageviews', 'vel_hits', 'vel_sessions']].fillna(1.0)
    )

    # ── E. BG/NBD + Gamma-Gamma features ──────────────────
    fold_transactions = df_feat[df_feat['totals_transactionRevenue'] > 0][
        ['fullVisitorId', 'date', 'totals_transactionRevenue']
    ].copy()

    if len(fold_transactions) > 10:
        fold_summary = summary_data_from_transaction_data(
            fold_transactions,
            customer_id_col='fullVisitorId',
            datetime_col='date',
            monetary_value_col='totals_transactionRevenue',
            observation_period_end=feature_end_date,
            freq='D',
        )
        fold_summary['p_alive'] = bgf.conditional_probability_alive(
            fold_summary['frequency'], fold_summary['recency'], fold_summary['T']
        )
        fold_summary['expected_purchases_92d'] = (
            bgf.conditional_expected_number_of_purchases_up_to_time(
                92, fold_summary['frequency'], fold_summary['recency'], fold_summary['T']
            )
        )
        fold_summary = fold_summary.reset_index()

        repeat = fold_summary[fold_summary['frequency'] > 0][
            ['fullVisitorId', 'frequency', 'monetary_value']
        ].copy()
        if len(repeat) > 0:
            repeat['expected_monetary'] = ggf.conditional_expected_average_profit(
                repeat['frequency'], repeat['monetary_value']
            )
            fold_summary = fold_summary.merge(
                repeat[['fullVisitorId', 'expected_monetary']],
                on='fullVisitorId', how='left'
            )
        else:
            fold_summary['expected_monetary'] = 0.0

        fold_summary['expected_monetary'] = fold_summary['expected_monetary'].fillna(0)
        X_user = X_user.merge(
            fold_summary[['fullVisitorId', 'p_alive',
                          'expected_purchases_92d', 'expected_monetary']],
            on='fullVisitorId', how='left'
        )
    else:
        X_user['p_alive']                = 0.0
        X_user['expected_purchases_92d'] = 0.0
        X_user['expected_monetary']      = 0.0

    X_user[['p_alive', 'expected_purchases_92d', 'expected_monetary']] = (
        X_user[['p_alive', 'expected_purchases_92d', 'expected_monetary']].fillna(0)
    )

    # ── E2. Interaction & Ratio Features ──────────────────
    X_user['interaction_hits_visits'] = np.log1p(
        X_user['totals_hits_sum'] * X_user['visitNumber_max']
    )
    X_user['interaction_time_visits'] = np.log1p(
        X_user['totals_timeOnSite_mean'] * X_user['visitId_count']
    )
    X_user['pageviews_per_session']  = X_user['totals_pageviews_sum'] / X_user['visitId_count']
    X_user['ratio_revenue_hits']     = (
        X_user['totals_transactionRevenue_sum'] / (X_user['totals_hits_sum'] + 1)
    )
    X_user['is_returning_visitor']   = (X_user['visitNumber_max'] > 1).astype(int)

    return X_user


# ════════════════════════════════════════════════════════════
# 4. MODE 1 — TRAIN DATA (gộp toàn bộ, không chia fold)
# ════════════════════════════════════════════════════════════
def run_train_mode():
    print("=" * 60)
    print("MODE 1 — KAGGLE TRAIN DATA (Gộp toàn bộ)")
    print("=" * 60)

    # Load
    full_df = pd.read_pickle(TRAIN_INPUT_PATH)
    if 'target_log_revenue' in full_df.columns:
        full_df = full_df.drop(columns=['target_log_revenue',
                                        'transactionRevenue',
                                        'totals_totalTransactionRevenue'], errors='ignore')
    print(f"Loaded: {full_df.shape}")
    print(f"Date range: {full_df['date'].min().date()} → {full_df['date'].max().date()}")

    # Fit encoders từ TOÀN BỘ train data (không chia fold)
    print(f"\nTrain sessions: {len(full_df):,} rows")
    artifacts = fit_encoders(full_df)

    # Feature engineering cho TOÀN BỘ data
    print(f"\nFeature engineering toàn bộ data...")
    
    feature_end_date = full_df['date'].max()
    feature_start_date = full_df['date'].min()
    feature_window_days = (feature_end_date - feature_start_date).days
    
    print(f"Feature window: {feature_start_date.date()} → {feature_end_date.date()} ({feature_window_days} days)")

    df_feat = full_df.copy()
    X_user = engineer_features(df_feat, feature_end_date, artifacts, feature_window_days)

    # F. Target y (tổng revenue trong toàn bộ period)
    y_user = (
        full_df
        .groupby('fullVisitorId')['totals_transactionRevenue']
        .sum().reset_index()
        .rename(columns={'totals_transactionRevenue': 'target_revenue'})
    )
    y_user['target_log_revenue'] = np.log1p(y_user['target_revenue'])

    # Merge X + y
    train_data = pd.merge(X_user, y_user, on='fullVisitorId', how='left')
    train_data['target_revenue'] = train_data['target_revenue'].fillna(0.0)
    train_data['target_log_revenue'] = np.log1p(train_data['target_revenue'])

    # Lưu output
    train_data.to_pickle(TRAIN_OUTPUT_PATH)

    # Statistics
    buyers = (train_data['target_revenue'] > 0).sum()
    hibernating = (train_data['hibernation_flag'] > 0).sum()

    print(f"\n✅ HOÀN TẤT!")
    print(f"  Shape: {train_data.shape}")
    print(f"  Users: {len(train_data):,} | Buyers: {buyers} | Hibernating: {hibernating:,}")
    print(f"  Output: {TRAIN_OUTPUT_PATH}\n")


# ════════════════════════════════════════════════════════════
# 5. MODE 2 — KAGGLE TEST DATA (không có target)
# ════════════════════════════════════════════════════════════
def run_kaggle_test_mode():
    print("=" * 60)
    print("MODE 2 — KAGGLE TEST DATA")
    print("=" * 60)

    # Load artifacts đã fit từ train
    if not os.path.exists(ARTIFACTS_PATH):
        raise FileNotFoundError(
            f"Không tìm thấy {ARTIFACTS_PATH}. "
            f"Hãy chạy MODE 1 (--mode train) trước để fit encoders."
        )
    with open(ARTIFACTS_PATH, 'rb') as f:
        artifacts = pickle.load(f)
    print(f"✅ Loaded artifacts từ {ARTIFACTS_PATH}")

    # Load Kaggle test data
    kaggle_test_df = pd.read_pickle(KAGGLE_TEST_INPUT_PATH)
    print(f"Kaggle test loaded: {kaggle_test_df.shape}")
    print(f"Date range: {kaggle_test_df['date'].min().date()} → {kaggle_test_df['date'].max().date()}")

    # Kaggle test không chia fold — dùng toàn bộ làm một feature window
    feature_start    = kaggle_test_df['date'].min()
    feature_end_date = kaggle_test_df['date'].max()
    feature_window_days = (feature_end_date - feature_start).days
    print(f"Feature window: {feature_start.date()} → {feature_end_date.date()} ({feature_window_days} days)")

    # Apply feature engineering (không có bước F vì không có target)
    df_feat = kaggle_test_df.copy()
    X_user  = engineer_features(df_feat, feature_end_date, artifacts, feature_window_days)

    # Lưu output
    X_user.to_pickle(KAGGLE_TEST_OUTPUT_PATH)
    print(f"\n✅ Saved: {X_user.shape} → {KAGGLE_TEST_OUTPUT_PATH}")
    print(f"\nLưu ý:")
    print(f"  - Không có cột target_revenue / target_log_revenue (bình thường)")
    print(f"  - Encoders được load từ train — KHÔNG fit lại trên test")
    print(f"  - File này sẵn sàng cho pipeline_predict_kaggle_test.py")


# ════════════════════════════════════════════════════════════
# 6. ENTRYPOINT
# ════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature Engineering Pipeline')
    parser.add_argument(
        '--mode',
        choices=['train', 'kaggle_test'],
        required=True,
        help='train: xử lý Kaggle train data (có target, chia fold)\n'
             'kaggle_test: xử lý Kaggle test data (không có target)',
    )
    args = parser.parse_args()

    if args.mode == 'train':
        run_train_mode()
    else:
        run_kaggle_test_mode()