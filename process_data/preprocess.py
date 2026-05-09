
import pandas as pd
import numpy as np
import json
import gc
import ast
from pathlib import Path


# ════════════════════════════════════════════════════════════
# 0. CONFIG
# ════════════════════════════════════════════════════════════

CHUNK_SIZE = 50000

# Junk detection
INTERNAL_DOMAINS        = ['google']
INTERNAL_CITIES         = ['Mountain View']
BOT_HITS_THRESHOLD      = 500
BOT_PAGEVIEWS_THRESHOLD = 500
BOT_BROWSER_PATTERNS    = ['CT_JOB_ID', 'ecgiwap']

# Missing values
WEIRD_VALUES = [
    "unknown.unknown",
    "(not set)",
    "not available in demo dataset",
    "(not provided)",
    "(none)",
]
CAMPAIGN_FILL   = 'no_campaign'
MEDIUM_FILL     = 'none'
HIGH_NULL_FILL  = 'unknown'
LOW_NULL_FILL   = 'unknown'


COLS_TO_DROP = [
    'trafficSource_adwordsClickInfo.gclId',
    'geoNetwork_networkDomain',
    'trafficSource_keyword',
    'trafficSource_referralPath',
    'trafficSource_adContent',
    'trafficSource_adwordsClickInfo.slot',
    'trafficSource_adwordsClickInfo.adNetworkType',
]

# Traffic source normalization
SOURCE_NORMALIZE_PATTERNS = [
    ('google',    ['google']),
    ('youtube',   ['youtube', r'yt\.be']),
    ('yahoo',     ['yahoo']),
    ('facebook',  ['facebook', r'fb\.com', r'm\.facebook']),
    ('bing',      ['bing']),
    ('reddit',    ['reddit']),
    ('pinterest', ['pinterest']),
    ('baidu',     ['baidu']),
    ('direct',    [r'\(direct\)', 'direct']),
]

# Revenue
MICRO_CONVERSION_FACTOR = 1e6
USE_LOG_TRANSFORM       = True

VERBOSE = True


# ════════════════════════════════════════════════════════════
# 1. PHASE 1 — JSON PARSING & FLATTENING
# ════════════════════════════════════════════════════════════

def safe_parse(x):
    """Parse JSON an toàn, ưu tiên tốc độ."""
    if not isinstance(x, str) or x == "" or x == "{}" or x == "[]":
        return {}
    try:
        return json.loads(x)
    except:
        # Chỉ khi json.loads lỗi mới dùng ast (chậm hơn)
        try:
            import ast
            return ast.literal_eval(x)
        except:
            return {}


def process_chunk_json(df_chunk):
    """Phiên bản tối ưu hóa: Dùng json_normalize để flatten siêu tốc."""
    df_chunk = df_chunk.copy()

    # A. Flatten 4 cột JSON chính
    for col in ["device", "geoNetwork", "totals", "trafficSource"]:
        if col in df_chunk.columns:
            # Parse chuỗi JSON thành list of dicts
            dicts = df_chunk[col].map(safe_parse).tolist()
            # Dùng json_normalize cực nhanh
            col_df = pd.json_normalize(dicts).set_index(df_chunk.index)
            col_df.columns = [f"{col}_{sub}" for sub in col_df.columns]
            
            df_chunk = pd.concat([df_chunk.drop(columns=[col]), col_df], axis=1)

    # B. customDimensions (Lấy value đầu tiên)
    if "customDimensions" in df_chunk.columns:
        def extract_cd(x):
            val = safe_parse(x)
            return val[0].get('value') if isinstance(val, list) and len(val) > 0 else np.nan
        df_chunk["customDimensions_value"] = df_chunk["customDimensions"].map(extract_cd)
        df_chunk.drop(columns=["customDimensions"], inplace=True)

    # C. hits (Đếm số lượng)
    if "hits" in df_chunk.columns:
        df_chunk["hits_count"] = df_chunk["hits"].map(lambda x: len(safe_parse(x)))
        df_chunk.drop(columns=["hits"], inplace=True)

    # D. Dọn giá trị rác & Ép kiểu số
    df_chunk = df_chunk.replace(WEIRD_VALUES, np.nan)
    
    num_cols = [
        "visitNumber", "totals_hits", "totals_pageviews", 
        "totals_transactionRevenue", "totals_timeOnSite", 
        "totals_bounces", "totals_sessionQualityDim",
        "totals_transactions"
    ]
    for col in num_cols:
        if col in df_chunk.columns:
            df_chunk[col] = pd.to_numeric(df_chunk[col], errors="coerce").fillna(0)

    # E. Parse date
    if "date" in df_chunk.columns:
        df_chunk["date"] = pd.to_datetime(df_chunk["date"].astype(str), format="%Y%m%d", errors="coerce")

    return df_chunk


def process_json_chunked(df, chunk_size=None):
    """Xử lý toàn bộ dataframe theo chunk để tiết kiệm RAM."""
    if chunk_size is None:
        chunk_size = CHUNK_SIZE

    chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    processed_list = []

    print(f"\n🚀 PHASE 1: JSON PARSING & FLATTENING")
    print(f"   {len(df):,} dòng | {len(chunks)} chunks\n")

    for i, chunk in enumerate(chunks):
        if VERBOSE:
            print(f"   Chunk {i+1}/{len(chunks)}")
        processed_list.append(process_chunk_json(chunk))
        gc.collect()

    df_out = pd.concat(processed_list, axis=0, ignore_index=True)
    print(f"   ✅ JSON parsing hoàn tất!\n")
    return df_out


# ════════════════════════════════════════════════════════════
# 2. PHASE 2 — DATA CLEANING & PREPROCESSING
# ════════════════════════════════════════════════════════════

def clean_revenue(df):
    """Chuẩn hóa revenue — chia micro units, tạo log target."""
    if VERBOSE:
        print("   > Cleaning revenue...")

    if 'totals_transactionRevenue' not in df.columns:
        df['totals_transactionRevenue'] = 0.0

    # Chia micro nếu cần (giá trị > 1M thường là micro units)
    if df['totals_transactionRevenue'].max() > 1_000_000:
        df['totals_transactionRevenue'] = df['totals_transactionRevenue'] / MICRO_CONVERSION_FACTOR

    df['totals_transactionRevenue'] = df['totals_transactionRevenue'].fillna(0.0)

    if USE_LOG_TRANSFORM:
        df['target_log_revenue'] = np.log1p(df['totals_transactionRevenue'])

    return df


def detect_and_remove_junk(df):
    """
    Loại bỏ internal traffic, bot, tracking errors, fake whales.
    """
    if VERBOSE:
        print("   > Detecting & removing junk...")

    internal_mask = (
        (df['geoNetwork_networkDomain'].astype(str)
             .str.contains('|'.join(INTERNAL_DOMAINS), na=False)) |
        (df['geoNetwork_city'].isin(INTERNAL_CITIES))
    )

    bot_mask = (
        ((df['totals_hits'] > BOT_HITS_THRESHOLD) &
         (df['totals_timeOnSite'] == 0)) |
        ((df['totals_pageviews'] > BOT_PAGEVIEWS_THRESHOLD) &
         (df['totals_bounces'] == 1)) |
        (df['device_browser'].str.contains(
            '|'.join(BOT_BROWSER_PATTERNS), na=False, regex=True))
    )

    tracking_error_mask = (
        (df['totals_bounces'] == 1) & (df['totals_transactionRevenue'] > 0)
    )

    fake_whale_mask = (
        (df['totals_pageviews'] <= 1) &
        (df['totals_timeOnSite'] == 0) &
        (df['totals_transactionRevenue'] > 0)
    )

    junk_mask = internal_mask | bot_mask | tracking_error_mask | fake_whale_mask
    n_removed = junk_mask.sum()

    df_clean = df[~junk_mask].reset_index(drop=True)

    if VERBOSE:
        print(f"      Loại bỏ {n_removed:,} sessions ({n_removed/len(df)*100:.2f}%)")
        print(f"      Còn lại: {len(df_clean):,} sessions")

    return df_clean


def drop_low_signal_cols(df):
    """
    Drop các cột high cardinality / low signal.
    Chạy SAU detect_and_remove_junk (vì junk detection cần networkDomain).
    """
    if VERBOSE:
        print("   > Dropping low-signal columns...")

    existing = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=existing)

    if VERBOSE:
        print(f"      Dropped: {existing}")

    return df


def fill_missing_values(df):
    """Xử lý missing values theo chiến lược từng nhóm cột."""
    if VERBOSE:
        print("   > Filling missing values...")

    if 'trafficSource_campaign' in df.columns:
        df['trafficSource_campaign'] = df['trafficSource_campaign'].fillna(CAMPAIGN_FILL)

    if 'trafficSource_medium' in df.columns:
        df['trafficSource_medium'] = df['trafficSource_medium'].fillna(MEDIUM_FILL)

    high_null_cols = ['geoNetwork_metro', 'geoNetwork_city', 'geoNetwork_region']
    for col in high_null_cols:
        if col in df.columns:
            df[col] = df[col].fillna(HIGH_NULL_FILL)

    low_null_cols = [
        'device_operatingSystem', 'geoNetwork_country', 'geoNetwork_subContinent',
        'geoNetwork_continent', 'trafficSource_source', 'device_browser',
    ]
    for col in low_null_cols:
        if col in df.columns:
            df[col] = df[col].fillna(LOW_NULL_FILL)

    return df


def normalize_traffic_source(df):
    """Chuẩn hóa trafficSource_source bằng regex grouping."""
    if VERBOSE:
        print("   > Normalizing traffic source...")

    if 'trafficSource_source' not in df.columns:
        return df

    source_col = df['trafficSource_source'].str.lower()
    conditions, choices = [], []

    for normalized_name, patterns in SOURCE_NORMALIZE_PATTERNS:
        conditions.append(
            source_col.str.contains('|'.join(patterns), na=False, regex=True)
        )
        choices.append(normalized_name)

    df['trafficSource_source'] = np.select(conditions, choices, default=source_col)
    return df


def add_time_features(df):
    """
    Tạo time features từ cột date.
    Feature engineering cần Date_Hour và Date_Dayofweek.
    """
    if 'date' not in df.columns:
        return df

    if VERBOSE:
        print("   > Adding time features...")

    df['Date_Hour']      = df['date'].dt.hour
    df['Date_Dayofweek'] = df['date'].dt.dayofweek   # 0=Mon, 6=Sun
    df['Date_Month']     = df['date'].dt.month
    df['Date_Year']      = df['date'].dt.year

    return df


def add_network_type(df):
    """
    Tạo network_type từ các tín hiệu session.
    """
    if VERBOSE:
        print("   > Adding network_type...")

    if 'device_isMobile' not in df.columns and 'device_deviceCategory' in df.columns:
        df['device_isMobile'] = (df['device_deviceCategory'] == 'mobile').astype(int)

    return df


def preprocess_data(df):
    print("\n📊 PHASE 2: DATA CLEANING & PREPROCESSING\n")

    df = clean_revenue(df)
    df = detect_and_remove_junk(df)
    df = drop_low_signal_cols(df)       # sau junk detection
    df = fill_missing_values(df)
    df = normalize_traffic_source(df)
    df = add_time_features(df)
    df = add_network_type(df)

    print("\n   ✅ Preprocessing hoàn tất!\n")
    return df


# ════════════════════════════════════════════════════════════
# 3. MAIN PIPELINE
# ════════════════════════════════════════════════════════════

def run_full_pipeline(input_path, output_dir='../data/', stage='train', chunk_size=None):
    print("=" * 70)
    print(f"🔄 FULL DATA PROCESSING PIPELINE ({stage.upper()})")
    print("=" * 70)

    # 1. Load
    print(f"\n📖 Loading: {input_path}")
    df = pd.read_pickle(input_path)
    print(f"   Shape: {df.shape}")

    # 2. JSON parsing
    df = process_json_chunked(df, chunk_size=chunk_size)

    # 3. Preprocessing
    df = preprocess_data(df)

    # 4. Save
    output_path = Path(output_dir) / f'{stage}_preprocessed.pkl'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(output_path)

    print(f"✅ HOÀN TẤT!")
    print(f"   Output:  {output_path}")
    print(f"   Shape:   {df.shape}")
    print(f"   Nulls:   {df.isnull().sum().sum():,}")
    print(f"   Memory:  {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB\n")

    return df


# ════════════════════════════════════════════════════════════
# 4. UTILITY
# ════════════════════════════════════════════════════════════

def get_pipeline_info():
    print(f"""
🔧 PIPELINE CONFIGURATION
{'='*50}
Chunk size:         {CHUNK_SIZE:,}
Bot hits threshold: {BOT_HITS_THRESHOLD}
Bot pv threshold:   {BOT_PAGEVIEWS_THRESHOLD}
Cols to drop:       {len(COLS_TO_DROP)}
Log transform:      {USE_LOG_TRANSFORM}
Micro conversion:   {MICRO_CONVERSION_FACTOR}
""")


if __name__ == "__main__":
    print("Pipeline module loaded.")
    print("Usage: from process_data import run_full_pipeline")
    print("       df = run_full_pipeline('../data/test_v2.pkl', stage='test')")