"""
Data Processing Pipeline Package
Gộp JSON parsing + Data cleaning & preprocessing

Usage in Jupyter Notebook:
    from process_data import run_full_pipeline, process_json_chunked, preprocess_data
    
    # Run full pipeline
    df = run_full_pipeline('../data/test_v2.pkl', stage='test')
    
    # Or use individual functions
    df = process_json_chunked(df)
    df = preprocess_data(df)
"""

from .preprocess import (
    # Main functions
    run_full_pipeline,
    process_json_chunked,
    preprocess_data,
    
    # Phase 1 functions
    safe_parse,
    process_chunk_json,
    
    # Phase 2 functions
    clean_revenue,
    detect_and_remove_junk,
    fill_missing_values,
    normalize_traffic_source,
    
    # Utility
    get_pipeline_info,
)

from .feature_engineering import (
    engineer_features,
    assign_split_type,
    generate_time_folds,
    fit_encoders
)

__version__ = '1.0.0'
__author__ = 'DDM Project'

__all__ = [
    'run_full_pipeline',
    'process_json_chunked',
    'preprocess_data',
    'safe_parse',
    'process_chunk_json',
    'clean_revenue',
    'detect_and_remove_junk',
    'fill_missing_values',
    'normalize_traffic_source',
    'get_pipeline_info',
    'engineer_features',
    'assign_split_type',
    'generate_time_folds',
    'fit_encoders'
]
