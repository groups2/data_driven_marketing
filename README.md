# Data-Driven Marketing Campaigns with Google Analytics: Historical Segmentation, Predictive Customer Value and Experimentation

## 📌 Project Overview
This repository contains an end-to-end machine learning framework designed to predict Customer Lifetime Value (CLV) and optimize marketing strategies. Utilizing the **Google Merchandise Store** dataset (extracted from Google Analytics clickstream logs), the project builds a robust, business-oriented analytical pipeline capable of processing massive, semi-structured session data to model customer conversion and forecast revenue generation.

**Team (Group 2, DSEB 65B - MFE, NEU):**
- Nguyen Thi Van Anh
- Vu Bui Dinh Tung
- Dao Khanh Linh
- Nguyen Hong Nhung
- Thieu Dieu Thuy

**Course:** DDM -- Data-Driven Marketing

## 🎯 Objectives
- **Behavioral Analytics:** Transform high-dimensional, semi-structured JSON clickstream data into robust customer-level representations and funnel indicators.
- **Customer Segmentation:** Apply base-aware K-Means clustering to distinguish actionable historical customer cohorts and identify behavioral patterns.
- **Conversion & Revenue Prediction:** Effectively overcome extreme transaction imbalance (~1.08% buyer rate) using a Two-Stage XGBoost hurdle architecture across a 7-fold sliding-window cross-validation scheme.
- **Marketing Experimentation & Optimization:** Bridge the gap between machine learning metrics and business KPIs through causal A/B testing designs (e.g., funnel Drop-off campaigns) and segment-level expected profitability optimizations (ROMI).

## 📂 Repository Structure

```text
.
├── data/                       # Raw and processed datasets (ignored in git)
├── Notebooks/                  # Jupyter Notebooks for exploration, modeling, and analysis
│   ├── 01_data_prep/           # Data merging, EDA, JSON flattening, and target analysis
│   ├── 02_modeling/            # Feature engineering, train-test splitting, and model training
│   ├── 03_inference/           # Inference on test sets and processing predictions
│   └── 04_analysis/            # Customer segmentation, CLV sensitivity, and business motifs
├── process_data/               # Python modules for data preprocessing
│   ├── preprocess.py           # Chunk-based JSON parsing, junk detection, preprocessing
│   └── feature_engineering.py  # Advanced FE (BTYD models, velocity, target encoding)
├── train_model/                # Model training and evaluation scripts
│   └── model.py                # XGBoost Two-Stage Hurdle architecture and Isotonic calibration
├── reports/                    # Generated plots, metrics, and business reports
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
└── report.tex                  # Comprehensive academic LaTeX report
```

## 🧠 Methodology & Architecture

### 1. Data Preprocessing (`process_data/preprocess.py`)
- **JSON Flattening:** Optimized, chunk-based parsing of nested tracking logs (`device`, `geoNetwork`, `totals`, `trafficSource`, and granular `hits` arrays).
- **Junk Detection:** Algorithmic filtering of internal traffic, bot behavior (e.g., >500 hits with zero time on site), tracking errors, and fake whales.
- **Data Normalization:** Regex-based grouping of traffic sources and parsing of extreme macro/micro revenue units into normalized log-transformed scales.

### 2. Feature Engineering & Historical Segmentation
- **Base-Aware K-Means Scoring:** Historical users are partitioned via a dual-pathway approach using a 1-5 standardized score scaling.
- **BTYD (Buy Till You Die) Models:** Integration of `lifetimes` BG/NBD and Gamma-Gamma fitters to extract probabilistic behavioral signals (`p_alive`, `expected_purchases_92d`, `expected_monetary`).
- **Session Velocity & Funnel Depth:** Engineered momentum indicators comparing 30-day versus 90-day historical behavior (`vel_pageviews`, `vel_hits`, `vel_sessions`) and sequential funnel milestones.
- **Target Encoding & Categorical Grouping:** Top-N mapping for high cardinality geographical/device data and global-mean smoothed target encoding.

### 3. Two-Stage Modeling Framework (`train_model/model.py`)
To prevent temporal leakage, models are trained using a **Sliding-Window Cross-Validation Scheme (7 Folds)** and evaluate a severely zero-inflated target variable via a customized **Two-Stage Hurdle Architecture**:
- **Stage 1 (Classification - Purchase Propensity):** An `XGBClassifier` predicts the probability of conversion (`p_buy`). Evaluated using **PR-AUC (0.0894 vs baseline 0.000744)**, configured with `scale_pos_weight` for imbalance, and strictly calibrated using **Isotonic Regression**.
- **Stage 2 (Regression - Conditional Value):** An `XGBRegressor` forecasts the conditional log-revenue (`e_revenue`) strictly for converting users, with outliers clipped at the 99th percentile to ensure stability. Evaluated via **RMSE (1.1961 vs baseline 18.02)**.
- **Ensemble Stabilization:** The final model predictions are averaged across multiple random seeds (`N_RUNS=5`). The ultimate user score is derived as the expected value: $E(\text{Revenue}) = p_{buy} \times e_{revenue}$.

## 📊 Business Insights & Strategy
- **Funnel Drop-Off A/B Testing:** Strategically targets the "Product View $\to$ Add to Cart" bottleneck using a rigorously sized 28-day experiment to optimize perceived risk, visual hierarchy, and urgency, capable of detecting a 2.60 pp uplift.
- **Segment-Level Profitability Optimization:** Simulates unit economics (CAC, assumed Conversion Rate, Break-even CR, and Margin) to calculate the **ROMI** of targeting historical groups. Prioritizes scaling "High-Value Engaged Buyers" aggressively while suppressing paid activation for low-intent visitors.
- **Predictive Customer Value (Stars vs. Low Priority):** Integrating the two-stage model outputs via K-Means separates the user base into a "Stars" segment (top 2.1% users driving outsized predicted revenue) vs. a "Low Priority" segment. 

## ⚙️ Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/groups2/data_driven_marketing.git
   cd data_driven_marketing
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Reproduce the Pipeline:**
   - **Data Preprocessing:** Run `process_data/preprocess.py` to flatten JSONs and clean data.
   - **Feature Engineering:** Execute `process_data/feature_engineering.py --mode train` to generate BTYD models, target encoders, and engineered datasets.
   - **Model Training & Inference:** Run `train_model/model.py` to execute the Two-Stage XGBoost pipeline, evaluate metrics, and generate final expected revenue outputs.
   - **Business Analysis:** Explore `Notebooks/04_analysis/` for customer segmentation strategies and A/B testing frameworks.

## 📄 Documentation
For a deep dive into the business framing, theoretical foundation of the BTYD features, and academic findings, please consult our final report: [`Group2_report.pdf`](./reports/Group2_report.pdf).