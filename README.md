# Recommendation_System
Recommendation System for E-commerce -
Overview

This project builds and evaluates an e-commerce recommendation system from clickstream data (views, add-to-cart, purchases). It follows CRISP-DM and balances performance with interpretability. Models are trained on implicit feedback and evaluated with Recall@K, MAP@K, and nDCG@K on a chronological holdout.

Project goal: Recommend items users are most likely to purchase next, and generate actionable insights that inform merchandising and timing strategies.

**Business Questions**

1. Which items are most frequently viewed and/or purchased?

2. What time of day/day of week has the highest engagement?

3. Which user groups show similar interests or purchase patterns?

4. Which product categories have the highest conversion (views → purchases)?

5. Do users interact more with items that have richer metadata?

6. What proportion of items receive little or no attention?

7. How do personalized recommendations compare with popularity-based ones in offline evaluation?

**Data**

Interactions: view, addtocart, transaction with timestamps and user/item IDs.

Item properties: available, categoryid, parentid (plus any extra metadata if present).

Time features: raw timestamp and derived (event_time, hour, weekday name).

All modeling uses implicit feedback. Event weights reflect intent strength (e.g., view < add-to-cart < purchase).


**Methodology (CRISP-DM)**

1) Business & Data Understanding

Defined objectives, success metrics, and constraints.

Assessed dataset size, sparsity, and event mix (views dominate; purchases are sparse).

2) Cleaning & Preprocessing

Standardized schemas, fixed mixed dtypes, parsed timestamps.

Normalized item property values; merged properties into interactions.

Built time features (hour, weekday) and per-event weights.

3) Exploratory Data Analysis (EDA)

Top items: most viewed/purchased products.

Temporal patterns: engagement by hour and weekday.

Category conversion: smoothed view→purchase conversion by category.

Metadata impact: items with richer metadata attract more interactions per item.

Catalog coverage: share of items with low/no attention (cold-start risk).

4) Modeling

- Candidate generators

- Popularity@K (baseline).

- Item-Item Cosine on user→item matrix with:

- RAW implicit weights

- TF-IDF re-weighting

- BM25 re-weighting (best simple baseline)

- SVD (MF) on BM25 (exploratory; limited gains here due to sparsity).

- Co-visitation + TF-IDF + LSA (fast, interpretable item-item similarity).

- Reranking (diagnostic)

- Lightweight metadata-aware reranker over BM25 candidates using:

- Item popularity, freshness (recency), category conversion rate, user↔category affinity, availability.

Showed meaningful gains on a diagnostic slice; full tuning recommended.

5) Evaluation

Chronological split: Train = earlier period; Test = future period.

Ground truth: user’s transactions in the holdout window.

Metrics: Recall@10 (primary), MAP@10, nDCG@10.

Scalability: Chunked evaluation over users to control runtime/memory.


**Takeaways**

Personalized models decisively beat popularity.

Item-Cosine with BM25 is a strong, fast, and interpretable baseline (≈46× Popularity on Recall@10).

Metadata-aware reranking improved Recall@10 on a slice; full hyperparameter tuning is a promising next step.

**Repository Structure**
.
├─ data/

│  ├─ raw/                      # original CSVs

│  ├─ interim/                  # cleaned properties & merged interactions

│  └─ processed/                # train/test splits, matrices, cached sims

├─ notebooks/

│  ├─ Business & Data Understanding.ipynb

│  ├─ Cleaning & Preprocessing.ipynb

│  ├─ EDA.ipynb

│  └─ Modeling & Evaluation_RecommendationSys.ipynb

├─ src/

│  ├─ preprocess.py             # cleaning & merge utilities

│  ├─ features.py               # time features, weights, encoders

│  ├─ models.py                 # candidate generation, reranking, similarity

│  ├─ weighting.py              # TF-IDF, BM25 matrix weighting

│  └─ eval.py                   # split, metrics, chunked evaluation

├─ reports/

│  ├─ figures/                  # exported plots

│  └─ tables/                   # evaluation summaries

└─ README.md

Environment & Setup
# Python 3.11+ recommended
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS/Linux

pip install -U pip
pip install numpy pandas scipy scikit-learn matplotlib tqdm
# (Optional) seaborn. We avoid packages that require C/C++ toolchains.

**How to Run**

Cleaning & Merge – run Cleaning & Preprocessing.ipynb → writes cleaned/interim data.

EDA – run EDA.ipynb → answers business questions 1–6 with visuals.

Modeling & Evaluation – run Modeling & Evaluation_RecommendationSys.ipynb:

Build user→item matrices (RAW/TF-IDF/BM25).

Train/evaluate candidate generators.

(Optional) Metadata-aware reranking over BM25 candidates.

Save evaluation tables/plots in reports/.

If evaluation is slow, reduce CHUNK_USERS or do a small sampled run first.

**Visual Highlights**

Top items: a few SKUs dominate exposure and sales.

Temporal patterns: clear hourly/weekday peaks for scheduling campaigns.

Category conversion: select categories convert far better → prioritize them.

Metadata impact: richer metadata correlates with more views/purchases per item.

Coverage: a large share of items are low/no-attention → address cold-start/long-tail.

**Recommendations & Roadmap**

Ship now: Item-Cosine + BM25 @10 (robust + interpretable).

Candidate coverage: add time-decayed co-visitation and category-aware boosts.

Learned reranker: XGBoost/LightGBM on features (base score, popularity, freshness, category conversion, user–category affinity, availability, price/brand). Optimize nDCG@10.

Diversity/coverage: MMR or category quotas to lift long-tail exposure.

Cold-start:

New users: trending-by-category + time-of-day context.

New items: content similarity + category priors.

Online A/B test: compare personalized vs popularity during peak windows found in EDA.

**Limitations**

Purchases are extremely sparse → low absolute Recall@10 is typical; relative gains are the signal.

Some advanced libraries require compilers on Windows; this pipeline avoids them by default.

Reranker results shown on a slice; full tuning likely yields further gains.

