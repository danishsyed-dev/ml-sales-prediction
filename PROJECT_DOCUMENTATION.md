# ML Sales Prediction - Complete Project Documentation

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Data Pipeline](#data-pipeline)
4. [Machine Learning Models](#machine-learning-models)
5. [API Endpoints](#api-endpoints)
6. [Output Format](#output-format)
7. [How to Run](#how-to-run)
8. [File Descriptions](#file-descriptions)

---

## 🎯 Project Overview

### What is this project?

This is a **Dynamic Pricing ML System** that predicts optimal prices for retail products to maximize sales revenue.

### Problem Statement
Given a product (identified by `stockcode`), predict:
- What quantity will sell at different price points?
- What is the optimal price to maximize total sales?

### Solution
- Uses **UCI Online Retail Dataset** (541,909 transactions)
- Trains **LightGBM** and **Elastic Net** regression models
- Serves predictions via **Flask REST API**
- Predicts quantity sold at 100 different price points
- Identifies the optimal price for maximum revenue

### Tech Stack
| Component | Technology |
|-----------|------------|
| Language | Python 3.12 |
| ML Framework | LightGBM, Scikit-learn, PyTorch |
| API | Flask + Waitress |
| Data | Pandas, Parquet |
| Package Manager | uv |
| Deployment | Docker, AWS Lambda (optional) |

---

## 📁 Project Structure

```
ml-sales-prediction/
│
├── 📄 app.py                    # Main Flask API application
├── 📄 pyproject.toml            # Python project configuration
├── 📄 requirements.txt          # Dependencies list
├── 📄 .env                      # Environment variables (local)
├── 📄 .env.sample               # Environment template
│
├── 📂 src/                      # Source code
│   ├── 📄 main.py               # Full training script (4 models)
│   ├── 📄 main_fast.py          # Fast training (LightGBM + ElasticNet)
│   ├── 📄 main_stockcode.py     # Train for specific product
│   │
│   ├── 📂 data_handling/        # Data processing
│   │   ├── 📄 main.py           # Main data pipeline
│   │   └── 📂 scripts/          # Processing functions
│   │       ├── loading.py       # Load raw data
│   │       ├── imputing.py      # Handle missing values
│   │       ├── engineering.py   # Feature engineering
│   │       └── transforming.py  # Data transformations
│   │
│   ├── 📂 model/                # ML model implementations
│   │   ├── 📂 sklearn_model/    # Scikit-learn models
│   │   ├── 📂 torch_model/      # PyTorch neural network
│   │   └── 📂 keras_model/      # TensorFlow/Keras models
│   │
│   └── 📂 _utils/               # Utility functions
│       ├── log.py               # Logging configuration
│       └── s3.py                # AWS S3 operations
│
├── 📂 templates/                # HTML templates
│   └── ui.html                  # Web UI for predictions
│
├── 📂 static/                   # Static assets
│   ├── ui.css                   # Styles (IBM Plex Serif, Space Grotesk)
│   └── ui.js                    # JavaScript (charts, forms, validation)
│
├── 📂 data/                     # Data files (gitignored)
│   ├── 📂 raw/                  # Raw data (online_retail.csv)
│   ├── 📂 processed/            # Processed data
│   ├── original_df.parquet      # Original dataframe
│   ├── processed_df.parquet     # Engineered features
│   └── x_test_df.parquet        # Test dataset
│
├── 📂 models/                   # Trained models (gitignored)
│   └── 📂 production/
│       ├── gbm_best.pth         # LightGBM model (1.5 MB)
│       └── en_best.pth          # Elastic Net model (687 bytes)
│
├── 📂 preprocessors/            # Data preprocessors
│   └── column_transformer.pkl   # Feature transformer
│
├── 📂 notebooks/                # Jupyter notebooks
│   ├── exp_v1.ipynb             # Experiment version 1
│   ├── exp_v2.ipynb             # Experiment version 2
│   └── exp_v3.ipynb             # Experiment version 3
│
├── 📂 tests/                    # Unit tests
│
└── 📄 Dockerfile.lambda         # Docker for AWS Lambda
└── 📄 Dockerfile.sagemaker      # Docker for SageMaker
```

---

## 📊 Data Pipeline

### Dataset: UCI Online Retail
- **Source**: https://archive.ics.uci.edu/ml/datasets/online+retail
- **Records**: 541,909 transactions
- **Time Period**: 2010-2011
- **Country**: Mostly UK

### Raw Data Columns
| Column | Type | Description |
|--------|------|-------------|
| InvoiceNo | String | Invoice number (unique per transaction) |
| StockCode | String | Product code (e.g., "85123A") |
| Description | String | Product description |
| Quantity | Integer | Number of units purchased |
| InvoiceDate | DateTime | Date/time of transaction |
| UnitPrice | Float | Price per unit |
| CustomerID | String | Customer identifier |
| Country | String | Customer's country |

### Feature Engineering
The data pipeline creates these features:

| Feature | Description |
|---------|-------------|
| `invoicedate` | Timestamp converted to numeric |
| `unitprice` | Original price |
| `year` | Year extracted from date |
| `year_month` | Year-month combination |
| `day_of_week` | Day of week (0-6) |
| `product_avg_quantity_last_month` | Rolling average quantity |
| `product_max_price_all_time` | Historical max price |
| `unitprice_vs_max` | Price ratio to max |
| `unitprice_to_avg` | Price ratio to average |
| `unitprice_squared` | Price squared |
| `unitprice_log` | Log of price |
| `is_registered` | If customer has ID |

### Data Splits
| Dataset | Size | Purpose |
|---------|------|---------|
| Training | 431,138 rows | Model training |
| Validation | 50,000 rows | Hyperparameter tuning |
| Test | 50,000 rows | Final evaluation |

---

## 🤖 Machine Learning Models

### Models Available

#### 1. LightGBM Regressor (Primary)
- **Type**: Gradient Boosting
- **Performance**: R² = 0.4717
- **File**: `models/production/gbm_best.pth`
- **Size**: 1.5 MB
- **Training Time**: ~15 minutes

Parameters:
```python
{
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'max_depth': 8,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'subsample': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1
}
```

#### 2. Elastic Net (Backup)
- **Type**: Linear Regression with L1/L2 regularization
- **Performance**: R² = 0.2649
- **File**: `models/production/en_best.pth`
- **Size**: 687 bytes
- **Training Time**: ~10 minutes

Parameters:
```python
{
    'alpha': 0.01,
    'l1_ratio': 0.5,
    'max_iter': 10000
}
```

#### 3. Deep Feedforward Network (Optional)
- **Type**: Neural Network (PyTorch)
- **Layers**: Multi-layer perceptron
- **Not trained in fast mode**

#### 4. SVR - Support Vector Regression (Optional)
- **Type**: Support Vector Machine
- **Very slow to train**
- **Not used in fast mode**

### Model Selection Logic
```
1. Try to load DFN (Neural Network) → If fails...
2. Try to load LightGBM → If fails...
3. Try to load Elastic Net → If fails...
4. Return empty prediction
```

---

## 🌐 API Endpoints

### Base URL
```
http://localhost:5002
```

### Endpoints

#### 1. Home
```
GET /
```
Response:
```html
<p>Hello, world</p><p>I am an API endpoint.</p>
```

#### 2. Web UI ⭐ (Interactive Interface)
```
GET /ui
```
Opens the beautiful web interface with:
- 📝 **Input Form**: Enter stock code and price range
- 📊 **Interactive Chart**: Visualize price vs. predicted sales
- 📈 **Summary Cards**: View optimal price and max revenue
- 📋 **Results Table**: Top 10 predictions sorted by sales

Features:
- Premium fonts (IBM Plex Serif, Space Grotesk)
- Responsive design for mobile and desktop
- Real-time form validation
- SVG chart rendering

#### 3. Health Check
```
GET /ping
```
Response: `200 OK` (empty)

#### 4. Price Prediction ⭐ (Main API Endpoint)
```
GET /v1/predict-price/{stockcode}
```

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| stockcode | string | Yes | - | Product code (e.g., "85123A") |
| unitprice_min | float | No | Auto | Minimum price to test |
| unitprice_max | float | No | Auto | Maximum price to test |
| num_price_bins | int | No | 100 | Number of price points |

**Example Request:**
```
GET /v1/predict-price/85123A?unitprice_min=5&unitprice_max=25
```

---

## 📤 Output Format

### Prediction Response (JSON Array)

Each prediction returns an array of 100 price-quantity pairs:

```json
[
  {
    "stockcode": "85123A",
    "unit_price": 2.0,
    "quantity": 349,
    "quantity_min": 155,
    "quantity_max": 1548,
    "predicted_sales": 699.0,
    "optimal_unit_price": 20.0,
    "max_predicted_sales": 6990.0
  },
  {
    "stockcode": "85123A",
    "unit_price": 2.18,
    "quantity": 349,
    "quantity_min": 155,
    "quantity_max": 1548,
    "predicted_sales": 762.55,
    "optimal_unit_price": 20.0,
    "max_predicted_sales": 6990.0
  },
  // ... 98 more entries
]
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `stockcode` | string | Product identifier |
| `unit_price` | float | Price point being tested |
| `quantity` | int | Predicted quantity sold (median) |
| `quantity_min` | int | Minimum predicted quantity |
| `quantity_max` | int | Maximum predicted quantity |
| `predicted_sales` | float | quantity × unit_price |
| `optimal_unit_price` | float | Best price for max revenue |
| `max_predicted_sales` | float | Maximum possible revenue |

### Interpretation

The API answers: **"If I price product X at $Y, how many will I sell?"**

Example output interpretation:
- At $2.00 → Sell 349 units → Revenue: $699
- At $20.00 → Sell 349 units → Revenue: $6,990 (OPTIMAL)
- Recommendation: Price at **$20.00** for maximum revenue

---

## 🚀 How to Run

### Prerequisites
- Python 3.12
- uv (package manager)
- 8GB+ RAM

### Step 1: Clone & Setup
```bash
git clone https://github.com/danishsyed-dev/ml-sales-prediction.git
cd ml-sales-prediction

# Create virtual environment
uv venv --python 3.12
```

### Step 2: Install Dependencies
```bash
uv sync
```

### Step 3: Setup Environment
```bash
# Create .env file
cp .env.sample .env
```

Edit `.env`:
```env
ENV=local
AWS_REGION_NAME=
S3_BUCKET_NAME=
ORIGINAL_DF_PATH=data/original_df.parquet
PROCESSED_DF_PATH=data/processed_df.parquet
X_TEST=data/x_test_df.parquet
```

### Step 4: Download Data
1. Download UCI Online Retail Dataset
2. Extract `Online Retail.xlsx` to `data/raw/`
3. Convert to CSV:
```bash
uv run python -c "import pandas as pd; pd.read_excel('data/raw/Online Retail.xlsx').to_csv('data/raw/online_retail.csv', index=False)"
```

### Step 5: Train Models (Fast Mode)
```bash
uv run src/main_fast.py
```
Output:
- `models/production/gbm_best.pth`
- `models/production/en_best.pth`

### Step 6: Run API
```bash
uv run app.py
```

### Step 7: Test
Open browser: http://localhost:5002/v1/predict-price/85123A

---

## 📄 File Descriptions

### Core Files

| File | Lines | Description |
|------|-------|-------------|
| `app.py` | 560 | Flask API server with all endpoints |
| `src/main.py` | 161 | Full training script (4 models) |
| `src/main_fast.py` | 210 | Fast training (LightGBM + EN only) |
| `src/data_handling/main.py` | 133 | Data processing pipeline |

### Configuration Files

| File | Description |
|------|-------------|
| `pyproject.toml` | Project metadata & dependencies |
| `.env` | Local environment variables |
| `.env.sample` | Template for .env |
| `.python-version` | Python version lock (3.12) |
| `uv.lock` | Dependency lock file |

### Docker Files

| File | Description |
|------|-------------|
| `Dockerfile.lambda` | AWS Lambda deployment |
| `Dockerfile.sagemaker` | AWS SageMaker deployment |

---

## 📈 Performance Metrics

### Model Performance

| Model | Train R² | Validation R² | RMSE |
|-------|----------|---------------|------|
| LightGBM | 0.4888 | 0.4717 | 0.716 |
| Elastic Net | 0.2623 | 0.2649 | ~1.0 |

### API Performance

| Metric | Value |
|--------|-------|
| Cold Start | ~5 seconds |
| Prediction Time | ~1-2 seconds |
| Response Size | ~15 KB |

---

## 🔧 Troubleshooting

### Common Issues

1. **"Error loading preprocessor"**
   - The preprocessor file may be corrupted
   - The API will use raw features instead (still works)

2. **"Local file not found"**
   - Missing data files in `data/` folder
   - Run the training script first

3. **CORS Errors**
   - Fixed in latest version
   - Origins must not contain `None` values

4. **Memory Errors**
   - Ensure 8GB+ RAM available
   - Close other applications during training

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Raw Data      │────▶│  Data Pipeline  │────▶│  Processed Data │
│  (CSV/Excel)    │     │  (Engineering)  │     │   (Parquet)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Predictions   │◀────│   Flask API     │◀────│  ML Models      │
│   (JSON)        │     │   (Waitress)    │     │  (LightGBM)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## 📝 Sample Stock Codes to Test

| StockCode | Description |
|-----------|-------------|
| 85123A | WHITE HANGING HEART T-LIGHT HOLDER |
| 22423 | REGENCY CAKESTAND 3 TIER |
| 85099B | JUMBO BAG RED RETROSPOT |
| 84879 | ASSORTED COLOUR BIRD ORNAMENT |
| 22720 | SET OF 3 CAKE TINS PANTRY DESIGN |

Try: `http://localhost:5002/v1/predict-price/22423`

---

## 📜 License

MIT License

---

*Documentation generated for ml-sales-prediction project*
*Last updated: February 5, 2026*

