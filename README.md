# 🛒 ML Sales Prediction System

![MIT license](https://img.shields.io/badge/License-MIT-green)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![Flask](https://img.shields.io/badge/Flask-3.1-red)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6-purple)

A **Dynamic Pricing ML System** that predicts optimal prices for retail products to maximize sales revenue. Built with LightGBM, Flask, and a beautiful web UI.

![UI Preview](https://img.shields.io/badge/UI-Available-brightgreen)

---

## 🎯 What Does This Project Do?

Given a product (identified by `stockcode`), this system predicts:
- **How many units will sell** at different price points
- **What is the optimal price** to maximize total revenue

### Business Question
> *"If I price product X at $Y, how many will I sell and what's my revenue?"*

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🤖 **ML Models** | LightGBM + ElasticNet ensemble for robust predictions |
| 🌐 **REST API** | Flask-based API with CORS support |
| 🎨 **Web UI** | Beautiful, responsive prediction interface |
| 📊 **Visualizations** | Interactive charts showing price vs. sales curves |
| 🚀 **Production Ready** | Docker support for AWS Lambda/SageMaker deployment |
| ⚡ **Fast Inference** | ~1-2 seconds per prediction |

---

## 🏗️ System Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Raw Data      │────→│  Data Pipeline  │────→│  Processed Data │
│  (CSV/Excel)    │     │  (Engineering)  │     │   (Parquet)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ↓
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Predictions   │←────│   Flask API     │←────│  ML Models      │
│   (JSON/UI)     │     │   (Waitress)    │     │  (LightGBM)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/danishsyed-dev/ml-sales-prediction.git
cd ml-sales-prediction

# Install uv (if not installed)
pip install uv

# Create virtual environment and install dependencies
uv venv --python 3.12
uv sync

# Setup environment variables
cp .env.sample .env
# Edit .env with your settings (for local development, defaults work fine)
```

### Running the Application

```bash
# Start the API server
uv run app.py
```

The application will be available at:
- **API**: http://localhost:5002/
- **Web UI**: http://localhost:5002/ui

---

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check / Home |
| `/ui` | GET | **Web Interface** - Interactive prediction UI |
| `/v1/predict-price/{stockcode}` | GET | **Prediction API** - Get price predictions |
| `/ping` | GET | Health check (SageMaker) |

### Prediction API Example

```bash
# Basic prediction
curl http://localhost:5002/v1/predict-price/85123A

# With parameters
curl "http://localhost:5002/v1/predict-price/85123A?unitprice_min=5&unitprice_max=25&num_price_bins=100"
```

### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `unitprice_min` | float | Auto | Minimum price to test |
| `unitprice_max` | float | Auto | Maximum price to test |
| `num_price_bins` | int | 100 | Number of price points to evaluate |

### Response Format

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
  }
]
```

---

## 🎨 Web UI

Access the beautiful web interface at **http://localhost:5002/ui**

### Features:
- 📝 **Input Form**: Enter stock code and price range
- 📊 **Interactive Chart**: Visualize price vs. predicted sales
- 📈 **Summary Cards**: View optimal price and max revenue
- 📋 **Results Table**: Top 10 predictions sorted by sales

---

## 📁 Project Structure

```
ml-sales-prediction/
│
├── 📄 app.py                    # Flask API server (entry point)
├── 📄 pyproject.toml            # Project configuration
├── 📄 requirements.txt          # Dependencies
│
├── 📂 src/                      # Source code
│   ├── main.py                  # Full training script
│   ├── main_fast.py             # Fast training (LightGBM + EN)
│   ├── data_handling/           # Data processing pipeline
│   └── model/                   # ML model implementations
│       ├── sklearn_model/       # LightGBM, ElasticNet
│       ├── torch_model/         # PyTorch neural network
│       └── keras_model/         # TensorFlow models
│
├── 📂 templates/                # HTML templates
│   └── ui.html                  # Web UI
│
├── 📂 static/                   # Static assets
│   ├── ui.css                   # Styles
│   └── ui.js                    # JavaScript
│
├── 📂 data/                     # Data files (gitignored)
├── 📂 models/                   # Trained models (gitignored)
├── 📂 notebooks/                # Jupyter experiments
│
└── 📂 Dockerfiles               # Deployment configs
```

---

## 🤖 Machine Learning Models

| Model | Type | Performance (R²) | Use |
|-------|------|-----------------|-----|
| **LightGBM** | Gradient Boosting | 0.4717 | Primary |
| **ElasticNet** | Linear Regression | 0.2649 | Backup |

### Dataset
- **Source**: [UCI Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/online+retail)
- **Records**: 541,909 transactions
- **Period**: 2010-2011

---

## 🔧 Development Commands

| Task | Command |
|------|---------|
| **Run API** | `uv run app.py` |
| **Train Models (Fast)** | `uv run src/main_fast.py` |
| **Train All Models** | `uv run src/main.py` |
| **Process Data** | `uv run src/data_handling/main.py` |
| **Add Package** | `uv add <package>` |
| **Run Pre-commit** | `uv run pre-commit run --all-files` |

---

## 🐳 Docker Deployment

```bash
# Build Docker image
docker build -t ml-sales-prediction -f Dockerfile.lambda .

# Run container
docker run -p 5002:5002 -e ENV=local ml-sales-prediction
```

### Deployment Options
- **AWS Lambda** - `Dockerfile.lambda`
- **AWS SageMaker** - `Dockerfile.sagemaker`

---

## 📊 Sample Stock Codes

| StockCode | Description |
|-----------|-------------|
| `85123A` | WHITE HANGING HEART T-LIGHT HOLDER |
| `22423` | REGENCY CAKESTAND 3 TIER |
| `85099B` | JUMBO BAG RED RETROSPOT |
| `84879` | ASSORTED COLOUR BIRD ORNAMENT |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.12 |
| **API Framework** | Flask + Waitress |
| **ML Libraries** | LightGBM, Scikit-learn, PyTorch |
| **Data Processing** | Pandas, NumPy |
| **Package Manager** | uv |
| **Caching** | Redis (optional) |
| **Cloud** | AWS (S3, Lambda, ECR) |

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Online Retail Dataset
- LightGBM team for the excellent gradient boosting library

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/danishsyed-dev">Danish Syed</a>
</p>
