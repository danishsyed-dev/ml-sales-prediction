"""
Fast Training Script - Optimized for ~30-45 minutes
Trains only the essential models: LightGBM + Elastic Net
Skips SVR (too slow) and reduces DFN complexity
"""
import os
import warnings
import pickle
import joblib
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from dotenv import load_dotenv

import src.data_handling as data_handling
from src._utils import s3_upload, main_logger

warnings.filterwarnings("ignore", category=RuntimeWarning)

# paths
PRODUCTION_MODEL_FOLDER_PATH = 'models/production'
GBM_FILE_PATH = os.path.join(PRODUCTION_MODEL_FOLDER_PATH, 'gbm_best.pth')
EN_FILE_PATH = os.path.join(PRODUCTION_MODEL_FOLDER_PATH, 'en_best.pth')
PREPROCESSOR_PATH = 'preprocessors/column_transformer.pkl'


def train_elastic_net_fast(X_train, y_train, X_val, y_val):
    """Train Elastic Net with reduced hyperparameter search"""
    main_logger.info("=== Training Elastic Net (Fast Mode) ===")
    
    # Simplified parameter grid for faster training
    param_grid = {
        'alpha': [0.01, 0.1, 1.0],
        'l1_ratio': [0.5, 0.9],
        'max_iter': [10000],
        'tol': [1e-4],
    }
    
    model = ElasticNet(random_state=42)
    grid_search = GridSearchCV(
        model, param_grid, 
        cv=3, 
        scoring='neg_mean_squared_error',
        n_jobs=-1,  # Use all CPU cores
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    # Evaluate
    train_score = best_model.score(X_train, y_train)
    val_score = best_model.score(X_val, y_val)
    main_logger.info(f"Elastic Net - Train R²: {train_score:.4f}, Val R²: {val_score:.4f}")
    main_logger.info(f"Best params: {grid_search.best_params_}")
    
    return best_model, grid_search.best_params_


def train_lightgbm_fast(X_train, y_train, X_val, y_val):
    """Train LightGBM with optimized settings for speed"""
    main_logger.info("=== Training LightGBM (Fast Mode) ===")
    
    # Fast but effective parameters
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'max_depth': 8,
        'learning_rate': 0.05,
        'n_estimators': 500,  # Reduced from 2000
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'n_jobs': -1,  # Use all CPU cores
        'verbose': -1,
    }
    
    model = lgb.LGBMRegressor(**params)
    
    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100)
        ]
    )
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)
    main_logger.info(f"LightGBM - Train R²: {train_score:.4f}, Val R²: {val_score:.4f}")
    
    return model, params


if __name__ == '__main__':
    load_dotenv(override=True)
    
    # Optimize for single-threaded operation where beneficial
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    
    os.makedirs(PRODUCTION_MODEL_FOLDER_PATH, exist_ok=True)
    
    main_logger.info("=" * 60)
    main_logger.info("FAST TRAINING MODE - Estimated time: 30-45 minutes")
    main_logger.info("=" * 60)
    
    # Load preprocessed data (skip reprocessing if exists)
    main_logger.info("Loading data...")
    
    if os.path.exists('data/processed_df.parquet'):
        main_logger.info("Using existing processed data...")
        import pandas as pd
        from sklearn.model_selection import train_test_split
        
        df = pd.read_parquet('data/processed_df.parquet')
        target_col = 'quantity'
        
        y = df[target_col]
        X = df.drop(target_col, axis='columns')
        
        # Split data
        test_size, random_state = 50000, 42
        X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=test_size, random_state=random_state)
        
        # Load or create preprocessor
        try:
            if os.path.exists(PREPROCESSOR_PATH) and os.path.getsize(PREPROCESSOR_PATH) > 100:
                preprocessor = joblib.load(PREPROCESSOR_PATH)
                main_logger.info("Loaded existing preprocessor")
            else:
                preprocessor = None
                main_logger.info("No valid preprocessor found, using raw features")
        except Exception as e:
            preprocessor = None
            main_logger.info(f"Could not load preprocessor ({e}), using raw features")
        
        # Handle categorical columns - drop them for fast training (use only numeric)
        main_logger.info("Selecting only numeric features for fast training...")
        numeric_cols = X_train.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        main_logger.info(f"Using {len(numeric_cols)} numeric columns: {numeric_cols[:5]}...")
        
        X_train = X_train[numeric_cols]
        X_val = X_val[numeric_cols]
        X_test = X_test[numeric_cols]
        
        # Convert to numpy for sklearn
        X_train = X_train.values if hasattr(X_train, 'values') else X_train
        X_val = X_val.values if hasattr(X_val, 'values') else X_val
        X_test = X_test.values if hasattr(X_test, 'values') else X_test
        y_train = y_train.values if hasattr(y_train, 'values') else y_train
        y_val = y_val.values if hasattr(y_val, 'values') else y_val
        y_test = y_test.values if hasattr(y_test, 'values') else y_test
        
    else:
        main_logger.info("Processing data from scratch...")
        X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = data_handling.main_script()
        
        if preprocessor is not None:
            joblib.dump(preprocessor, PREPROCESSOR_PATH)
            s3_upload(PREPROCESSOR_PATH)
    
    main_logger.info(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Train LightGBM (fastest and often best)
    main_logger.info("\n" + "=" * 60)
    main_logger.info("STEP 1/2: Training LightGBM (~15-20 min)")
    main_logger.info("=" * 60)
    
    best_gbm, best_params_gbm = train_lightgbm_fast(X_train, y_train, X_val, y_val)
    
    with open(GBM_FILE_PATH, 'wb') as f:
        pickle.dump({'best_model': best_gbm, 'best_hparams': best_params_gbm}, f)
    s3_upload(file_path=GBM_FILE_PATH)
    main_logger.info(f"✓ LightGBM saved to {GBM_FILE_PATH}")
    
    # Train Elastic Net (fast linear model)
    main_logger.info("\n" + "=" * 60)
    main_logger.info("STEP 2/2: Training Elastic Net (~10-15 min)")
    main_logger.info("=" * 60)
    
    best_en, best_params_en = train_elastic_net_fast(X_train, y_train, X_val, y_val)
    
    with open(EN_FILE_PATH, 'wb') as f:
        pickle.dump({'best_model': best_en, 'best_hparams': best_params_en}, f)
    s3_upload(file_path=EN_FILE_PATH)
    main_logger.info(f"✓ Elastic Net saved to {EN_FILE_PATH}")
    
    # Summary
    main_logger.info("\n" + "=" * 60)
    main_logger.info("🎉 TRAINING COMPLETE!")
    main_logger.info("=" * 60)
    main_logger.info(f"Models saved to: {PRODUCTION_MODEL_FOLDER_PATH}/")
    main_logger.info("- gbm_best.pth (LightGBM)")
    main_logger.info("- en_best.pth (Elastic Net)")
    main_logger.info("\nYou can now run the API with: uv run app.py")
