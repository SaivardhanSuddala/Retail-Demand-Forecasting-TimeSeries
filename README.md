# Demand Prediction using Machine Learning

## Overview
This project predicts product sales using historical store-item data.  
It uses time-based feature engineering and regression models to forecast demand.

---

## Problem
Retail stores need accurate demand forecasting to avoid:
- Overstocking
- Stockouts
- Revenue loss

---

## Solution
Built a machine learning pipeline that:
- Processes historical sales data
- Extracts time-based and lag features
- Trains regression models to predict future sales

---

## Dataset
- Input: `train.csv`
- Features:
  - store
  - item
  - date
  - sales

---

## Feature Engineering
- Date features:
  - Month
  - Day
  - Weekday
- Lag features:
  - Previous day sales (`lag1`)
  - Previous week sales (`lag7`)
- Rolling mean:
  - 7-day average (`rolling7`)

---

## Data Processing
- Sorted by store, item, date
- Train-test split (80-20)
- Standard scaling applied using Pipeline

---

## Models Used

### 1. Linear Regression
- Baseline model

### 2. Random Forest Regressor
- Better performance
- Parameters:
  - n_estimators = 10
  - max_depth = 10
  - min_samples_leaf = 4

---

## Evaluation Metrics
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

---

## Results
Random Forest performed better than Linear Regression in terms of RMSE and MAE.

---

## Output
- Trained model saved as:
