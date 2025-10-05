# Project:  Tesla Stock Price Forecasting Project

## Project Overview
This project implements time-series forecasting for Tesla stock prices using two different approaches:
1. **ARIMA** - Traditional statistical model
2. **LSTM** - Deep learning neural network

## Dataset
- **Stock**: Tesla (TSLA)
- **Period**: 2010-2020
- **Data Source**: Kaggle (https://www.kaggle.com/datasets/timoboz/tesla-stock-data-from-2010-to-2020 )
- **Features**: Open, High, Low, Close, Adj Close, Volume (focusing on Close prices)

## Start Guide

### Installation
```bash
# Install required packages
pip install -r requirements.txt
```

### Running the Jupyter Notebook
```python
# Open and run the main notebook
jupyter notebook Time_Series_Forecasting.ipynb
```


## 📁 Project Structure
```
├── requirements.txt                # Python dependencies
├── Notebook/
    ├──Time_Series_Forecasting.ipynb   # Main analysis notebook
├── README.md                       # Project documentation
└── outputs/
    ├── tesla_eda.png              # Exploratory data analysis plots
    ├── lstm_training_history.png  # LSTM training progress
    ├── prophet_components.png     # Prophet decomposition
    ├── model_comparison.png       # Performance comparison
    ├── all_models_predictions.png # All predictions visualized
    └── forecasting_report.txt     # Detailed analysis report
```

## Model Descriptions

### 1. ARIMA (AutoRegressive Integrated Moving Average)
A traditional statistical method for time series forecasting.

**How it works:**
- **AR (AutoRegressive)**: Uses past values to predict future values
- **I (Integrated)**: Makes data stationary by differencing
- **MA (Moving Average)**: Uses past forecast errors

**Parameters (p, d, q):**
- `p`: Number of lag observations
- `d`: Degree of differencing (to make data stationary)
- `q`: Size of moving average window

**Example:** ARIMA(5,1,0) means:
- Use 5 past values
- Difference the data once
- No moving average component

**Pros:**
- Fast training and prediction
- Interpretable results
- Works well with linear trends

**Cons:**
- Assumes linear relationships
- Requires stationary data
- May struggle with sudden changes

### 2. LSTM (Long Short-Term Memory)
A type of recurrent neural network (deep learning).

**How it works:**
- Has "memory cells" that can remember long-term patterns
- Uses gates to decide what information to keep or forget
- Processes sequences of data (60 days → 1 day prediction)

**Architecture:**
```
Input (60 days) → LSTM Layer (50 units) → Dropout (20%) 
→ LSTM Layer (50 units) → Dropout (20%) 
→ Dense Layer (25 units) → Output (1 day)
```

**Key Concepts:**
- **Sequence Length**: Uses 60 days of history to predict next day
- **Normalization**: Scales prices to 0-1 range for better training
- **Dropout**: Prevents overfitting by randomly dropping connections

**Pros:**
- Captures non-linear patterns
- Handles long-term dependencies
- Powerful for complex data

**Cons:**
- Requires more data
- Longer training time
- Needs GPU for faster training
- Can overfit easily

## Performance Metrics Explained

### RMSE (Root Mean Squared Error)
- Measures average prediction error in dollars
- **Lower is better**
- Penalizes large errors more heavily
- Example: RMSE of $5.50 means average error is $5.50

### MAE (Mean Absolute Error)
- Average absolute difference between predicted and actual
- **Lower is better**
- More interpretable than RMSE
- Example: MAE of $4.20 means on average, predictions are $4.20 off

### MAPE (Mean Absolute Percentage Error)
- Shows error as a percentage
- **Lower is better**
- Good for comparing across different price ranges
- Example: MAPE of 3.5% means predictions are off by 3.5% on average

## Results Summary

Based on the analysis, here's what each model achieved:

| Model   | RMSE    | MAE     | MAPE   | Best For                          |
|---------|---------|---------|--------|-----------------------------------|
| ARIMA   | $12.43  | $7.68   | 2.47%  | Quick forecasts, linear trends    |
| LSTM    | $21.14  | $14.26  | 4.51%  | Complex patterns, large datasets  |
<img width="4467" height="1468" alt="model_comparison" src="https://github.com/user-attachments/assets/51b8555d-3db2-4ab9-98ac-54725181d07a" />

### Forecasting results:


<img width="4771" height="2369" alt="all_models_predictions" src="https://github.com/user-attachments/assets/6e2ef3f2-3e3e-49b3-bc13-27baf9c243d9" />


## Key Code Sections Explained

### Rolling Window Evaluation
```python
# This tests the model on new data iteratively
for t in range(len(test_data)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit()
    prediction = model_fit.forecast()
    # Add actual value to history for next prediction
    history.append(actual_value)
```
**Why?** This simulates real-world forecasting where you get new data daily.

### Data Normalization for LSTM
```python
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)
```
**Why?** Neural networks work better with normalized data (0-1 range).

### Sequence Creation for LSTM
```python
# Use past 60 days to predict day 61
X = [day1-60], [day2-61], [day3-62], ...
y = [day61], [day62], [day63], ...
```
**Why?** LSTM needs sequences to learn temporal patterns.



## Learning Resources

### Understanding ARIMA:
- **Stationarity**: Data should have constant mean and variance
- **ACF/PACF Plots**: Help determine p and q parameters
- **Differencing**: Makes non-stationary data stationary

### Understanding LSTM:
- **Sequences**: LSTM processes sequences of data
- **Gates**: Control information flow (forget, input, output gates)
- **Training**: Requires many epochs and backpropagation


## Important Notes

1. **Stock Market Disclaimer**: 
   - These models are for educational purposes only
   - Stock prices are influenced by many unpredictable factors
   - Never use predictions as sole investment advice

2. **Data Quality**:
   - More data = better models
   - Tesla's stock has high volatility
   - Consider using multiple years of data

3. **Model Selection**:
   - No single model is "best" for all situations
   - Consider ensemble methods (combining models)
   - Validate on recent data

4. **Hyperparameter Tuning**:
   - ARIMA: Try different (p,d,q) combinations
   - LSTM: Adjust layers, units, sequence length
   - Prophet: Modify seasonality and changepoint settings

##  Next Steps & Improvements

1. **Add More Features**:
   - Include volume, technical indicators
   - Add sentiment analysis from news
   - Include market indices (S&P 500)

2. **Try Ensemble Methods**:
   - Combine predictions from all models
   - Weighted average based on performance

3. **Real-Time Updates**:
   - Fetch latest data automatically
   - Retrain models periodically

4. **More Stocks**:
   - Extend to multiple stocks
   - Portfolio optimization

## Contributing
Feel free to fork this project and submit pull requests!

