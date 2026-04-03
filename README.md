# Stock Price Prediction (1,000,000 Rows)

A complete, highly-optimized end-to-end Machine Learning project to generate and predict stock prices using a massive dataset of 1,000,000 records. Built in Python leveraging pandas, numPy, scikit-learn, and matplotlib.

## Overview

Working with large datasets natively presents unique challenges out of traditional loop-based constraints. This project demonstrates best practices by implementing:
- **Vectorized Generation:** Creating a full million rows dynamically and writing to disk in seconds. 
- **Efficient Processing:** Performing instantaneous forward filling over massive arrays.
- **Fast Training:** Applying Multiple Linear Regression effectively across `Open`, `High`, `Low`, and `Volume` parameters to forecast `Close` targets seamlessly.

## 📊 Dataset
- Generated using Python (pandas + numpy)
- Contains 1,000,000+ rows
- Simulates realistic stock price movement using previous-day dependency and random fluctuations

> **📦 Note on Dataset Size & GitHub Uploads**
> The generated dataset `stock_data.csv` is approximately 54 MB. To prevent GitHub upload limitations and keep this repository lightning-fast to clone, the CSV is intentionally ignored via `.gitignore` and not uploaded. 
> **You do not need to download the CSV.** Just run the `main.py` script once; it will seamlessly reconstruct the entire 1,000,000-row file locally on your system in under 5 seconds! 
> *(Want larger/smaller data? Simply open `main.py` and change `n_rows=1000000` inside the `main()` function to any number you intuitively prefer!)*

## 🤖 Algorithm Used
- **Model:** Multiple Linear Regression (via `scikit-learn`)
- **Reasoning:** It's an exceptionally fast and highly interpretable baseline approach. Since `Close` is heavily correlated dimensionally with `(Open, High, Low)`, it is perfectly adequate for tracking continuous variance across simulated datasets without the vast overhead cost or overfitting risks introduced by Deep Learning sequence models (LSTMs).

## Features Included
1. **Dynamic Dataset Creation**: Programmatically creates `stock_data.csv` replicating geometric variations similar to actual random walk occurrences within markets. 
2. **Missing Value Processing**: Introduces explicit `np.nan` values and cleans them securely without data bleeding.
3. **Machine Learning Predictor**: Utilizes a baseline Linear Regression module trained rigorously across 80% split datasets evaluating on the final 20% chunks.
4. **Data Visualization**: Generates a tailored graphical trend output capturing granular sub-plots (avoids heavy overlapping graph clusters) stored directly to `prediction_plot.png`.

## Setup & Run Instructions

### Prerequisites
Make sure `python` corresponds to Python 3.8+ on your system. 

```bash
# Install required libraries
pip install numpy pandas scikit-learn matplotlib
```

### Execution
Run the full predictive sequence through `main.py`.

```bash
python main.py
```

### Expected Output Summary
1. `stock_data.csv`: A fresh ~60MB comma-separated dataset with 1,000,000 real-like entries.
2. `prediction_plot.png`: A plotted graph of True versus Predicted outcomes mapping the latest timeframe sequence.
3. **Console Metrics**: Immediate real-time output including MAE (Mean Absolute Error) and R2 Scoring to identify confidence correlations.

## Project Structure
```text
stock_price_prediction/
├── main.py              # Main ML implementation and Generation scripts
├── README.md            # Active documentation footprint (this file)
├── stock_data.csv       # Generating automatically (Ignored from version control ideally)
└── prediction_plot.png  # Output graph visually representing tests
```
