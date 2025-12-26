# Financial Market Analytics: Apple vs S&P 500

## Overview
This project performs a comparative financial analysis between Apple Inc. (AAPL) and the S&P 500 index using historical market data.  
It evaluates return, volatility, and risk-adjusted performance to demonstrate practical financial analytics skills.

## Key Objectives
- Analyze historical price data
- Compute daily and cumulative returns
- Measure volatility and Sharpe Ratio
- Compare individual stock performance against a market benchmark

## Data Sources
- Apple (AAPL) historical prices (CSV)
- S&P 500 historical index data (CSV)

## Metrics Calculated
- Annualized Return
- Annualized Volatility
- Sharpe Ratio (risk-free rate = 0)
- Cumulative Returns

## Outputs
- Performance metrics printed to terminal
- Comparative cumulative return plot saved in `outputs/plots/`

## Project Structure 
FINANCIAL-MARKET-ANALYTICS/
- analysis/
  --market_analysis.py

- data/
  --AAPL.csv
  --SP500.csv

- outputs/
  --plots/
  --apple_vs_sp500.png

- README.md
- requirements.txt 

## How to run
- pip install -r requirements.txt
- python market_analysis.py

Author: Ayush Raj Singh