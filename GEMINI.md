# GEMINI.md

This file provides an overview of the "Quantitative Trading" project, its architecture, and instructions for running it.

## Project Overview

This project is a quantitative stock analysis and position management system. It combines technical analysis indicators (MA60, RSI) with AI-powered sentiment analysis of recent news to generate daily trading suggestions. The system is designed to manage a portfolio, scan for new opportunities, and backtest strategies.

The core logic is written in Python and leverages several libraries:
- **Data Analysis:** `pandas`, `numpy`, `pandas_ta`
- **Financial Data:** `yfinance` for stock prices and news
- **AI/Sentiment Analysis:** `google-genai` for analyzing news headlines with a Gemini model.
- **Web Scraping:** `requests` and `beautifulsoup4` for fetching S&P 500 tickers.

### Architecture

The system's workflow is centered around a daily cycle of pre-market analysis and post-market confirmation.

- **Daily Workflow:**
  1.  **Pre-market:** `premarket_main.py` runs to analyze existing positions and scan for new opportunities. It loads the current portfolio, fetches the latest prices, performs risk checks, scans for candidates based on technicals, analyzes news sentiment using AI, and finally outputs a set of recommended actions to a `data/actions_YYYYMMDD.json` file.
  2.  **Post-market:** `confirm_main.py` is used to log which of the recommended actions were executed during the day, updating the master `data/portfolio.json`.

- **Core Modules (`src/`):**
  - `portfolio.py`: Manages the state of the portfolio, including positions and transactions.
  - `risk.py`: Implements risk management rules, such as stop-loss and position size limits.
  - `premarket.py`: The main decision-making engine that generates trading actions (HOLD, EXIT, ADD).
  - `data_loader.py`: Handles fetching stock data from `yfinance`, including a caching mechanism to avoid redundant downloads.
  - `strategy.py`: Defines the technical trading signals (e.g., Price > MA60 and RSI < 70).
  - `ai_analyst.py`: Fetches news headlines and uses the Gemini API to perform sentiment analysis, which gracefully degrades to a neutral score if API limits are reached.
  - `backtester.py`: An engine for running historical backtests of the trading strategies.
  - `main.py`: The main entry point for running historical stress tests.

## Building and Running

### 1. Installation

Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

### 2. Configuration

The system requires an API key for the Gemini model. Create a `.env` file in the root directory and add your key:

```
GEMINI_API_KEY=<your_google_gemini_api_key>
```

### 3. Running the Application

The project has several entry points for different tasks:

- **Pre-market Analysis (Primary use):**
  Generates daily trading recommendations.
  ```bash
  # On first use, run with --init to create the portfolio interactively
  python premarket_main.py --init

  # Subsequently, run daily to get action recommendations
  python premarket_main.py
  ```

- **Post-market Confirmation:**
  Confirms the trades that were executed based on the day's recommendations.
  ```bash
  # Replace with the actual date of the actions file
  python confirm_main.py YYYY-MM-DD
  ```

- **Stress Testing:**
  Runs the strategy over a historical period to evaluate performance during market downturns.
  ```bash
  python main.py
  ```

- **Standalone Scanner:**
  Runs the scanning module independently.
  ```bash
  python scanner_main.py
  ```

## Development Conventions

- **No Formal Tests:** The project currently lacks a formal test suite, linter, or CI/CD pipeline.
- **Strategy Logic:** The primary strategy is a dual-factor model combining a 60-day moving average (MA60) for trend and the Relative Strength Index (RSI) for momentum.
  - **Buy Signal:** `Price > MA60` AND `RSI < 70`
  - **Sell Signal:** `Price < MA60` OR `RSI > 85`
- **Data Caching:** Historical stock data fetched from `yfinance` is cached in the `data/` directory as CSV files to speed up subsequent runs.
- **State Management:** The primary state of the portfolio is stored in `data/portfolio.json`. Daily action plans are stored in `data/actions_YYYYMMDD.json`.
- **AI Graceful Degradation:** The AI sentiment analysis module is designed to handle API errors (e.g., rate limits, billing issues) by defaulting to a neutral sentiment score, allowing the rest of the system to function based on technicals alone.
