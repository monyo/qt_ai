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
  1.  **Pre-market:** `premarket_main.py` runs to analyze existing positions and scan for new opportunities. It loads the current portfolio, fetches the latest prices, performs risk checks, scans for candidates based on technicals, analyzes news sentiment using AI, outputs a set of recommended actions to a `data/actions_YYYYMMDD.json` file, and sends a formatted HTML email report.
  2.  **Post-market:** `confirm_main.py` is used to log which of the recommended actions were executed during the day, updating the master `data/portfolio.json`.

- **Core Modules (`src/`):**
  - `portfolio.py`: Manages the state of the portfolio, including positions and transactions.
  - `risk.py`: Implements risk management rules, such as stop-loss and position size limits.
  - `premarket.py`: The main decision-making engine that generates trading actions (HOLD, EXIT, ADD).
  - `data_loader.py`: Handles fetching stock data from `yfinance`, including a caching mechanism to avoid redundant downloads.
  - `strategy.py`: Defines the technical trading signals (e.g., Price > MA60 and RSI < 70).
  - `ai_analyst.py`: Fetches news headlines and uses the Gemini API to perform sentiment analysis, which gracefully degrades to a neutral score if API limits are reached.
  - `notifier.py`: Formats and sends the daily pre-market analysis report via Gmail SMTP.
  - `backtester.py`: An engine for running historical backtests of the trading strategies.
  - `main.py`: The main entry point for running historical stress tests.

## Building and Running

### 1. Installation

Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

### 2. Configuration

The system is configured via a `.env` file in the root directory.

- **`GEMINI_API_KEY`**: Your API key for the Google Gemini model, used for sentiment analysis.
- **`EMAIL_ENABLED`**: Set to `true` to enable email notifications.
- **`GMAIL_SENDER`**: The Gmail address from which the reports will be sent.
- **`GMAIL_APP_PASSWORD`**: An "App Password" generated from your Google account for authentication. This is not your regular login password.
- **`GMAIL_RECIPIENT`**: The email address that will receive the reports.

Example `.env` file:
```
GEMINI_API_KEY=<your_google_gemini_api_key>
EMAIL_ENABLED=true
GMAIL_SENDER=your_gmail@gmail.com
GMAIL_APP_PASSWORD=your_gmail_app_password
GMAIL_RECIPIENT=your_recipient_email@gmail.com
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
- **Strategy Logic:** The current core strategy is momentum-based.
  - **Candidate Pool:** The system scans the full S&P 500 list, plus a user-defined watchlist and existing holdings.
  - **Entry Signal:** Recommends buying stocks that rank in the top 5 by momentum (based on the last 21 days of performance).
  - **Exit Signals:** Uses a three-tiered exit system: a trailing stop-loss from the high, a break below the 200-day moving average (MA200), and a hard stop-loss based on the entry price.
- **Data Caching:** Historical stock data fetched from `yfinance` is cached in the `data/` directory as CSV files to speed up subsequent runs.
- **State Management:** The primary state of the portfolio is stored in `data/portfolio.json`. Daily action plans are stored in `data/actions_YYYYMMDD.json`.
- **AI Graceful Degradation:** The AI sentiment analysis module is designed to handle API errors (e.g., rate limits, billing issues) by defaulting to a neutral sentiment score, allowing the rest of the system to function based on technicals alone.
