<div align="center">

# Regime-Aware Investment Research Copilot

### A desk-style multi-agent financial ML system that detects hidden market regimes, routes predictions through specialized expert classifiers, and resolves conflicting signals — inspired by how quantitative research desks combine regime, risk, macro, and forecasting signals.

<br/>

-> https://regime-aware-investment-copilot.streamlit.app/

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Charts-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

<br/>

<!-- SCREENSHOT: Full dashboard overview — paste your Desk View screenshot here -->
> ![Dashboard Overview](assets/dashboard_overview.png)

</div>

---

## The Problem This Solves

A single ML model trained across all market conditions is a structural mistake.

Bull markets, bear markets, and crisis periods have fundamentally different volatility profiles, momentum dynamics, and return distributions. Training one model across all three forces it to average out signal that only exists in specific regimes — and dilutes performance in all of them.

This system solves that with a three-layer approach:

1. **Detect the hidden regime** — HMM identifies whether the market is currently in a Bull, Bear, or Crisis state
2. **Route to specialists** — Soft-Gated Mixture of Experts weights regime-specialized classifiers dynamically
3. **Resolve conflicts explicitly** — A supervisor agent reconciles when structural regime and momentum signal disagree, with transparent reasoning and regime-conditional position scaling

---

## Results

<div align="center">

| Metric | Value |
|:---|:---:|
| Best model | Random Forest + Soft-Gated MoE |
| **Walk-forward Sharpe (all-asset)** | **2.329** |
| Baseline Sharpe (no regime routing) | 1.917 |
| **Net Sharpe after 10 bps transaction costs** | **2.298** |
| Buy-and-hold Sharpe | 1.732 |
| **Strategy max drawdown** | **-5.02%** |
| Buy-and-hold max drawdown | -10.85% |
| Bull regime directional accuracy | 64.7% |

</div>

> All results from walk-forward validation. No look-ahead bias. Transaction costs and volatility-based position sizing applied throughout.

### Regime Persistence

The Hidden Markov Model identified highly persistent market states:

| Regime | Self-Transition Probability |
|:---|:---:|
| Bull | 96.9% |
| Bear | 98.3% |
| Crisis | 96.3% |

Regimes are not noise — they are structurally stable. This persistence is what justifies routing predictions to regime-specialized experts rather than using a single global forecasting model. A regime, once entered, is statistically likely to continue.

---

## System Architecture

```
                     ┌───────────────────────────────┐
                     │         CONTROL PANEL         │
                     │  Asset · Cost · Signal · Shock│
                     └──────────────┬────────────────┘
                                    │
                     ┌──────────────▼────────────────┐
                     │         REGIME AGENT          │
                     │   HMM → Bull / Bear / Crisis  │
                     │   Posterior state probability │
                     └──────────────┬────────────────┘
                                    │
         ┌──────────────────────────┼─────────────────────────┐
         │                          │                         │
┌────────▼────────┐      ┌──────────▼──────────┐    ┌────────▼────────┐
│   QUANT AGENT   │      │     RISK AGENT      │    │   MACRO AGENT   │
│ Soft-Gated MoE  │      │ Vol-based sizing    │    │ FRED: FEDFUNDS  │
│ RF/XGB/LightGBM │      │ Regime scalar       │    │ T10Y2Y, HY Sprd │
│ Up probability  │      │ VaR · Drawdown      │    │ India VIX / RBI │
└────────┬────────┘      └──────────┬──────────┘    └────────┬────────┘
         │                          │                        │
         └──────────────────────────▼────────────────────────┘
                                    │
                     ┌──────────────▼─────────────────┐
                     │       SUPERVISOR AGENT         │
                     │  Explicit conflict resolution  │
                     │  Bear + Bullish → 40% exposure │
                     │  Regime-conditional override   │
                     └──────────────┬─────────────────┘
                                    │
         ┌──────────────────────────┼─────────────────────────┐
         │                          │                         │
┌────────▼─────────┐      ┌─────────▼───────────┐    ┌────────▼─────────┐
│ HIST. ANALOGS    │      │   RESEARCH NOTE     │    │ MODEL DIAGNOSTICS│
│ Cosine similarity│      │ Structured synthesis│    │ Walk-fwd tracker │
│ Fwd return dist  │      │ from agent inputs   │    │ Model leaderboard│
└──────────────────┘      └─────────────────────┘    └──────────────────┘
```

---

## Dashboard

### Desk View — Live Agent Status + Conflict Resolution

<!-- SCREENSHOT: Desk View tab — showing Bear regime, Bullish quant, 40% supervisor exposure -->
![Dashboard Overview](assets/dashboard_overview.png)

The headline feature. When the Regime Agent and Quant Agent disagree — for example, HMM detects Bear with 99.9% confidence while the MoE classifier gives a Bullish signal — the Supervisor Agent doesn't average them. It applies explicit logic:

```python
("Bear", "Bullish"): {
    "stance":            "Cautiously Neutral",
    "reasoning":         "Regime and momentum signals disagree. "
                         "Reduce size and wait for confirmation.",
    "exposure_override": 0.40   # down from raw 100%
}
```

Every decision is shown with its reasoning. No black boxes.

---

### Research Copilot — Structured Investment Note

<!-- SCREENSHOT: Research Copilot tab — showing the investment note + structured JSON inputs -->
![Research Copilot](assets/research_copilot.png)

All agent outputs are assembled into a structured JSON context block, then synthesized into a 3-paragraph investment note that:
- States the regime and what HMM confidence implies about market structure
- Explicitly resolves the conflict between regime and quant signal
- References the historical analog median forward return
- States the desk action with specific exposure and primary risk factors

---

### Risk Lab — Portfolio Analytics

<!-- SCREENSHOT: Risk Lab tab — showing equity curve + cost sensitivity chart -->
![Risk Lab](assets/risk_lab.png)

- Supervisor strategy vs buy-and-hold equity curve (walk-forward window)
- Portfolio cost sensitivity: Sharpe vs transaction costs from 0–20 bps
- Stress scenario calculator: if next session return = X, current exposure impact = Y

---

### Historical Analogs — Nearest Neighbor Regime Retrieval

<!-- SCREENSHOT: Historical Analogs tab — showing the analog table with forward returns -->
![Historical Analogs](assets/historical_analogs.png)

Retrieves the closest historical market states using cosine similarity over:
- Regime label
- 30-day rolling volatility
- Current drawdown
- Asset-specific VIX level
- Momentum

Shows forward 10D and 30D returns from those analog periods, with median recovery days — providing empirical base rates for the current setup.

---

### Model Diagnostics — Full Leaderboard + Accuracy Breakdown

<!-- SCREENSHOT: Model Diagnostics tab — showing leaderboard + accuracy by regime chart -->
![Model Diagnostics](assets/model_diag.png)
![Model Diagnostics](assets/model_diag2.png)

| Model | Routing | Sharpe | Max DD | Accuracy |
|---|---|---|---|---|
| Random Forest | Soft MoE | **2.329** | -10.57% | 53.9% |
| LightGBM | Soft MoE | 2.184 | -8.03% | 50.3% |
| HistGradientBoosting | Soft MoE | 1.920 | -8.68% | 50.1% |
| Random Forest | Baseline | 1.917 | -12.72% | 52.9% |
| XGBoost | Soft MoE | 1.915 | -10.28% | 50.9% |

Directional accuracy broken down by regime and recency window (Last 30 / 60 / 90 / Bull / Bear / Crisis).

> Crisis accuracy is 28% — honestly reported. Crisis periods are rare (25 observations), non-stationary, and structurally different from Bull/Bear dynamics. This result is consistent with the academic literature on regime-switching model limitations during structural breaks.

---

## Assets

| Asset | Ticker | VIX Proxy |
|:---|:---:|:---:|
| S&P 500 | `^GSPC` | `^VIX` |
| NASDAQ | `^IXIC` | `^VIX` |
| NIFTY 50 | `^NSEI` | `^INDIAVIX` |

---

## Notebook Pipeline

| # | Notebook | Purpose |
|:---:|:---|:---|
| 01 | `01_download_market_data` | Download raw market data via Yahoo Finance |
| 02 | `02_feature_engineering` | Returns, volatility, momentum, moving averages, drawdown |
| 03 | `03_hmm_regime_detection` | HMM training and Bull / Bear / Crisis labeling |
| 04 | `04_mixture_of_experts` | Return-regression MoE baseline |
| 05 | `05_directional_moe_classifier` | Direction classification MoE |
| 06 | `06_walk_forward_validation` | Walk-forward backtesting framework |
| 07 | `07_final_research_report` | S&P 500 focused report |
| 08 | `08_all_assets_boosted_experts` | All-asset RF / XGBoost / LightGBM comparison |
| 09 | `09_final_all_asset_report` | Final multi-asset research report |
| 10 | `10_transaction_costs_position_sizing` | Transaction costs + volatility-based position sizing |
| 11 | `11_india_vix_macro_enrichment` | India VIX, FRED, RBI macro enrichment |
| 12 | `12_advanced_regime_and_deep_models` | Research extensions: Markov-switching ARIMA, neural gating experiments |
| 13 | `13_macro_enriched_hmm_moe` | HMM + MoE rerun with macro-enriched features |

---

## Tech Stack

```
Core ML          hmmlearn · scikit-learn · XGBoost · LightGBM
Data             yfinance · pandas-datareader · FRED API
Feature Eng.     pandas · NumPy
Dashboard        Streamlit · Plotly
Macro            FRED (FEDFUNDS · T10Y2Y · BAMLH0A0HYM2) · India VIX · RBI DBIE
```

---

## Setup

```bash
git clone https://github.com/Niyati001/Regime-Aware-Investment-Copilot.git
cd Regime-Aware-Investment-Copilot

py -3.11 -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install --upgrade pip
pip install -r requirements.txt
```

**Launch the dashboard:**
```bash
streamlit run app.py
```

**Run notebooks:**
```bash
python -m ipykernel install --user --name market-regime --display-name "Python 3.11 Market Regime."
jupyter lab
```

---

## Optional: FRED Macro Data

Get a free API key at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html):

```bash
# Windows
setx FRED_API_KEY "your_key_here"

# macOS / Linux
export FRED_API_KEY="your_key_here"
```

The system fetches `FEDFUNDS`, `T10Y2Y`, and `BAMLH0A0HYM2` automatically.

---

## Optional: RBI Macro Data

Export selected series from [RBI DBIE](https://dbie.rbi.org.in) as CSV into:

```
data/external/rbi_macro.csv
```

Expected columns: `Date, india_cpi, india_repo_rate, india_gdp_growth, india_10y_yield, usd_inr`

---

## Research Conclusions

- Regime-aware routing consistently outperformed single-model baselines in walk-forward testing across all three assets
- Soft-Gated MoE outperformed Hard MoE and baseline across every model family tested
- Random Forest was the strongest expert classifier at Sharpe 2.329 — outperforming XGBoost (1.915) and LightGBM (2.184)
- India VIX improved NIFTY-specific regime financial correctness; macro enrichment did not improve overall Sharpe on this dataset — an honest null result
- Walk-forward Sharpe degrades from 2.33 in-sample to ~1.06 on recent NIFTY out-of-sample windows — the realistic live-trading expectation, reported transparently

---

<div align="center">

*Research prototype. Not investment advice.*
*Designed to demonstrate regime-aware decision support, explicit agent conflict resolution, and model-risk transparency.*

</div>
