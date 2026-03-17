=============================================================================
BYTE ALGO TRADING SPRINT 2026 - SUBMISSION
Team: Stonks++ 
Team members: Amritesh Kumar Rai, Ayush Kumar, Nishchay Saluja
SYSTEM OVERVIEW & THE "EDGE"
This trading agent is not a simple price-follower. It operates on a hybrid
architecture that combines a strict Quantitative Risk Engine with a Machine
Learning predictive model.

The core market pattern (our "edge") relies on the intersection of volume surges
and localized volatility compression. Rather than attempting to predict long-term
trends in a noisy, anonymized asset, the agent looks for short-term momentum
breakouts confirmed by institutional-level volume spikes, executed exclusively
when the Average Directional Index (ADX) confirms a valid regime.

To survive the 3-hour live simulation, the system isolates the predictive model
behind a rigorous 5-Layer Safety Framework, ensuring the bot prioritizes capital
preservation and Sharpe ratio stability over raw win rate.

=============================================================================
2. MACHINE LEARNING ARCHITECTURE
The predictive brain of the agent is a Voting Classifier Ensemble:

XGBoost: Optimized for rapid tabular pattern recognition (max_depth=5).

CatBoost: Configured with strict L2 leaf regularization (l2_leaf_reg=3) to
mathematically penalize overly complex decision trees and prevent overfitting
to the training data.

Stationarity & Feature Engineering:
Tree-based models cannot extrapolate absolute values. To prevent the model from
crashing if the Day 2 live asset opens at an unprecedented price, ALL absolute
prices (Open, High, Low, Close, raw SMAs) are dynamically dropped before
inference. The model evaluates 40+ purely relative features, including:

Z-Scores and Bollinger Band Widths (30-period)

Oscillators: RSI (14), Stochastic (14), and MACD histograms

Volatility: 14-period Average True Range (ATR) percentages

Momentum: 1-minute and 5-minute rolling percentage returns

=============================================================================
3. POSITION SIZING & CASH DECAY MITIGATION
The hackathon environment enforces a 60% maximum position cap and penalizes
undeployed cash with a 0.02% decay per minute.

To balance these constraints, the agent utilizes Aggressive Tiered Sizing:

Standard Deployment: When a valid confluence signal is generated, the agent
calculates the affordable shares to deploy exactly 59% of the current Net Worth.

This keeps the system legally under the 60% server cap while heavily minimizing
the cash decay penalty dragging down the idle portfolio.

=============================================================================
4. THE 5-LAYER SAFETY FRAMEWORK (RISK MANAGEMENT)
Every tick processed by the agent must survive five distinct checkpoints before
an order reaches the exchange:

LAYER 1: PRE-FLIGHT & REGIME DETECTION
The bot buffers the first 30 ticks to build reliable rolling indicators. It
then calculates the ADX. If the volume is anomalous but the trend is dead,
the system suppresses the ML model and holds cash.

LAYER 2: ML SIGNAL GENERATION
The XGBoost/CatBoost committee outputs a probability of upward movement.

BUY Threshold: > 0.55

SELL Threshold: < 0.45
Anything between 0.45 and 0.55 is treated as low-conviction noise and ignored.

LAYER 3: ENTRY CONFLUENCE
A high-probability ML Buy signal is still blocked if it fails market logic:

RSI Guard: Blocks buys if the asset is overbought (RSI > 60).

Volatility Guard: Blocks entries if the current candle range exceeds 3x the ATR.

Risk-Reward (R:R) Guard: Simulates a trade. If the distance to the Take-Profit
is not at least 1.5x the distance to the ATR Stop-Loss, the trade is aborted.

LAYER 4: POSITION MANAGEMENT (Dynamic Exits)
Once in a trade, the system ignores the ML model and manages risk mechanically:

Hard Stop-Loss: 1.5x ATR below entry (or fixed 2% fallback).

Break-Even Shift: Once the trade reaches +1.0% profit, the Stop-Loss is moved
up to the entry price plus the 0.1% exchange fee, guaranteeing a risk-free trade.

Tier 1 Take-Profit: At +1.5% profit, 50% of the position is automatically sold
to secure realized gains.

Tier 2 Trailing Stop: The remaining 50% is tracked against the peak price with
a tight 0.5% trailing stop to capture extended momentum.

LAYER 5: SYSTEM GUARD (The Kill Switch)

Drawdown Limit: If the portfolio Net Worth ever drops by 3% from its starting
value, the kill switch triggers, halting all trading to protect the remaining
capital and finalize the Sharpe ratio.

Smart Cooldown: After any trade closes, the bot forces a 9-tick (~90 second)
cooldown to prevent revenge-trading during whipsaw volatility.

=============================================================================
5. HOW TO RUN
Ensure the environment has the packages listed in requirements.txt.

Ensure models/xgb_model.pkl is present in the working directory.

Configure the .env file with the team API key and server URL.

Execute the agent:
$ python agent.py

The agent will automatically connect, buffer the required tick history, and
run autonomously for the duration of the market session.