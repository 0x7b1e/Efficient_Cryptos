"""Efficient_Cryptos: Binance USD-M factor research and execution.

Pipeline (one package, one direction of dependencies):

    common      config, local parquet storage, Binance REST client, costs, signal schedule
    data        liquidity universe selection (downloads live in scripts/01_data)
    features    leakage-safe bar/funding/positioning features and labels
    mining      LLM factor generator (DeepSeek/OpenAI/Grok) and its prompts
    factors     sandboxed execution + look-ahead checks for generated factors, Alpha101/191
    backtest    Python portfolio backtest, IC metrics, funding settlement mapping, reports
    engine      LIVE/SHADOW risk state machine shared by replay and live trading
    execution   Binance adapter, live runner, realtime signal sources, exchange-side guards
    ml          sequence models (torch) trained on the feature frame
    kol, meme   optional social-signal and meme-event research
"""

__version__ = "1.0.0"
