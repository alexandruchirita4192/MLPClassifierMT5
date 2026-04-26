# MLP MT5 strategy files
MLP has been ranked first between individual classifiers on the data I have tested, followed by LightGBM and HistGradientBoosting. On other data, an weighted ensemble of MLP 0.25, LightGBM 0.25, 0.50 HistGradientBoosting was better than MLP, LightGBM or HistGradientBoosting individually. In tests for XAGUSD it is doubling the money in ~6 months (10000 to ~20000). Disclaimer: Tested data based on history cannot guarantee future data, meaning that XAGUSD might behave differently generating less or no profit even if in tests it is profitable with some optimized parameters.

## Files
- `train_mt5_mlp_classifier.py`
- `MT5_MLP_Classifier_ONNX_Strategy.mq5`

## Python Installation
```powershell
pip install MetaTrader5 pandas numpy scikit-learn skl2onnx onnx
```

## Training example (non scale invariant example)
```powershell
python train_mt5_mlp_classifier.py --symbol XAGUSD --timeframe M15 --bars 80000 --horizon-bars 8 --train-ratio 0.82 --output-dir output_mlp_XAGUSD_M15_h8_82
```

## Steps for MT5
1. Copy `ml_strategy_classifier_mlp.onnx` near the `.mq5` file
2. Recompile the EA
3. Run the tester only on the `TEST UTC` from `run_in_mt5.txt`

## Recommended setup in the EA
- `InpUseTrendFilter = true`
- `InpTrendMAPeriod = 100`
- `InpUseTrendDistanceFilter = false`
- `InpUseAtrVolFilter = true`
- `InpAtrMinPercentile = 0.25`
- `InpAtrMaxPercentile = 0.85`
- `InpUseKillSwitch = false`

## Scale-invariant change
The model input feature `atr_14` was replaced with `atr_14_pct = atr_14 / close` in both Python and the EA. Raw ATR is still kept in the EA for stop-loss/take-profit distances because those order distances must remain in symbol price units.

After this change, retrain and export a new `ml_strategy_classifier_mlp.onnx`, then copy that new ONNX file next to the updated `.mq5` file and recompile.

## Scale invariant training example
```powershell
python train_mt5_mlp_classifier_scale_invariant.py --symbol XAGUSD --timeframe M15 --bars 80000 --horizon-bars 8 --train-ratio 0.82 --output-dir output_mlp_XAGUSD_M15_h8_82_scale_invariant
```
