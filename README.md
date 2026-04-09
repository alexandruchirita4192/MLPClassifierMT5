# MLP MT5 strategy files

## Files
- `train_mt5_mlp_classifier.py`
- `MT5_MLP_Classifier_ONNX_Strategy.mq5`

## Python Installation
```powershell
pip install MetaTrader5 pandas numpy scikit-learn skl2onnx onnx
```

## Running example
```powershell
python train_mt5_mlp_classifier.py --symbol XAGUSD --timeframe M15 --bars 20000 --horizon-bars 8 --train-ratio 0.70 --output-dir output_mlp_XAGUSD_M15_h8
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
