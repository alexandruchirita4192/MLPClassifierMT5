# MLP MT5 strategy files

## Fisiere
- `train_mt5_mlp_classifier.py`
- `MT5_MLP_Classifier_ONNX_Strategy.mq5`

## Instalare Python
```powershell
pip install MetaTrader5 pandas numpy scikit-learn skl2onnx onnx
```

## Exemplu rulare
```powershell
python train_mt5_mlp_classifier.py --symbol XAGUSD --timeframe M15 --bars 20000 --horizon-bars 8 --train-ratio 0.70 --output-dir output_mlp_XAGUSD_M15_h8
```

## Pasii pentru MT5
1. Copiaza `ml_strategy_classifier_mlp.onnx` langa fisierul `.mq5`
2. Recompileaza EA-ul
3. Ruleaza testerul doar pe `TEST UTC` din `run_in_mt5.txt`

## Setari de start recomandate in EA
- `InpUseTrendFilter = true`
- `InpTrendMAPeriod = 100`
- `InpUseTrendDistanceFilter = false`
- `InpUseAtrVolFilter = true`
- `InpAtrMinPercentile = 0.25`
- `InpAtrMaxPercentile = 0.85`
- `InpUseKillSwitch = false`
