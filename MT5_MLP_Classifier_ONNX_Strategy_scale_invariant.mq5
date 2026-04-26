#property strict
#property version   "1.01"
#property description "EA MT5: MLP classifier ONNX, run in Strategy Tester"
#property description "With trend filter + ATR volatility filter + optional kill switch"
#property description "Model features are scale-invariant; raw ATR is kept only for stops/filters"

#include <Trade/Trade.mqh>

#resource "ml_strategy_classifier_mlp.onnx" as uchar ExtModel[]

input double InpLots                  = 0.10;
input double InpEntryProbThreshold    = 0.60;
input double InpMinProbGap            = 0.15;
input bool   InpUseAtrStops           = true;
input double InpStopAtrMultiple       = 1.50;
input double InpTakeAtrMultiple       = 2.00;
input int    InpMaxBarsInTrade        = 8;
input bool   InpCloseOnOppositeSignal = true;
input bool   InpAllowLong             = true;
input bool   InpAllowShort            = true;

input bool   InpUseTrendFilter        = true;
input ENUM_TIMEFRAMES InpTrendTF      = PERIOD_H1;
input int    InpTrendMAPeriod         = 100;
input bool   InpTrendRequireSlope     = true;

input bool   InpUseTrendDistanceFilter = false;
input double InpTrendMinDistancePct    = 0.0010;

input bool   InpUseAtrVolFilter        = true;
input int    InpAtrVolLookback         = 50;
input double InpAtrMinPercentile       = 0.25;
input double InpAtrMaxPercentile       = 0.85;

input bool   InpUseKillSwitch                 = false;
input int    InpKillSwitchLookbackTrades      = 8;
input double InpKillSwitchMinWinRate          = 0.40;
input double InpKillSwitchMinProfitFactor     = 0.95;
input int    InpKillSwitchConsecutiveLosses   = 4;
input int    InpKillSwitchPauseBars           = 96;
input bool   InpKillSwitchFlatOnActivate      = true;

input long   InpMagic                 = 26042026;
input bool   InpLog                   = false;
input bool   InpDebugLog              = false;

const int FEATURE_COUNT = 10;
const int CLASS_COUNT   = 3;
const long EXT_INPUT_SHAPE[]  = {1, FEATURE_COUNT};
const long EXT_LABEL_SHAPE[]  = {1};
const long EXT_PROBA_SHAPE[]  = {1, CLASS_COUNT};

CTrade trade;
long g_model_handle = INVALID_HANDLE;
int  g_trend_ma_handle = INVALID_HANDLE;

datetime g_last_bar_time = 0;
int g_bars_in_trade = 0;

bool g_kill_switch_active = false;
int  g_kill_switch_pause_remaining = 0;
int  g_consecutive_losses = 0;
double g_recent_closed_profits[];
int g_last_history_deals_total = 0;

enum SignalDirection
  {
   SIGNAL_SELL = -1,
   SIGNAL_FLAT =  0,
   SIGNAL_BUY  =  1
  };

bool IsNewBar()
  {
   datetime current_bar_time = iTime(_Symbol, _Period, 0);
   if(current_bar_time == 0)
      return false;

   if(g_last_bar_time == 0)
     {
      g_last_bar_time = current_bar_time;
      return false;
     }

   if(current_bar_time != g_last_bar_time)
     {
      g_last_bar_time = current_bar_time;
      return true;
     }
   return false;
  }

double Mean(const double &arr[], int start_shift, int count)
  {
   double sum = 0.0;
   for(int i = start_shift; i < start_shift + count; i++)
      sum += arr[i];
   return sum / count;
  }

double StdDev(const double &arr[], int start_shift, int count)
  {
   double m = Mean(arr, start_shift, count);
   double s = 0.0;
   for(int i = start_shift; i < start_shift + count; i++)
     {
      double d = arr[i] - m;
      s += d * d;
     }
   return MathSqrt(s / MathMax(count - 1, 1));
  }

double CalcATR(const MqlRates &rates[], int start_shift, int period)
  {
   double sum_tr = 0.0;
   for(int i = start_shift; i < start_shift + period; i++)
     {
      double high = rates[i].high;
      double low = rates[i].low;
      double prev_close = rates[i + 1].close;
      double tr1 = high - low;
      double tr2 = MathAbs(high - prev_close);
      double tr3 = MathAbs(low - prev_close);
      double tr = MathMax(tr1, MathMax(tr2, tr3));
      sum_tr += tr;
     }
   return sum_tr / period;
  }

double GetPercentileFromArray(const double &arr[], int count, double q)
  {
   if(count <= 0)
      return 0.0;

   if(q <= 0.0)
      q = 0.0;
   if(q >= 1.0)
      q = 1.0;

   double tmp[];
   ArrayResize(tmp, count);
   for(int i = 0; i < count; i++)
      tmp[i] = arr[i];

   ArraySort(tmp);

   double pos = q * (count - 1);
   int lo = (int)MathFloor(pos);
   int hi = (int)MathCeil(pos);

   if(lo == hi)
      return tmp[lo];

   double w = pos - lo;
   return tmp[lo] * (1.0 - w) + tmp[hi] * w;
  }

bool BuildFeatureVector(matrixf &features, double &atr14)
  {
   MqlRates rates[];
   ArraySetAsSeries(rates, true);

   if(CopyRates(_Symbol, _Period, 0, 80, rates) < 40)
      return false;

   double closes[];
   ArrayResize(closes, ArraySize(rates));
   ArraySetAsSeries(closes, true);
   for(int i = 0; i < ArraySize(rates); i++)
      closes[i] = rates[i].close;

   int s = 1;

   double ret_1  = (closes[s] / closes[s + 1]) - 1.0;
   double ret_3  = (closes[s] / closes[s + 3]) - 1.0;
   double ret_5  = (closes[s] / closes[s + 5]) - 1.0;
   double ret_10 = (closes[s] / closes[s + 10]) - 1.0;

   double one_bar_returns[];
   ArrayResize(one_bar_returns, 30);
   for(int i = 0; i < 30; i++)
      one_bar_returns[i] = (closes[s + i] / closes[s + i + 1]) - 1.0;

   double vol_10 = StdDev(one_bar_returns, 0, 10);
   double vol_20 = StdDev(one_bar_returns, 0, 20);

   double sma_10 = Mean(closes, s, 10);
   double sma_20 = Mean(closes, s, 20);
   if(sma_10 == 0.0 || sma_20 == 0.0)
      return false;

   double dist_sma_10 = (closes[s] / sma_10) - 1.0;
   double dist_sma_20 = (closes[s] / sma_20) - 1.0;

   double mean_20 = Mean(closes, s, 20);
   double std_20  = StdDev(closes, s, 20);
   double zscore_20 = 0.0;
   if(std_20 > 0.0)
      zscore_20 = (closes[s] - mean_20) / std_20;

   atr14 = CalcATR(rates, s, 14);
   if(closes[s] <= 0.0)
      return false;

   // Scale-invariant model input: Python trains on atr_14_pct = atr_14 / close.
   // Keep raw atr14 as the output parameter because stops need price-distance units.
   double atr14_pct = atr14 / closes[s];

   features.Resize(1, FEATURE_COUNT);
   features[0][0] = (float)ret_1;
   features[0][1] = (float)ret_3;
   features[0][2] = (float)ret_5;
   features[0][3] = (float)ret_10;
   features[0][4] = (float)vol_10;
   features[0][5] = (float)vol_20;
   features[0][6] = (float)dist_sma_10;
   features[0][7] = (float)dist_sma_20;
   features[0][8] = (float)zscore_20;
   features[0][9] = (float)atr14_pct;
   return true;
  }

bool PredictClassProbabilities(double &pSell, double &pFlat, double &pBuy, double &atr14)
  {
   matrixf x;
   if(!BuildFeatureVector(x, atr14))
      return false;

   long predicted_label[1];
   matrixf probs;
   probs.Resize(1, CLASS_COUNT);

   if(!OnnxRun(g_model_handle, 0, x, predicted_label, probs))
      return false;

   pSell = probs[0][0];
   pFlat = probs[0][1];
   pBuy  = probs[0][2];
   return true;
  }

SignalDirection SignalFromProbabilities(double pSell, double pFlat, double pBuy)
  {
   double best = pFlat;
   double second = -1.0;
   SignalDirection signal = SIGNAL_FLAT;

   if(pBuy >= pSell && pBuy > best)
     {
      second = MathMax(best, pSell);
      best = pBuy;
      signal = SIGNAL_BUY;
     }
   else if(pSell > pBuy && pSell > best)
     {
      second = MathMax(best, pBuy);
      best = pSell;
      signal = SIGNAL_SELL;
     }
   else
     {
      second = MathMax(pBuy, pSell);
      signal = SIGNAL_FLAT;
     }

   double gap = best - second;

   if(signal == SIGNAL_BUY)
     {
      if(!InpAllowLong)
         return SIGNAL_FLAT;
      if(pBuy < InpEntryProbThreshold || gap < InpMinProbGap)
         return SIGNAL_FLAT;
      return SIGNAL_BUY;
     }

   if(signal == SIGNAL_SELL)
     {
      if(!InpAllowShort)
         return SIGNAL_FLAT;
      if(pSell < InpEntryProbThreshold || gap < InpMinProbGap)
         return SIGNAL_FLAT;
      return SIGNAL_SELL;
     }

   return SIGNAL_FLAT;
  }

bool GetTrendFilterValues(double &htf_close_1, double &ema_1, double &ema_2)
  {
   if(!InpUseTrendFilter)
      return true;

   if(g_trend_ma_handle == INVALID_HANDLE)
      return false;

   htf_close_1 = iClose(_Symbol, InpTrendTF, 1);
   if(htf_close_1 == 0.0)
      return false;

   double ema_buf[];
   ArraySetAsSeries(ema_buf, true);

   if(CopyBuffer(g_trend_ma_handle, 0, 1, 2, ema_buf) < 2)
      return false;

   ema_1 = ema_buf[0];
   ema_2 = ema_buf[1];
   return true;
  }

bool TrendAllows(SignalDirection signal)
  {
   if(!InpUseTrendFilter || signal == SIGNAL_FLAT)
      return true;

   double htf_close_1 = 0.0;
   double ema_1 = 0.0;
   double ema_2 = 0.0;
   if(!GetTrendFilterValues(htf_close_1, ema_1, ema_2))
      return false;

   bool slope_up   = (ema_1 > ema_2);
   bool slope_down = (ema_1 < ema_2);

   double distance_pct = 0.0;
   if(ema_1 != 0.0)
      distance_pct = MathAbs(htf_close_1 - ema_1) / ema_1;

   if(InpUseTrendDistanceFilter && distance_pct < InpTrendMinDistancePct)
      return false;

   if(signal == SIGNAL_BUY)
     {
      if(htf_close_1 <= ema_1)
         return false;
      if(InpTrendRequireSlope && !slope_up)
         return false;
      return true;
     }

   if(signal == SIGNAL_SELL)
     {
      if(htf_close_1 >= ema_1)
         return false;
      if(InpTrendRequireSlope && !slope_down)
         return false;
      return true;
     }

   return true;
  }

bool AtrVolatilityAllows(double current_atr14)
  {
   if(!InpUseAtrVolFilter)
      return true;

   MqlRates rates[];
   ArraySetAsSeries(rates, true);

   int need_bars = InpAtrVolLookback + 20;
   if(CopyRates(_Symbol, _Period, 0, need_bars, rates) < need_bars)
      return false;

   double atr_values[];
   ArrayResize(atr_values, InpAtrVolLookback);

   int s = 1;
   for(int i = 0; i < InpAtrVolLookback; i++)
      atr_values[i] = CalcATR(rates, s + i, 14);

   double atr_min = GetPercentileFromArray(atr_values, InpAtrVolLookback, InpAtrMinPercentile);
   double atr_max = GetPercentileFromArray(atr_values, InpAtrVolLookback, InpAtrMaxPercentile);

   if(current_atr14 < atr_min)
      return false;
   if(current_atr14 > atr_max)
      return false;

   return true;
  }

bool HasOpenPosition(long &pos_type, double &pos_price)
  {
   if(!PositionSelect(_Symbol))
      return false;

   if((long)PositionGetInteger(POSITION_MAGIC) != InpMagic)
      return false;

   pos_type = (long)PositionGetInteger(POSITION_TYPE);
   pos_price = PositionGetDouble(POSITION_PRICE_OPEN);
   return true;
  }

void CloseOpenPosition()
  {
   if(PositionSelect(_Symbol) && (long)PositionGetInteger(POSITION_MAGIC) == InpMagic)
      trade.PositionClose(_Symbol);
  }

void PushClosedTradeProfit(double value)
  {
   int size = ArraySize(g_recent_closed_profits);
   ArrayResize(g_recent_closed_profits, size + 1);
   g_recent_closed_profits[size] = value;

   if(ArraySize(g_recent_closed_profits) > InpKillSwitchLookbackTrades)
     {
      for(int i = 1; i < ArraySize(g_recent_closed_profits); i++)
         g_recent_closed_profits[i - 1] = g_recent_closed_profits[i];
      ArrayResize(g_recent_closed_profits, InpKillSwitchLookbackTrades);
     }
  }

void ActivateKillSwitch(string reason)
  {
   if(!InpUseKillSwitch)
      return;

   g_kill_switch_active = true;
   g_kill_switch_pause_remaining = InpKillSwitchPauseBars;

   if(InpKillSwitchFlatOnActivate)
      CloseOpenPosition();
  }

void DecrementKillSwitchPause()
  {
   if(!g_kill_switch_active)
      return;

   if(g_kill_switch_pause_remaining > 0)
      g_kill_switch_pause_remaining--;

   if(g_kill_switch_pause_remaining <= 0)
     {
      g_kill_switch_active = false;
      g_consecutive_losses = 0;
      ArrayResize(g_recent_closed_profits, 0);
     }
  }

void EvaluateKillSwitch()
  {
   if(!InpUseKillSwitch || g_kill_switch_active)
      return;

   if(g_consecutive_losses >= InpKillSwitchConsecutiveLosses)
     {
      ActivateKillSwitch("");
      return;
     }

   int n = ArraySize(g_recent_closed_profits);
   if(n < InpKillSwitchLookbackTrades)
      return;

   int wins = 0;
   double gross_profit = 0.0;
   double gross_loss_abs = 0.0;

   for(int i = 0; i < n; i++)
     {
      double p = g_recent_closed_profits[i];
      if(p > 0.0)
        {
         wins++;
         gross_profit += p;
        }
      else if(p < 0.0)
        {
         gross_loss_abs += MathAbs(p);
        }
     }

   double win_rate = (double)wins / (double)n;
   double profit_factor = (gross_loss_abs > 0.0 ? gross_profit / gross_loss_abs : 999.0);

   if(win_rate < InpKillSwitchMinWinRate)
     {
      ActivateKillSwitch("");
      return;
     }

   if(profit_factor < InpKillSwitchMinProfitFactor)
     {
      ActivateKillSwitch("");
      return;
     }
  }

void UpdateClosedTradeStats()
  {
   if(!InpUseKillSwitch)
      return;

   if(!HistorySelect(0, TimeCurrent()))
      return;

   int total = HistoryDealsTotal();
   if(total <= g_last_history_deals_total)
      return;

   for(int i = g_last_history_deals_total; i < total; i++)
     {
      ulong deal_ticket = HistoryDealGetTicket(i);
      if(deal_ticket == 0)
         continue;

      string symbol = HistoryDealGetString(deal_ticket, DEAL_SYMBOL);
      long magic    = HistoryDealGetInteger(deal_ticket, DEAL_MAGIC);
      long entry    = HistoryDealGetInteger(deal_ticket, DEAL_ENTRY);

      if(symbol != _Symbol || magic != InpMagic || entry != DEAL_ENTRY_OUT)
         continue;

      double profit     = HistoryDealGetDouble(deal_ticket, DEAL_PROFIT);
      double swap       = HistoryDealGetDouble(deal_ticket, DEAL_SWAP);
      double commission = HistoryDealGetDouble(deal_ticket, DEAL_COMMISSION);
      double net = profit + swap + commission;

      PushClosedTradeProfit(net);

      if(net < 0.0)
         g_consecutive_losses++;
      else if(net > 0.0)
         g_consecutive_losses = 0;
     }

   g_last_history_deals_total = total;
   EvaluateKillSwitch();
  }

void OpenTrade(SignalDirection signal, double atr14)
  {
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);

   double min_stop = (double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * point;
   double sl_dist = MathMax(atr14 * InpStopAtrMultiple, min_stop);
   double tp_dist = MathMax(atr14 * InpTakeAtrMultiple, min_stop);

   double sl = 0.0;
   double tp = 0.0;

   trade.SetExpertMagicNumber(InpMagic);
   trade.SetDeviationInPoints(20);

   bool ok = false;

   if(signal == SIGNAL_BUY)
     {
      if(InpUseAtrStops)
        {
         sl = ask - sl_dist;
         tp = ask + tp_dist;
        }
      ok = trade.Buy(InpLots, _Symbol, ask, sl, tp, "MLP buy");
      if(ok)
         g_bars_in_trade = 0;
     }
   else if(signal == SIGNAL_SELL)
     {
      if(InpUseAtrStops)
        {
         sl = bid + sl_dist;
         tp = bid - tp_dist;
        }
      ok = trade.Sell(InpLots, _Symbol, bid, sl, tp, "MLP sell");
      if(ok)
         g_bars_in_trade = 0;
     }
  }

void ManageExistingPosition(SignalDirection signal)
  {
   long pos_type;
   double pos_price;
   if(!HasOpenPosition(pos_type, pos_price))
      return;

   g_bars_in_trade++;
   bool should_close = false;

   if(InpCloseOnOppositeSignal)
     {
      if(pos_type == POSITION_TYPE_BUY  && signal == SIGNAL_SELL)
         should_close = true;
      if(pos_type == POSITION_TYPE_SELL && signal == SIGNAL_BUY)
         should_close = true;
     }

   if(!should_close && g_bars_in_trade >= InpMaxBarsInTrade)
      should_close = true;

   if(should_close)
      CloseOpenPosition();
  }

int OnInit()
  {
   trade.SetExpertMagicNumber(InpMagic);

   g_model_handle = OnnxCreateFromBuffer(ExtModel, ONNX_DEFAULT);
   if(g_model_handle == INVALID_HANDLE)
      return INIT_FAILED;

   if(!OnnxSetInputShape(g_model_handle, 0, EXT_INPUT_SHAPE))
      return INIT_FAILED;

   if(!OnnxSetOutputShape(g_model_handle, 0, EXT_LABEL_SHAPE))
      return INIT_FAILED;

   if(!OnnxSetOutputShape(g_model_handle, 1, EXT_PROBA_SHAPE))
      return INIT_FAILED;

   if(InpUseTrendFilter)
     {
      g_trend_ma_handle = iMA(_Symbol, InpTrendTF, InpTrendMAPeriod, 0, MODE_EMA, PRICE_CLOSE);
      if(g_trend_ma_handle == INVALID_HANDLE)
         return INIT_FAILED;
     }

   if(HistorySelect(0, TimeCurrent()))
      g_last_history_deals_total = HistoryDealsTotal();
   else
      g_last_history_deals_total = 0;

   ArrayResize(g_recent_closed_profits, 0);
   g_consecutive_losses = 0;
   g_kill_switch_active = false;
   g_kill_switch_pause_remaining = 0;

   return INIT_SUCCEEDED;
  }

void OnDeinit(const int reason)
  {
   if(g_model_handle != INVALID_HANDLE)
      OnnxRelease(g_model_handle);

   if(g_trend_ma_handle != INVALID_HANDLE)
      IndicatorRelease(g_trend_ma_handle);
  }

void OnTick()
  {
   if(!IsNewBar())
      return;

   UpdateClosedTradeStats();
   DecrementKillSwitchPause();

   if(g_kill_switch_active)
      return;

   double pSell = 0.0;
   double pFlat = 0.0;
   double pBuy  = 0.0;
   double atr14 = 0.0;

   if(!PredictClassProbabilities(pSell, pFlat, pBuy, atr14))
      return;

   if(!AtrVolatilityAllows(atr14))
     {
      ManageExistingPosition(SIGNAL_FLAT);
      return;
     }

   SignalDirection raw_signal = SignalFromProbabilities(pSell, pFlat, pBuy);
   SignalDirection filtered_signal = raw_signal;

   if(!TrendAllows(raw_signal))
      filtered_signal = SIGNAL_FLAT;

   if(InpDebugLog && InpLog)
     {
      PrintFormat(
         "Probabilities sell=%.4f flat=%.4f buy=%.4f entry_prob=%.4f min_gap=%.4f raw_signal=%d filtered_signal=%d atr14=%.5f",
         pSell, pFlat, pBuy, InpEntryProbThreshold, InpMinProbGap, raw_signal, filtered_signal, atr14
      );
     }

   ManageExistingPosition(filtered_signal);

   long pos_type;
   double pos_price;
   if(HasOpenPosition(pos_type, pos_price))
      return;

   if(filtered_signal == SIGNAL_BUY || filtered_signal == SIGNAL_SELL)
      OpenTrade(filtered_signal, atr14);
  }
