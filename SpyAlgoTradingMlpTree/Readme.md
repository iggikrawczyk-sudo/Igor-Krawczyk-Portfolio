This project was initially intended to test whether weather data and other non-financial data have an impact on different assets.
However, as I improved the models, they all started to outperform the market. What really swayed me from my original idea was that 
I tested all of the models using the same configurations for the horizons (`signal_horizon` — how far into the future we predict the change in the market, and `pnl_horizon` — how long we hold the position), 
availability to short, and certainty threshold. This was not a fair comparison.
Whole config:
TRAIN_START = "2000-01-01"
TRAIN_END   = "2018-12-31"
TEST_START  = "2019-01-01"
TEST_END    = "2026-07-30"

TICKER = "SPY"
WEATHER_LAT = 40.7128   # NYC -- proxy for "NYSE weather"
WEATHER_LON = -74.0060

THRESHOLDS = [
    0.0000,
    0.0005,
    0.0010,
    0.0015,
    0.0020,
    0.0025,
    0.0030,
    0.0040,
    0.0050,
]
ALLOW_SHORT_OPTIONS = [False, True]
COST_BPS          = 0.6
SNAPSHOT_SLIP_BPS = 0.0     # extra per-side cost vs official close (near-close fill)

RETRAIN   = "frozen"        # "frozen" or "rolling"
ROLL_FREQ = "MS"
ANN       = 252
Therefore, I changed my original idea to focus on building the best models possible and running 28,800 simulations to find the best configuration for each model. (see grid_heatmaps.png)


----------------------------------------------------------------------------------------------------
     model    features signal_horizon pnl_horizon  threshold allow_short 
      Tree fin+weather             13          18     0.0000        True   
      Tree    fin-only              4           8     0.0020        True      
       MLP fin+weather              5           4     0.0005       False      
       MLP    fin-only             14          12     0.0000       False       
 
     
The way I determined which model is the best is by using my "Score," which is calculated as:
**Score = Total Return × (0.3372 / Max Drawdown)**
0.3372 is the maximum drawdown of SPY over that period. 
I decided that maximum drawdown should be the deciding factor for or against using leverage, and that SPY's maximum drawdown should be the maximum drawdown to accept.
At first, the Score formula also included exposure, since higher exposure and leverage result in higher fees. However, this produced worse results with very few trades 
like 14 in 6.5 years which I deemed more luck than real results



Here is my results **costs included** (MLP will vary because its starts weightening from seed which will differ so the end results will sliglty vary when you run this code)

| Model | Features | Score | Total Return | Max Drawdown | Exposure | Sharpe | Sortino | Hit Rate | Trades | Signal Horizon | PnL Horizon | Threshold | Allow Short |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| Tree | Fin + Weather | 8.716 | 394.42% | -15.26% | 94.69% | 1.21 | 1.72 | 51.47% | 101 | 13 | 18 | 0.0000 | Yes |
| Tree | Fin-only | 7.837 | 338.26% | -14.55% | 42.88% | 1.28 | 1.28 | 51.36% | 102 | 4 | 8 | 0.0020 | Yes |
| MLP | Fin + Weather | 4.603 | 305.08% | -22.35% | 61.80% | 1.24 | 1.33 | 46.39% | 294 | 5 | 4 | 0.0005 | No |
| MLP | Fin-only | 3.098 | 242.60% | -26.41% | 86.02% | 1.01 | 1.23 | 51.72% | 137 | 14 | 12 | 0.0000 | No |
| Buy & Hold | — | — | 225.93% | -33.72% | 100.00% | 0.90 | 1.11 | 55.63% | — | — | — | — | — |

So if we use leverage on the best model fin-only tree (I dediced to choose that one even tho it is not the best score nor return because of exposure and that pnl 
horizon is below 10 which changes the bps for this one the costs are quite well approximated for fin+weather with pnl horizon >10 the costs would be higher) 
from my resoning we would leverage -33.72/14.55 = 2.3175257732
So the Return would be 338.26 * 2.3175257732 = 783.926268041% which is around 38.5% annualized for the 6.5 years of tests
