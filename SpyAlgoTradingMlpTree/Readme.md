This project was initially intended to test whether weather data and other non-financial data have an impact on different assets.
However, as I improved the models, they all started to outperform the market. What really swayed me from my original idea was that 
I tested all of the models using the same configurations for the horizons (`signal_horizon` — how far into the future we predict the change in the market, and `pnl_horizon` — how long we hold the position), 
availability to short, and certainty threshold. This was not a fair comparison.
Therefore, I changed my original idea to focus on building the best models possible and running 28,800 simulations to find the best configuration for each model.


      model    features signal_horizon pnl_horizon  threshold allow_short 
      Tree    fin-only             12          17     0.0040        True      
      Tree fin+weather             12          17     0.0050        True      
       MLP fin+weather              5           4     0.0005       False      
       MLP    fin-only             14          12     0.0000       False       
     
The way I determined which model is the best is by using my "Score," which is calculated as:
**Score = Total Return × (0.3372 / Max Drawdown)**
0.3372 is the maximum drawdown of SPY over that period. 
I decided that maximum drawdown should be the deciding factor for or against using leverage, and that SPY's maximum drawdown should be the maximum drawdown to accept.
At first, the Score formula also included exposure, since higher exposure and leverage result in higher fees. However, this produced worse results, and the Tree models, 
which would benefit the most from leverage in my case, have low exposure. As a result, their scores are not particularly high, especially for the Tree models that would benefit the most from leverage.



Here is my results MLP will vary because its starts weightening from seed which will differ so the end results will sliglty vary when you run this code

model    features signal_horizon pnl_horizon  threshold allow_short Total Return Max Drawdown Exposure  Score  Sharpe  Sortino Hit Rate Trades
Tree    fin-only             12          17     0.0040        True      134.19%      -10.43%   12.51% 69.398    0.92     0.52   55.56%     14
Tree fin+weather             12          17     0.0050        True      133.46%      -10.43%   12.51% 69.021    0.92     0.52   55.56%     14
MLP fin+weather              5           4     0.0005       False      305.09%      -22.35%    61.8% 14.897    1.24     1.33   46.39%    294
MLP    fin-only             14          12     0.0000       False       242.6%      -26.41%   86.02%  7.203    1.01     1.23   51.72%    137
Buy & Hold                 NaN         NaN       NaN        NaN         225.93%      -33.72%   100.0%   NaN    0.90     1.11   55.63%      

So if we use leverage on the best model fin-only tree from my resoning we would leverage -33.72/10.43 = 3.23298178332
So the Return would be 134.19 * 3.23298178332 = 433.833825503% which is around 25.3% annualized
