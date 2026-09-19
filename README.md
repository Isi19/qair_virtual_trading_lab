# Qair Virtual Trading Lab

Data-science project for renewable electricity trading. It simulates a DE-LU Qair portfolio made of Langer Wald site(onshore wind, 31 MW) and Perleberg site (solar, 22.1 MW).

The objective is to turn Day-Ahead production and price forecasts into a nomination decision, then evaluate its economic outcome after delivery.

## Research question

Before the Day-Ahead market, should the portfolio nominate P10, P50 or P90 of its forecast production for each quarter-hour?

The backtest covers July and August 2026, or 5,952 quarter-hours. It evaluates a decision made with information available on D-1; it is not a live demonstration.

## Data sources

| Data | Source | Use in the project |
|---|---|---|
| Historical weather | Open-Meteo Archive API, ERA5 | Weather histories used to simulate virtual production and train the production and price models. Hourly ERA5 values are aligned to the 15-minute delivery grid. |
| D-1 weather forecasts | Open-Meteo Single Runs API, ICON-D2 | Archived forecasts selected from the latest eligible run available before the cutoff and covering the delivery day. They reproduce the weather information available during the backtest. |
| DE-LU Day-Ahead price, load, generation and neighbouring-zone prices | SMARD | Price target, system analysis, load forecast and external price drivers. |
| D2CF system and cross-border variables | JAO Core Publication Tool | Pre-clearing vertical load, generation, net-position and capacity information. Hourly publications are repeated over their four delivery quarter-hours. |
| German imbalance price (reBAP) | Netztransparenz | Realised settlement and historical Day-Ahead/reBAP spreads used by the nomination rule. |

Langer Wald and Perleberg form a virtual portfolio. Their production is generated from reanalysis weather, asset parameters and physical wind/PV models; it is not measured Qair production. Reanalysis data is also not equivalent to an on-site weather station.

The models are trained on historical values. Archived D-1 forecasts are then used where the validation and backtest must reproduce the information that would actually have been available in operaional conditions orbefore the market cutoff.

## Decision chain

1. Production models forecast both assets at 15-minute resolution with P10, P50 and P90.
2. A price model forecasts the DE-LU Day-Ahead price with P10, P50 and P90.
3. Three portfolio nominations are compared at each quarter-hour: `Q = P10`, `Q = P50` and `Q = P90`.
4. Each nomination is evaluated across 9 scenarios: 3 production levels x 3 price levels.
5. The highest expected PnL selects the nomination. After delivery, settlement uses realised production, Day-Ahead price and reBAP.

```text
cash-flow = Q x day_ahead_price + (production - Q) x reBAP
```

### Scenario and expectation assumptions

The probabilistic forecasts are reduced to three representative quantiles: P10, P50 and P90. This gives three production scenarios and three price scenarios instead of a larger scenario distribution. The resulting 9 combinations keep the decision and its presentation simple, but do not describe the complete distribution or its extreme tails.

The marginal weights assigned to P10, P50 and P90 are respectively 0.25, 0.50 and 0.25. They are a symmetric, pragmatic choice for this demonstrator, not probabilities learned from data and not a direct probabilistic interpretation of the quantiles. Production and price scenarios are assumed independent, so the weight of a joint scenario is the product of their two weights. Modelling their dependence would require joint historical scenarios or a dependence model.

For a nomination `Q`, a production scenario `A` and a price scenario `P`, the expected score approximates:

```text
E[CF(Q)] = sum of scenario weights x [Q x P + (A - Q) x expected reBAP]
expected reBAP = P + expected signed spread
signed spread = reBAP - Day-Ahead price
```

The market information used by the bidding rule combines the probabilistic Day-Ahead price forecast with a proxy for the imbalance-price regime. This proxy is the average signed spread observed strictly before the delivery date, conditioned on the local delivery month and one of four hour blocks: 00-05, 06-11, 12-17 or 18-23. Undercovered and overcovered positions use separate spread histories. Signed spreads are not clipped, so a favourable or adverse imbalance-price regime can affect the selected nomination in either direction. No separate categorical market-state model is fitted.

This is an uncertainty-aware expected-PnL rule, but not a complete risk-adjusted optimisation: it has no explicit risk-aversion coefficient, downside penalty or CVaR constraint. It also approximates reBAP uncertainty by a conditional historical mean rather than a dedicated imbalance-price forecast. These assumptions make the result understandable and reproducible, but they must be revisited before operational use.

The dynamic decision is made only with ex-ante information. Post-settlement replays the same nominations with realised values; it is not a second optimisation.


## Backtest result

| Policy | Gross PnL | Absolute imbalance |
|---|---:|---:|
| Fixed P10 | EUR 1,728,769 | 8,339 MWh |
| Fixed P50 | EUR 1,665,901 | **6,839 MWh** |
| Fixed P90 | EUR 1,590,459 | 13,476 MWh |
| Dynamic decision | EUR 1,724,402 | 9,125 MWh |

The dynamic rule selects P10 in 85.5% of quarter-hours and P90 in 14.5%; it never selects P50. Its expected ex-ante PnL is EUR 1.845M, versus EUR 1.724M realised after settlement. P10 has the highest realised PnL on this sample, but its advantage depends strongly on favourable imbalance settlement for surplus production. P50 remains the policy with the lowest physical exposure.

PnL is gross: fees, intraday trading, contracts, curtailment and balance-responsibility costs are not included.

## Project structure

```text
data/
  asset_data/       ERA5 weather and archived D-1 forecasts
  market_15m/       prices, reBAP, regional weather, load and DE-LU JAO data
  modeling/         price-model dataset
  backtesting/      forecasts, portfolio, decisions and settlement outputs

notebooks/          validation, EDA, modeling, portfolio and nomination studies
src/
  weather/          ERA5 and D-1 forecast collectors
  generation/       virtual wind and solar production
  features/         model features
  modeling/         model training and evaluation
  pipelines/        dataset and backtest construction
  analysis/          portfolio and settlement analysis
```


## Reading the project

Open the notebooks in this order:

1. Asset validation and site EDA.
2. Production feature engineering and modeling.
3. DE-LU price EDA and modeling.
4. Portfolio exposure.
5. Nomination and settlement backtest.

The `nomination_backtest_de_lu.ipynb` notebook separates the ex-ante decision, physical imbalance, post-settlement PnL, expected-versus-realised comparison and daily risk.

## Data and limitations

The test window is short and the portfolio is virtual. A stronger evaluation would use more seasons, joint price-production scenarios and an explicit reBAP-risk forecast. The dynamic rule is not proven optimal.
## Takeaway

This project demonstrates the full chain from structured market and weather data to a trading decision:

- virtual wind and solar production are modelled at 15-minute resolution;
- Day-Ahead price and production uncertainty are converted into P10/P50/P90 nomination candidates;
- each candidate is evaluated before delivery with an explicit expected-cash-flow rule;
- the decision is replayed after delivery with realised production, prices and reBAP;
- financial performance, physical exposure and difficult delivery days are analysed together.

The result is a decision-support backtest, not a claim that one nomination rule is universally optimal. The current sample favours fixed P10 on realised Gross PnL, while P50 gives the lowest physical exposure. This distinction is central to interpreting the result for a trading desk.

## Dashboard

The Streamlit dashboard exposes the same work in four views: portfolio overview, production forecast, Day-Ahead price, and strategy/backtest. The backtest page separates the information available before delivery from the realised settlement outcome.


