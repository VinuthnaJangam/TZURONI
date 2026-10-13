# Walk-forward model evaluation

## Selected strategy per held-out event

| Metric | Value |
|---|---:|
| Out-of-sample event decisions | 137 |
| Expanding walk-forward folds | 5 |
| Sharpe ratio (252 event periods/year) | 0.125 |
| Sortino ratio (252 event periods/year) | 0.101 |
| Maximum drawdown | -6.49% |

## Strategy performance on out-of-sample folds

| Strategy | Events scored | Sharpe | Sortino | Max drawdown | Mean P&L |
|---|---:|---:|---:|---:|---:|
| fade_z0 | 137 | -0.476 | -0.372 | -10.57% | -0.01672% |
| fade_z0.5 | 137 | -0.563 | -0.339 | -7.13% | -0.01568% |
| fade_z1 | 137 | 0.458 | 0.224 | -2.61% | 0.00935% |
| follow_z0 | 137 | 0.081 | 0.072 | -9.12% | 0.00285% |
| follow_z0.5 | 137 | 0.227 | 0.176 | -4.54% | 0.00633% |
| follow_z1 | 137 | -0.715 | -0.375 | -3.54% | -0.01461% |

## Method and interpretation

The target is the forward close-to-close market return multiplied by the
strategy's surprise-following or surprise-fading side, less configured costs.
At each out-of-sample event, the fitted model selects the candidate strategy
with the highest predicted P&L. All candidates for a given event remain in the
same time split. Features use the event's actual/forecast release values,
past-only surprise history, scheduled time, and news published before the
event. News sentiment is a simple lexical measure.

The Sharpe and Sortino ratios annualize event-level observations using 252
periods per year; macro events are not evenly spaced daily returns, so these
ratios are not directly comparable to daily portfolio metrics. Maximum
drawdown is calculated from compounded selected event returns. Results are
conditional on the supplied instrument, date range, calendar actor output,
price alignment, costs, and incomplete data handling. They are historical
estimates, not evidence of future profitability. No result should be
interpreted without inspecting the source data and generated plots.
