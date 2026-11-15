# SolarSense — Hackathon Project Write-up

**Track:** Climate  
**Project:** SolarSense — Local AI for Next-Day Solar Radiation Forecasting  
**Location:** Ghotki, Sindh, Pakistan  
**Repository:** https://github.com/asifalimahar1235-cyber/SolarSense-Climate-AI

## Problem

Daily solar availability changes with weather and season. People exploring solar energy in smaller cities may benefit from understandable, location-specific information. SolarSense investigates whether daily weather observations can help estimate the next day's solar radiation for Ghotki.

## Our solution

SolarSense is an experimental machine-learning prototype. It retrieves historical daily observations from NASA POWER, constructs a next-calendar-day prediction task, compares four scikit-learn regression models, and presents the project in a Streamlit dashboard. The enhanced Colab benchmark selects Extra Trees using time-series cross-validation on the training period and compares it with persistence and seasonal-climatology baselines. The Streamlit dashboard presents the saved enhanced Extra Trees benchmark, baseline comparisons, held-out predictions, error analysis, and NASA POWER data explorer. It reads the experiment outputs rather than training a separate competing model.

The target is **daily solar radiation (kWh/m²/day)**, not household electricity generation.

## Data and method

- **Historical source:** NASA POWER daily point API, at 28.0167° N, 69.3167° E.
- **Variables:** all-sky surface shortwave downward radiation (`ALLSKY_SFC_SW_DWN`), 2 m air temperature (`T2M`), relative humidity (`RH2M`), and cloud amount (`CLOUD_AMT`).
- **Features:** solar-radiation lags and rolling averages, same-day temperature, relative humidity and cloud amount, selected daily changes, and sine/cosine seasonal features.
- **Prediction task:** observations on calendar day *t* are used to estimate solar radiation on calendar day *t+1*.
- **Models:** Linear Regression, Random Forest, Extra Trees, and HistGradientBoosting.
- **Model selection:** four-fold `TimeSeriesSplit` cross-validation using training-period data only.
- **Baselines:** persistence (tomorrow = today's observed solar radiation) and seasonal climatology (training-period mean target for the same day of year).
- **Evaluation:** chronological 80% training / 20% held-out test split; four-fold time-series cross-validation within the training period only; MAE, RMSE, and R² for each model. Both baselines are evaluated on the same held-out dates. No random shuffling.

## Results — enhanced Colab benchmark

The enhanced notebook was executed successfully using 2,557 NASA POWER daily calendar records from 2019-01-01 to 2025-12-31. It generated 2,550 supervised next-calendar-day observations: 2,040 training observations and 510 held-out test observations. The held-out target dates run from 2024-08-09 to 2025-12-31.

| Model | Training CV MAE | Held-out MAE | Held-out RMSE | Held-out R² | MAE improvement vs persistence | MAE improvement vs seasonal |
|---|---:|---:|---:|---:|---:|---:|
| **Extra Trees** | **0.4546** | **0.3779** | **0.5658** | **0.8053** | **7.69%** | **29.15%** |
| Linear Regression | 0.4573 | 0.4025 | 0.5756 | 0.7985 | 1.69% | 24.54% |
| Random Forest | 0.4697 | 0.3914 | 0.5814 | 0.7944 | 4.42% | 26.63% |
| HistGradientBoosting | 0.5015 | 0.4021 | 0.5882 | 0.7896 | 1.79% | 24.62% |

The Extra Trees model was selected using training-period cross-validation and achieved held-out MAE **0.3779 kWh/m²/day**, RMSE **0.5658 kWh/m²/day**, and R² **0.8053**. The persistence baseline MAE was **0.4094 kWh/m²/day**, while the seasonal-climatology baseline MAE was **0.5334 kWh/m²/day**. This is a **7.69% MAE reduction** relative to persistence and a **29.15% reduction** relative to seasonal climatology on the held-out period.

**Interpretation:** The benchmark indicates that Extra Trees performed better than both the persistence and seasonal-climatology baselines on this historical holdout. The result is promising but limited to one location and one historical test period; it does not establish that the model is reliable in live operation or in other locations. The largest-error cases are retained in the notebook for error analysis. Exported comparison, prediction and summary files are in the repository's [`results/`](results/) folder.

### Initial experiment

The earlier Linear Regression notebook reported MAE 0.4079, RMSE 0.5947, R² 0.7872, and persistence MAE 0.4083 kWh/m²/day on its own chronological holdout. It is retained as an initial baseline experiment, not the primary result.

## Limitations and responsible use

This is a small, single-location educational prototype, not a validated operational forecast. It uses a limited set of weather variables and a simple model. Missing data, cloud variability, spatial representativeness, and forecast-input quality can affect results. Historical test performance does not guarantee future accuracy. The Streamlit dashboard presents the same saved enhanced benchmark results reported above. It is an interactive research presentation, not a live operational forecasting service. Solar radiation is not equivalent to the electricity produced by a specific panel system.

## Tools and acknowledgements

NASA POWER provides the historical data; Open-Meteo provides optional forecast weather inputs; scikit-learn, pandas, Streamlit, and Matplotlib support the implementation and dashboard. AI coding assistance (GitHub Copilot and ChatGPT) was used during development, including code drafting, debugging, and documentation. The project author ran the enhanced notebook in Google Colab and reviewed the resulting outputs.

## Final submission details — complete before submitting

- **Author/team member name(s):** Asif Ali (individual project)
- **School:** IBA-Islamia Public Higher Secondary School, Ghotki, Pakistan
- **Contact email(s):** asifali.academic@gmail.com
- **GitHub repository:** https://github.com/asifalimahar1235-cyber/SolarSense-Climate-AI