# SolarSense — Hackathon Project Write-up

**Track:** Climate  
**Project:** SolarSense — Local AI for Next-Day Solar Radiation Forecasting  
**Location:** Ghotki, Sindh, Pakistan  
**Repository:** https://github.com/asifalimahar1235-cyber/SolarSense-Climate-AI

## Problem

Daily solar availability changes with weather and season. People exploring solar energy in smaller cities may benefit from understandable, location-specific information. SolarSense investigates whether daily weather observations can help estimate the next day's solar radiation for Ghotki.

## Our solution

SolarSense is an experimental machine-learning prototype. It retrieves historical daily observations from NASA POWER, constructs a next-calendar-day prediction task, trains a scikit-learn Linear Regression pipeline, and presents its evaluation in a Streamlit dashboard. It compares the model with a persistence baseline that predicts tomorrow's solar radiation using today's observed value.

The target is **daily solar radiation (kWh/m²/day)**, not household electricity generation.

## Data and method

- **Historical source:** NASA POWER daily point API, at 28.0167° N, 69.3167° E.
- **Variables:** all-sky surface shortwave downward radiation (`ALLSKY_SFC_SW_DWN`), 2 m air temperature (`T2M`), relative humidity (`RH2M`), and cloud amount (`CLOUD_AMT`).
- **Features:** daily temperature, humidity, cloud amount, and sine/cosine seasonal features for the target date.
- **Prediction task:** weather on day *t* predicts solar radiation on day *t+1*.
- **Model:** median imputation, standardization, and Linear Regression.
- **Baseline:** persistence (tomorrow = today's observed solar radiation).
- **Evaluation:** chronological 80% training / 20% held-out test split; MAE, RMSE, and R² for the model, plus MAE for the baseline. No random shuffling.

## Results — complete after running the project

Do not submit estimated or invented values. Copy the actual metrics shown in the dashboard after a successful run.

| Held-out test metric | Actual result |
|---|---:|
| Number of test observations | **[FILL IN]** |
| Model MAE (kWh/m²/day) | **[FILL IN]** |
| Model RMSE (kWh/m²/day) | **[FILL IN]** |
| Model R² | **[FILL IN]** |
| Persistence baseline MAE (kWh/m²/day) | **[FILL IN]** |
| MAE improvement vs baseline (%) | **[FILL IN]** |

**Interpretation:** State honestly whether SolarSense beats the persistence baseline on the held-out test period. Include one example or pattern from the largest-error table.

## Limitations and responsible use

This is a small, single-location educational prototype, not a validated operational forecast. It uses a limited set of weather variables and a simple model. Missing data, cloud variability, spatial representativeness, and forecast-input quality can affect results. Historical test performance does not guarantee future accuracy. The optional Open-Meteo demonstration is separate from the historical held-out evaluation and should not be described as validated forecast performance. Solar radiation is not equivalent to the electricity produced by a specific panel system.

## Tools and acknowledgements

NASA POWER provides the historical data; Open-Meteo provides optional forecast weather inputs; scikit-learn, pandas, Streamlit, and Matplotlib support the implementation and dashboard. AI coding assistance (GitHub Copilot and ChatGPT) was used during development; generated code and outputs are reviewed and tested by the project author.

## Final submission details — complete before submitting

- **Author/team member name(s):** [FILL IN]
- **School:** [FILL IN]
- **Contact email(s):** [FILL IN]
- **Demo/video link (if required):** [FILL IN]
- **GitHub repository:** https://github.com/asifalimahar1235-cyber/SolarSense-Climate-AI