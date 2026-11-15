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
- **Features:** same-day solar radiation, temperature, relative humidity, cloud amount, and sine/cosine seasonal features.
- **Prediction task:** observations on day *t* are used to estimate solar radiation on the next recorded day (*t+1*).
- **Model:** scikit-learn Linear Regression.
- **Baseline:** persistence (tomorrow = today's observed solar radiation).
- **Evaluation:** chronological 80% training / 20% held-out test split; MAE, RMSE, and R² for the model, plus MAE for the baseline. No random shuffling.

## Results — Colab experiment

The following values were reported by the project's successful Google Colab run using 2,557 NASA POWER daily records and a chronological 80% / 20% split.

| Held-out test metric | Observed result |
|---|---:|
| Number of test observations | **512** |
| Model MAE (kWh/m²/day) | **0.4079** |
| Model RMSE (kWh/m²/day) | **0.5947** |
| Model R² | **0.7872** |
| Persistence baseline MAE (kWh/m²/day) | **0.4083** |
| MAE improvement vs baseline | **0.10%** |

**Interpretation:** Linear Regression marginally outperformed the persistence baseline by 0.10% MAE on this held-out period. The difference is very small, so these results do not demonstrate a meaningful practical advantage over the simple baseline. The R² score indicates that the model explains 78.72% of the variation in the held-out target values, but this alone does not establish operational forecast reliability.

## Limitations and responsible use

This is a small, single-location educational prototype, not a validated operational forecast. It uses a limited set of weather variables and a simple model. Missing data, cloud variability, spatial representativeness, and forecast-input quality can affect results. Historical test performance does not guarantee future accuracy. The optional Open-Meteo demonstration is separate from the historical held-out evaluation and should not be described as validated forecast performance. Solar radiation is not equivalent to the electricity produced by a specific panel system.

## Tools and acknowledgements

NASA POWER provides the historical data; Open-Meteo provides optional forecast weather inputs; scikit-learn, pandas, Streamlit, and Matplotlib support the implementation and dashboard. AI coding assistance (GitHub Copilot and ChatGPT) was used during development; generated code and outputs are reviewed and tested by the project author.

## Final submission details — complete before submitting

- **Author/team member name(s):** Asif Ali (individual project)
- **School:** Islamia Public Higher Secondary School, Ghotki, Pakistan
- **Contact email(s):** asifali.academic@gmail.com
- **Demo/video link (if required):** [FILL IN]
- **GitHub repository:** https://github.com/asifalimahar1235-cyber/SolarSense-Climate-AI