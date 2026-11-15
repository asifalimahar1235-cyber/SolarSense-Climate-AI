# SolarSense — Local AI for Next-Day Solar Radiation Forecasting

**Turning climate data into smarter solar-energy decisions.**

SolarSense is an experimental climate-AI prototype that estimates next-day daily solar radiation for Ghotki, Sindh, Pakistan, using NASA POWER historical observations and scikit-learn.

> **Important:** SolarSense predicts solar radiation in kWh/m²/day. It does not directly predict household electricity generation, guarantee solar-panel output, or provide financial advice.

## Problem and intended users

Solar availability changes with weather and season. Households and schools in the Ghotki area may find local estimates useful when thinking about energy planning. This project investigates whether daily weather observations and seasonal features can support a next-day estimate.

## Location

- Latitude: 28.0167
- Longitude: 69.3167
- Location: Ghotki, Pakistan

## How it works

1. Fetch daily solar radiation, temperature, humidity and cloud amount from the NASA POWER API.
2. Clean missing or invalid observations.
3. Create a supervised dataset in which weather on day *t* predicts solar radiation on day *t+1*.
4. Build lag, rolling-average, weather-change and seasonal features.
5. Compare Linear Regression, Random Forest, Extra Trees and HistGradientBoosting.
6. Select the model using four-fold time-series cross-validation on training data only.
7. Evaluate once on a later chronological holdout and compare against persistence and seasonal-climatology baselines.
8. Optionally use Open-Meteo forecast weather inputs for a live demonstration.

## Data sources

- [NASA POWER](https://power.larc.nasa.gov/) — historical daily point data.
- [Open-Meteo](https://open-meteo.com/) — optional weather forecast inputs for the demonstration.
- [scikit-learn](https://scikit-learn.org/) — machine-learning tools.

Variables:
- `ALLSKY_SFC_SW_DWN`: all-sky surface shortwave downward radiation (kWh/m²/day)
- `T2M`: 2 m air temperature (°C)
- `RH2M`: relative humidity at 2 m (%)
- `CLOUD_AMT`: cloud amount (NASA POWER community-specific units)

## Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/asifalimahar1235-cyber/SolarSense-Climate-AI.git
cd SolarSense-Climate-AI
python -m venv .venv
```

Activate the environment, then install dependencies:

**Windows**
```powershell
.venv\Scripts\activate
pip install -r requirements.txt
```

**macOS/Linux**
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the dashboard

```bash
streamlit run app.py
```

The dashboard downloads NASA POWER data on first run and caches a local CSV in `data/`.

## Run tests

```bash
pytest -q
```

## Enhanced model benchmark

An additional notebook, [SolarSense_Enhanced_Benchmark.ipynb](SolarSense_Enhanced_Benchmark.ipynb), compares Linear Regression, Random Forest, Extra Trees and HistGradientBoosting. It adds short-term solar lags and rolling features, preserves missing calendar days, uses four-fold time-series cross-validation on the training period for model selection, and evaluates on a separate later 20% holdout.

[Open the enhanced benchmark in Google Colab](https://colab.research.google.com/github/asifalimahar1235-cyber/SolarSense-Climate-AI/blob/main/SolarSense_Enhanced_Benchmark.ipynb).

A short, metrics-aware recording script is available in [DEMO_SCRIPT.md](DEMO_SCRIPT.md).

**Execution status:** The enhanced benchmark notebook was executed successfully in Google Colab. Its exported comparison, prediction and summary CSV files are stored in the [`results/`](results/) folder. The measured enhanced results are reported below.

## Evaluation and results

### Enhanced benchmark (primary result)

The enhanced Google Colab benchmark used 2,557 daily NASA POWER calendar records (2019-01-01 to 2025-12-31). After creating next-calendar-day targets and removing rows without complete features, it produced 2,550 supervised observations: 2,040 training observations and 510 chronological held-out test observations. Four models were compared, with model selection based on four-fold `TimeSeriesSplit` cross-validation within the training period only. The final test period was kept separate from model selection.

| Model | Training CV MAE | Held-out MAE | Held-out RMSE | Held-out R² | MAE improvement vs persistence | MAE improvement vs seasonal |
|---|---:|---:|---:|---:|---:|---:|
| Extra Trees | 0.4546 | **0.3779** | **0.5658** | **0.8053** | **7.69%** | **29.15%** |
| Linear Regression | 0.4573 | 0.4025 | 0.5756 | 0.7985 | 1.69% | 24.54% |
| Random Forest | 0.4697 | 0.3914 | 0.5814 | 0.7944 | 4.42% | 26.63% |
| HistGradientBoosting | 0.5015 | 0.4021 | 0.5882 | 0.7896 | 1.79% | 24.62% |

The selected Extra Trees model achieved a held-out MAE of **0.3779 kWh/m²/day**, RMSE of **0.5658 kWh/m²/day**, and R² of **0.8053**. The persistence baseline MAE was **0.4094**, while the seasonal-climatology baseline MAE was **0.5334 kWh/m²/day**. Extra Trees reduced MAE by **7.69% versus persistence** and **29.15% versus seasonal climatology** on this test period. These are historical, single-location benchmark results—not proof of operational forecast reliability.

The benchmark also reports seasonal-climatology MAE **0.5334 kWh/m²/day**. The comparison and summary CSVs have been refreshed with the seasonal-baseline metrics; the prediction CSV retains the held-out actual, Extra Trees prediction, persistence prediction and error series.

### Initial baseline experiment

The original `SolarSense_Veritas_AI_Hackathon_2026.ipynb` also reported a chronological holdout result: Linear Regression MAE **0.4079**, RMSE **0.5947**, R² **0.7872**, and persistence MAE **0.4083 kWh/m²/day** (0.10% MAE improvement). This is retained as an initial experiment; the enhanced benchmark above is the primary result. The original prediction output is in [solarsense_results.csv](solarsense_results.csv).

The full project write-up is in [SUBMISSION.md](SUBMISSION.md).

**Verification note:** The Colab experiment was run. The Streamlit dashboard and automated test suite have not been independently verified as running successfully.

## Limitations

- This is an experimental educational prototype, not an operational forecasting service.
- The model uses a limited set of daily variables and a simple regression method.
- Historical test performance does not guarantee future performance.
- The optional Open-Meteo demonstration uses forecast weather inputs; it is not a verified forecast evaluation.
- Predicted solar radiation is not the same as electricity generated by a photovoltaic installation. Panel characteristics, orientation, shading, temperature and system losses also matter.

## AI assistance disclosure

AI assistance was used during development, including GitHub Copilot and ChatGPT. Generated code and project outputs must be reviewed and tested by the project author.

## Hackathon submission

See [SUBMISSION.md](SUBMISSION.md) for the project write-up, experiment results, limitations and AI-assistance disclosure.

## Competition context

Developed for the Veritas AI International High School Hackathon 2026, Climate Track.
