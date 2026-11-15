"""SolarSense Streamlit dashboard."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.data import fetch_nasa_power, save_dataset
from src.features import FEATURES, make_supervised, chronological_split
from src.model import train_model
from src.evaluation import evaluate, error_table
from src.forecasting import get_next_day_prediction

st.set_page_config(page_title="SolarSense | Ghotki", page_icon="☀️", layout="wide")

st.title("☀️ SolarSense")
st.subheader("Local AI for Next-Day Solar Radiation Forecasting")
st.caption("Climate Track · Ghotki, Sindh, Pakistan · Experimental educational prototype")

st.info(
    "SolarSense predicts daily solar radiation (kWh/m²/day). It does not directly "
    "predict household electricity generation or guarantee real-world output."
)
st.warning(
    "Dashboard model note: this app trains a separate Linear Regression model. "
    "Its metrics are not the enhanced Extra Trees benchmark reported in the project write-up. "
    "The enhanced Colab notebook is the primary competition evaluation."
)

@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    path = Path("data/ghotki_nasa_power.csv")
    if path.exists():
        try:
            cached = pd.read_csv(path, parse_dates=["date"], index_col="date")
            if not cached.empty:
                return cached
        except Exception:
            pass
    frame = fetch_nasa_power()
    save_dataset(frame)
    return frame

try:
    with st.spinner("Loading NASA POWER daily climate observations..."):
        climate = load_data()
    supervised = make_supervised(climate)
    train, test = chronological_split(supervised)
    model = train_model(train)
    predictions = model.predict(test[FEATURES])
    metrics = evaluate(test, predictions)
    failures = error_table(test, predictions)

    st.success(
        f"Loaded {len(climate):,} calendar days, including "
        f"{len(climate.dropna()):,} complete daily observations, "
        f"from {climate.index.min().date()} to {climate.index.max().date()}."
    )

    tab_overview, tab_data, tab_model, tab_errors, tab_forecast, tab_about = st.tabs(
        ["Overview", "Climate data", "Model performance", "Error analysis", "Next-day demo", "About"]
    )

    with tab_overview:
        st.markdown(
            "### The problem\n"
            "Solar availability varies with weather and season. SolarSense explores whether "
            "locally available climate observations can support a next-day solar-radiation estimate.\n\n"
            "### Intended users\n"
            "The prototype is designed for households and schools interested in understanding "
            "expected solar availability in the Ghotki area."
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Location", "Ghotki, Pakistan")
        c2.metric("Daily observations", f"{len(climate):,}")
        c3.metric("Test observations", f"{len(test):,}")

    with tab_data:
        st.markdown("### Historical solar radiation")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(climate.index, climate["solar"], linewidth=0.7)
        ax.set_ylabel("kWh/m²/day")
        ax.set_xlabel("Date")
        ax.grid(alpha=0.25)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("### Weather variables")
        st.line_chart(climate[["temperature", "humidity", "cloud"]])
        st.caption(
            "NASA POWER variables: ALLSKY_SFC_SW_DWN (solar radiation), "
            "T2M (2 m air temperature), RH2M (relative humidity), CLOUD_AMT (cloud amount)."
        )
        st.dataframe(climate.tail(10), use_container_width=True)

    with tab_model:
        st.markdown("### Dashboard Linear Regression — separate chronological holdout")
        m1, m2, m3 = st.columns(3)
        m1.metric("Model MAE", f'{metrics["model_mae"]:.3f} kWh/m²/day')
        m2.metric("Model RMSE", f'{metrics["model_rmse"]:.3f} kWh/m²/day')
        m3.metric("Model R²", f'{metrics["model_r2"]:.3f}')
        st.metric("Persistence baseline MAE", f'{metrics["baseline_mae"]:.3f} kWh/m²/day')
        st.metric("MAE improvement vs baseline", f'{metrics["mae_improvement_percent"]:.2f}%')
        if metrics["model_mae"] < metrics["baseline_mae"]:
            st.success("The model achieved a lower MAE than the persistence baseline on this test set.")
        else:
            st.warning("The model did not beat the persistence baseline on this test set.")

        plot_frame = pd.DataFrame(
            {"Actual": test["target"].to_numpy(),
             "SolarSense": predictions,
             "Persistence baseline": test["baseline"].to_numpy()},
            index=test["target_date"]
        )
        st.line_chart(plot_frame.tail(120))
        st.caption(
            "The final 20% of this dashboard run’s supervised observations are held out in chronological order. "
            "These dynamically refreshed dashboard metrics are separate from the fixed enhanced Colab benchmark."
        )

    with tab_errors:
        st.markdown("### Five largest absolute prediction errors")
        st.dataframe(failures.head(5), use_container_width=True)
        residuals = test["target"].to_numpy() - predictions
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.hist(residuals, bins=30)
        ax.set_xlabel("Actual minus predicted (kWh/m²/day)")
        ax.set_ylabel("Frequency")
        ax.set_title("Test residual distribution")
        ax.grid(alpha=0.2)
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            "Large errors may occur during unusual weather, rapid cloud changes, data-quality "
            "issues, or because the simple model cannot represent complex atmospheric processes."
        )

    with tab_forecast:
        st.markdown("### Optional next-day demonstration")
        st.write(
            "Open-Meteo forecast weather inputs are passed into the dashboard’s separate Linear Regression "
            "model trained on NASA POWER historical observations. This is an experimental demonstration, "
            "not the enhanced Extra Trees model and not a verified operational forecast."
        )
        if st.button("Generate next-day estimate", type="primary"):
            try:
                with st.spinner("Fetching Open-Meteo forecast inputs..."):
                    live = get_next_day_prediction(model)
                st.metric(
                    f'Predicted solar radiation for {live["target_date"]}',
                    f'{live["predicted_solar"]:.3f} kWh/m²/day'
                )
                st.caption(f'Weather input date: {live["input_date"]}')
            except Exception as exc:
                st.error(f"Could not generate the live demonstration: {exc}")

    with tab_about:
        st.markdown(
            "### Methodology\n"
            "- Historical observations: NASA POWER daily point API.\n"
            "- Model: scikit-learn Linear Regression.\n"
            "- Features: temperature, humidity, cloud amount and target-day seasonal terms.\n"
            "- Baseline: tomorrow equals today's observed solar radiation.\n"
            "- Split: chronological 80% training / 20% test.\n\n"
            "### Limitations\n"
            "- This dashboard uses a separate Linear Regression baseline implementation; it is not the enhanced Extra Trees benchmark.\n"
            "- The model uses a small set of daily weather variables.\n"
            "- Forecast-input quality and historical-data consistency affect results.\n"
            "- Predicted solar radiation is not the same as panel electricity output.\n"
            "- Performance on this historical test period does not guarantee future accuracy.\n\n"
            "### Sources\n"
            "- [NASA POWER](https://power.larc.nasa.gov/)\n"
            "- [Open-Meteo API](https://open-meteo.com/)\n"
            "- [scikit-learn](https://scikit-learn.org/)\n\n"
            "### AI assistance disclosure\n"
            "AI assistance was used during development, including GitHub Copilot and ChatGPT. "
            "Generated code and outputs must be reviewed and tested by the project author."
        )

except Exception as exc:
    st.error(f"SolarSense could not start: {exc}")
    st.info("Check the internet connection and NASA POWER API availability, then try again.")
