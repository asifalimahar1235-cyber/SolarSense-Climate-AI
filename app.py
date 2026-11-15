"""SolarSense research dashboard: transparent view of the published benchmark."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.data import fetch_nasa_power, save_dataset

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DATA_PATH = ROOT / "data" / "ghotki_nasa_power.csv"

st.set_page_config(
    page_title="SolarSense | Ghotki Climate Intelligence",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ink:#102c36; --muted:#617982; --teal:#087e78; --line:#dce9e6; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
[data-testid="stAppViewContainer"] {
 background: radial-gradient(ellipse at 7% 0%,rgba(8,126,120,.10),transparent 34%),
 linear-gradient(180deg,#f7fbfa 0%,#f2f7f6 100%);
}
[data-testid="stHeader"] { background:rgba(247,251,250,.82); }
.block-container { max-width:1240px; padding-top:1.7rem; padding-bottom:3rem; }
h1,h2,h3 { color:var(--ink); font-family:'Manrope',sans-serif; letter-spacing:-.035em; }
h1 { font-weight:800; }
[data-testid="stMetric"] { background:rgba(255,255,255,.92); border:1px solid var(--line);
 padding:1rem 1.1rem; border-radius:16px; box-shadow:0 5px 18px rgba(16,43,53,.045); }
[data-testid="stMetricLabel"] { color:var(--muted); font-weight:600; }
[data-testid="stMetricValue"] { color:var(--ink); }
.stTabs [data-baseweb="tab-list"] { gap:.35rem; }
.stTabs [data-baseweb="tab"] { border-radius:10px 10px 0 0; padding:.65rem .9rem; }
div[data-testid="stAlert"] { border-radius:12px; }
div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
.ss-hero { background:linear-gradient(115deg,#103c46 0%,#087e78 70%,#20a18f 100%);
 padding:2rem 2rem; border-radius:22px; color:white; margin:.65rem 0 1.35rem;
 box-shadow:0 14px 34px rgba(8,80,78,.16); }
.ss-hero h2 { color:white; margin:0 0 .5rem; font-size:1.8rem; font-weight:800; }
.ss-hero p { color:rgba(255,255,255,.9); margin:0; font-size:1rem; max-width:760px; }
.ss-kicker { color:#bce9df; text-transform:uppercase; letter-spacing:.15em;
 font-size:.72rem; font-weight:800; margin-bottom:.65rem; }
.ss-tag { display:inline-block; border:1px solid rgba(255,255,255,.35); color:white;
 border-radius:999px; padding:.28rem .65rem; margin-top:1rem; font-size:.75rem; }
.ss-section { color:var(--teal); text-transform:uppercase; font-size:.72rem;
 font-weight:800; letter-spacing:.14em; margin-bottom:.35rem; }
.ss-card { background:rgba(255,255,255,.82); border:1px solid var(--line);
 border-radius:15px; padding:1rem 1.1rem; height:100%; }
.ss-card h4 { color:var(--ink); margin:0 0 .4rem; font-family:'Manrope',sans-serif; }
.ss-card p { color:var(--muted); margin:0; font-size:.91rem; }
.ss-foot { border-top:1px solid var(--line); padding-top:1rem; margin-top:2rem;
 color:#6a8087; font-size:.82rem; }
</style>
<div class="ss-hero">
 <div class="ss-kicker">Climate intelligence · Ghotki, Pakistan</div>
 <h2>Local sunlight. Better-informed decisions.</h2>
 <p>SolarSense is an evidence-led student research prototype exploring next-day solar-radiation prediction from public climate observations.</p>
 <span class="ss-tag">RESEARCH PROTOTYPE · CLIMATE AI</span>
</div>
""", unsafe_allow_html=True)

st.title("☀️ SolarSense")
st.caption("Next-day daily solar radiation research · Ghotki, Sindh, Pakistan")
st.info(
    "**What this predicts:** daily solar radiation in kWh/m²/day. "
    "It does not predict a household's electricity bill or guarantee solar-panel output."
)

@st.cache_data(ttl=3600, show_spinner=False)
def load_climate() -> pd.DataFrame:
    if DATA_PATH.exists():
        try:
            cached = pd.read_csv(DATA_PATH, parse_dates=["date"], index_col="date")
            if not cached.empty:
                return cached
        except Exception:
            pass
    frame = fetch_nasa_power(start="20190101", end="20251231")
    save_dataset(frame, str(DATA_PATH))
    return frame

@st.cache_data
def load_benchmark():
    comparison = pd.read_csv(RESULTS / "solarsense_model_comparison.csv")
    summary = pd.read_csv(RESULTS / "solarsense_enhanced_summary.csv").iloc[0]
    predictions = pd.read_csv(RESULTS / "solarsense_enhanced_predictions.csv", parse_dates=["input_date", "target_date"])
    return comparison, summary, predictions

try:
    comparison, summary, predictions = load_benchmark()
    climate = load_climate()

    st.success(
        f"Research results loaded · {int(summary['test_observations']):,} held-out test days · "
        f"{predictions['target_date'].min().date()} to {predictions['target_date'].max().date()}"
    )

    overview, benchmark, errors, data_tab, methodology = st.tabs(
        ["Overview", "Model benchmark", "Error analysis", "Climate explorer", "Research notes"]
    )

    with overview:
        st.markdown('<div class="ss-section">The research result</div>', unsafe_allow_html=True)
        st.markdown("### A model tested against simple alternatives")
        st.write(
            "The enhanced experiment compares four machine-learning approaches and checks whether "
            "the selected model improves on two transparent baselines. The final test period is later "
            "than the training period and was not used to choose the model."
        )
        a, b, c, d = st.columns(4)
        a.metric("Selected model", str(summary["selected_model"]))
        b.metric("Test MAE", f"{summary['test_mae']:.4f}", help="Lower is better; kWh/m²/day")
        c.metric("Test R²", f"{summary['test_r2']:.4f}", help="Closer to 1 is better")
        d.metric("Held-out days", f"{int(summary['test_observations']):,}")
        st.markdown("### At a glance")
        left, right = st.columns(2)
        with left:
            st.markdown('<div class="ss-card"><h4>What makes the test meaningful?</h4><p>Dates stay in chronological order. Model selection uses training-only time-series cross-validation. The later test dates are reserved for the final comparison.</p></div>', unsafe_allow_html=True)
        with right:
            st.markdown('<div class="ss-card"><h4>What the result does not prove</h4><p>This is one location and one historical test period. It is not proof of reliable live forecasts or performance in other cities.</p></div>', unsafe_allow_html=True)
        st.markdown("### Held-out predictions")
        plot = predictions.set_index("target_date")[["actual", "predicted", "persistence"]].rename(
            columns={"actual":"Observed radiation", "predicted":"Extra Trees estimate", "persistence":"Persistence baseline"}
        )
        st.line_chart(plot, height=340)
        st.caption("All three lines use the same 510 held-out target dates. Values are daily solar radiation, not electricity output.")

    with benchmark:
        st.markdown('<div class="ss-section">Fair model comparison</div>', unsafe_allow_html=True)
        st.markdown("### Four models, one untouched test period")
        st.dataframe(
            comparison.rename(columns={
                "model":"Model",
                "training_time_series_cv_mae":"Training CV MAE",
                "test_mae":"Held-out MAE",
                "test_rmse":"Held-out RMSE",
                "test_r2":"Held-out R²",
                "mae_improvement_vs_persistence_percent":"MAE gain vs persistence (%)",
                "mae_improvement_vs_seasonal_percent":"MAE gain vs seasonal (%)",
            }).round(4),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("### Baseline comparison")
        x, y, z = st.columns(3)
        x.metric("Extra Trees MAE", f"{summary['test_mae']:.4f}")
        y.metric("Persistence MAE", f"{summary['persistence_mae']:.4f}")
        z.metric("Seasonal baseline MAE", f"{summary['seasonal_climatology_mae']:.4f}")
        st.caption("MAE is mean absolute error in kWh/m²/day. Lower values indicate smaller average prediction errors.")
        chart = comparison.set_index("model")[["test_mae"]].sort_values("test_mae")
        st.bar_chart(chart, height=300)
        st.success(
            f"On this historical holdout, Extra Trees reduced MAE by "
            f"{summary['mae_improvement_vs_persistence_percent']:.2f}% versus persistence "
            f"and {summary['mae_improvement_vs_seasonal_percent']:.2f}% versus seasonal climatology."
        )

    with errors:
        st.markdown('<div class="ss-section">Honest error analysis</div>', unsafe_allow_html=True)
        st.markdown("### The difficult days matter too")
        worst = predictions.nlargest(10, "absolute_error").copy()
        st.dataframe(worst.round(4), use_container_width=True, hide_index=True)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(predictions["actual"] - predictions["predicted"], bins=28)
        ax.set_title("Held-out prediction error distribution")
        ax.set_xlabel("Observed minus estimated (kWh/m²/day)")
        ax.set_ylabel("Number of days")
        ax.grid(alpha=.2)
        st.pyplot(fig)
        plt.close(fig)
        st.warning(
            "Large errors are part of the result, not something to hide. Rapid weather changes, "
            "cloud variability, data limitations and the model's simplified inputs may contribute."
        )

    with data_tab:
        st.markdown('<div class="ss-section">Explore the source data</div>', unsafe_allow_html=True)
        st.markdown("### NASA POWER daily observations")
        st.write("The historical data used by the project comes from NASA POWER's public daily point API.")
        solar = climate[["solar"]].dropna()
        st.line_chart(solar, height=330)
        st.caption("ALLSKY_SFC_SW_DWN · Daily all-sky surface shortwave downward radiation (kWh/m²/day)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Calendar days", f"{len(climate):,}")
        c2.metric("Complete daily rows", f"{len(climate.dropna()):,}")
        c3.metric("Data start", str(climate.index.min().date()))
        st.markdown("### Latest rows in the historical dataset")
        st.dataframe(climate.tail(12), use_container_width=True)

    with methodology:
        st.markdown('<div class="ss-section">Reproducibility and responsible use</div>', unsafe_allow_html=True)
        st.markdown("### How the experiment works")
        st.markdown("""
1. Download daily solar radiation, temperature, humidity and cloud observations from NASA POWER.
2. Keep the calendar complete so missing dates cannot become false next-day examples.
3. Use information available on day *t* to estimate solar radiation on day *t+1*.
4. Create solar lags, rolling averages, weather-change and seasonal features.
5. Compare Linear Regression, Random Forest, Extra Trees and HistGradientBoosting.
6. Select using four-fold time-series cross-validation on the training period only.
7. Evaluate once on the later chronological holdout and compare with persistence and seasonal-climatology baselines.
        """)
        st.markdown("### Study boundaries")
        st.markdown("""
- One geographic point near Ghotki, Pakistan.
- Historical period: 2019-01-01 to 2025-12-31.
- 2,550 supervised observations: 2,040 training and 510 held-out test observations.
- The model predicts solar radiation, not photovoltaic electricity generation.
- Historical accuracy does not guarantee future accuracy.
- The project is an educational research prototype, not an operational energy service.
        """)
        st.markdown("### Data and software")
        st.markdown("""
- [NASA POWER](https://power.larc.nasa.gov/) — historical climate observations.
- [scikit-learn](https://scikit-learn.org/) — machine-learning models and evaluation.
- [pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/) — data processing.
- [Streamlit](https://streamlit.io/) — interactive research dashboard.
        """)
        st.markdown("### AI assistance disclosure")
        st.write(
            "GitHub Copilot and ChatGPT were used during development for code drafting, debugging "
            "and documentation. The project author ran the enhanced notebook in Google Colab and "
            "reviewed the reported experiment outputs."
        )

    st.markdown(
        '<div class="ss-foot">SOLARSENSE · IBA-Islamia Public Higher Secondary School, Ghotki, Pakistan · '
        'Student research prototype · Historical results, not an operational forecast.</div>',
        unsafe_allow_html=True,
    )

except FileNotFoundError as exc:
    st.error(f"Project result file not found: {exc}")
    st.info("Please use the complete GitHub repository so the benchmark result files are available.")
except Exception as exc:
    st.error(f"SolarSense could not load the dashboard: {exc}")
    st.info("Check the internet connection and NASA POWER availability, then refresh.")
