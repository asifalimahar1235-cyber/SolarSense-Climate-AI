# SolarSense — 90-second demo video script

**Recording target:** 75–100 seconds. Screen-record the enhanced Colab notebook or dashboard while reading this script. Replace the bracketed result fields only after the enhanced benchmark has completed successfully.

## Voiceover

Hello, I'm Asif, and this is SolarSense, a climate-AI prototype built for Ghotki, Sindh, Pakistan.

Solar energy availability changes from day to day. My project explores whether public climate observations can help estimate the next day's solar radiation for a specific local area. The output is daily solar radiation in kilowatt-hours per square metre, not a guarantee of how much electricity a solar panel will generate.

SolarSense uses daily NASA POWER observations for solar radiation, temperature, humidity and cloud amount. It creates recent-history and seasonal features, then compares Linear Regression, Random Forest, Extra Trees and Histogram Gradient Boosting.

To avoid a random split that mixes dates, I use time-series cross-validation on the earlier training period. The latest 20 percent of dates are kept separate for final testing. I also compare the selected model with a persistence baseline: tomorrow is estimated as today's solar radiation.

The selected model was **[MODEL NAME]**. On the held-out test period, its MAE was **[MODEL MAE]**, compared with **[BASELINE MAE]** for persistence. That is a **[IMPROVEMENT]%** change in MAE. These are measured results from this run, not a claim of perfect forecasting.

SolarSense is an educational prototype for one location. It still needs validation against local ground-measured solar data and testing in other locations before it could support real operational decisions.

Thank you for watching.

## What to show on screen

1. Project title and Ghotki location.
2. NASA POWER data download and observation count.
3. Model comparison table and time-series split.
4. Actual-versus-predicted chart and persistence comparison.
5. Limitations section.

## Recording checklist

- Keep the video under 2 minutes.
- Make the YouTube video unlisted, or share a Google Drive video with “anyone with the link can view”.
- Test the link in a private/incognito window.
- Do not read out unverified or placeholder metrics.
