# Sri Lanka Weather Prediction

A machine learning project that forecasts **temperature** and **rainfall** for Ratnapura, Sri Lanka using historical daily weather data.

---

## What This Project Does

- Visualises weather station locations across Sri Lanka on an interactive map
- Explores monthly rainfall and temperature patterns for **Ratnapura**
- Forecasts **daily temperature** using Facebook Prophet
- Forecasts **daily rainfall** using LightGBM with lag features

---

## Dataset

- **Source:** Sri Lanka Weather Dataset (Kaggle)
- **Coverage:** 2010 – 2023, multiple cities
- **Main columns:** `temperature_2m_mean`, `rain_sum`, `weathercode`, `windspeed_10m_max`

---

## Data Exploration (Ratnapura)

Ratnapura is one of the wettest cities in Sri Lanka, located in the wet zone lowlands (~30m elevation). The exploration includes:

- **Monthly rainfall bar chart** — comparing 2019 to 2022
- **Temperature heatmap** — monthly mean temperature across all years

---

## Models

### Temperature → Facebook Prophet
Trained on 2020–2022 data. Prophet automatically learns the trend and seasonal pattern.

Key settings:
- `yearly_seasonality = 10` — captures Sri Lanka's double-peak cycle (April + August)
- `changepoint_prior_scale = 0.01` — smoother trend to avoid overfitting
- Forecasts 100 days ahead

### Rainfall → LightGBM
Trained on 2020–2021, tested on 2022. Uses lag features to capture serial dependence in rainfall.

Features used:

| Feature | Description |
|---|---|
| `lag_1` | Yesterday's rainfall |
| `lag_7` | Rainfall 7 days ago |
| `lag_14` | Rainfall 14 days ago |
| `lag_30` | Rainfall 30 days ago |
| `rolling_7` | 7-day rolling mean |
| `rolling_30` | 30-day rolling mean |
| `day_of_year` | Captures monsoon seasonality |

---

## Key Findings

- `lag_1` (yesterday's rainfall) is the strongest predictor of today's rainfall
- `day_of_year` captures monsoon seasonality better than explicit month or monsoon category features
- Prophet successfully captured Ratnapura's double-peak temperature pattern
- Ratnapura mean temperature is ~25°C, consistent with its wet-zone lowland location

---

## Setup

```bash
pip install pandas numpy matplotlib seaborn folium prophet lightgbm scikit-learn
```

Then run the notebook top to bottom.

---

## Notebook

View the full notebook on Kaggle: [weather-prediction-sl](https://www.kaggle.com/code/klndevinda/weather-prediction-sl)

---

## Acknowledgements

Weather data via [Open-Meteo API](https://open-meteo.com/) on Kaggle.
