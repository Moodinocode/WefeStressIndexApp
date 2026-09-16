# Litani WEFE Stress Index

A proposed method for measuring Water–Energy–Food–Ecosystem (WEFE) stress in Lebanon's Litani River basin from satellite data, with an interactive dashboard for exploring it.

University course project, accompanied by a co-authored research paper (`Research Paper Final Version.docx`).

## The problem

The WEFE nexus is a well-established framework for thinking about how water, energy, food, and ecosystem pressures compound one another. What is *not* established is how to actually compute it — there is no agreed formula, and existing work measures it in ways that differ substantially and often depend on ground data that Lebanon does not reliably collect.

This project proposes a calculation method built entirely from **remote sensing**, so it can be run for the Litani basin without needing ground instrumentation.

## The four pillars

Each pillar of the nexus is represented by a satellite-derived indicator, all pulled through **Google Earth Engine**:

| Pillar | Indicator | What it captures |
|---|---|---|
| Water | CHIRPS rainfall | Precipitation anomalies and drought onset |
| Energy | VIIRS night-time radiance | Electrical supply shortfalls, as a proxy for grid stress |
| Food | NDVI | Vegetation health and irrigation pressure on farmland |
| Ecosystem | NDTI | Water turbidity and ecological degradation |

These are combined into a single **WEFE stress index** scaled 0–1, with higher values meaning more compounded stress, and banded into Green / Amber / Red categories.

## The dataset

`Litani_WEFE_Stressed_Data.csv` — **4,301 daily observations, 1 January 2014 to 10 October 2025**.

| Column | Range in this dataset |
|---|---|
| `ndvi` | -0.017 – 0.549 |
| `ndti` | -0.204 – 0.030 |
| `rainfall` (mm/day) | 0 – 118.69 |
| `viirs_proxy` | 1,601 – 9,306 |
| `wefe_stress_index` | 0.395 – 0.861 |
| `category` | Green / Amber / Red |

Category bands: **Green** below 0.40, **Amber** 0.40–0.70, **Red** at or above 0.70.

The distribution is itself a finding: across nearly twelve years, the basin sits at **Amber on 3,211 days and Red on 1,089**, and reaches Green exactly **once**. Under this index, the Litani is essentially never unstressed.

## Modelling

Two regressors are fitted to predict the index from the four indicators, each on an 80/20 split with 5-fold cross-validation:

| Model | Test R² | 5-fold CV R² | RMSE |
|---|---|---|---|
| Linear Regression | 0.948 | 0.960 | 0.0153 |
| Random Forest | **0.980** | 0.979 | 0.0096 |

Random Forest feature importance:

| Indicator | Importance |
|---|---|
| NDTI (ecosystem) | 0.572 |
| NDVI (food) | 0.185 |
| Rainfall (water) | 0.181 |
| VIIRS (energy) | 0.062 |

**How to read these numbers.** The index is computed *from* these four indicators, so the models are recovering the proposed relationship rather than forecasting an independent outcome. High R² should not be read as predictive accuracy against reality. What it does show is that the proposed index is well behaved — learnable, stable across folds, and not dominated by noise — and the feature importances show which pillars carry the most weight in it, with NDTI contributing more than the other three combined.

## Repository

```
app.py                              Streamlit dashboard
linearRegresion.py                  Linear regression + 5-fold CV
randomForest.py                     Random forest + CV + feature importances
Litani_WEFE_Stressed_Data.csv       Processed daily observations, 2014-2025
Research Paper Final Version.docx   The accompanying paper
figures/                            Figures used in the paper
```

## Running it

Requires Python 3.10+.

```bash
pip install streamlit pandas numpy scikit-learn
```

The dashboard:

```bash
streamlit run app.py
```

The model scripts on their own:

```bash
python linearRegresion.py
python randomForest.py
```

### Using the dashboard

- **Monitoring Visuals** plot each indicator against the stress index over the full record.
- **Scenario Builder** lets you move each indicator and see the predicted index from both models, a blended score, and the resulting alert band.
- Sidebar shows held-out R² and RMSE, and accepts an alternative CSV with the same columns.

## Fixes applied 16 September 2026

Cleanup after the original submission. The research, dataset, and paper are unchanged; these correct defects in the dashboard.

- **Slider ranges were wrong, which broke the Scenario Builder.** The VIIRS slider ran 0–60 while the data ranges 1,601–9,306 — its maximum sat 26× *below* the data's minimum, so every prediction was made far outside the training distribution and the output was meaningless. Rainfall was likewise capped at 60 against a true maximum of 118.69. All sliders now derive their bounds from the dataset and start at its median, so they cannot drift out of range again.
- **Alert thresholds did not match the data.** The dashboard used 0.33 and 0.66, but the dataset's own labels place the boundaries at 0.40 and 0.70. Because the index never falls below 0.395, the old thresholds made the "Green" band unreachable. Bands now follow the dataset's labels.
- **The dashboard trained on the full dataset** with no held-out split and reported no metrics, unlike the two standalone scripts. It now uses the same 80/20 split and reports R² and RMSE in the sidebar.
- Figures were renamed from browser-download names (`download (1).png`, `download (3).png`, …) to meaningful ones and moved into `figures/`.

## Limitations

- The index is a proposal, not a validated standard, and has not been checked against ground measurements of water quality, yield, or electricity supply.
- The energy pillar uses night-time radiance as a proxy for grid stress, which also moves with population, cloud cover, and urban development.
- The analysis covers the Litani basin only; the thresholds would need re-deriving elsewhere.
- Daily satellite observations carry gaps and cloud contamination that the processing pipeline smooths over.
