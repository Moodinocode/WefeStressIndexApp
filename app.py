import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path("Litani_WEFE_Stressed_Data.csv")
FEATURE_COLUMNS = ["ndvi", "ndti", "rainfall", "viirs_proxy"]
TARGET_COLUMN = "wefe_stress_index"

# Boundaries taken from the `category` labels in the dataset rather than chosen
# by hand: Amber begins at 0.40 and Red at 0.70.
AMBER_THRESHOLD = 0.40
RED_THRESHOLD = 0.70

GRAPH_FILES = [
    ("figures/wefe_composite_stress.png", "Composite WEFE stress signals discussed in the paper."),
    ("figures/ndvi_seasonal_departures.png", "Seasonal NDVI departures across Litani farms."),
    ("figures/ndti_turbidity_hotspots.png", "NDTI-derived turbidity hotspots."),
    ("figures/rainfall_anomalies_2025.png", "CHIRPS rainfall anomalies during the 2025 drought."),
    ("figures/viirs_radiance_decline.png", "Night-time radiance declines used as an energy proxy."),
]

SLIDER_LABELS = {
    "ndvi": "NDVI (vegetation health)",
    "ndti": "NDTI (water turbidity)",
    "rainfall": "Rainfall (mm/day)",
    "viirs_proxy": "VIIRS radiance (energy proxy)",
}


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame | None:
    missing_cols = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in df.columns]
    if missing_cols:
        st.warning(f"Dataset is missing required columns: {missing_cols}")
        return None
    return df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])


@st.cache_data(show_spinner=False)
def load_dataset(source: Path) -> pd.DataFrame | None:
    if not source.exists():
        return None
    df = pd.read_csv(source)
    return validate_dataset(df)


@st.cache_resource(show_spinner=False)
def train_models(df: pd.DataFrame):
    """Fit both models on a training split and score them on held-out data.

    The scores are reported in the sidebar so the figures on screen come from
    data the models never saw, rather than from the rows they were fitted on.
    """
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    rf_model = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    rf_model.fit(X_train, y_train)

    scores = {}
    for name, model in (("Linear Regression", lr_model), ("Random Forest", rf_model)):
        preds = model.predict(X_test)
        scores[name] = {
            "r2": r2_score(y_test, preds),
            "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        }

    return lr_model, rf_model, scores, len(X_train), len(X_test)


def slider_bounds(df: pd.DataFrame, column: str) -> tuple[float, float, float, float]:
    """Derive a slider's range from the data so the controls cannot drift out of
    the range the models were trained on. VIIRS radiance in particular runs in
    the thousands, and a hardcoded range silently produced meaningless
    predictions."""
    low = float(df[column].min())
    high = float(df[column].max())
    span = high - low
    step = max(span / 200, 1e-4)
    default = float(df[column].median())
    return low, high, default, step


def classify(score: float) -> tuple[str, str]:
    if score >= RED_THRESHOLD:
        return "error", "Red alert: multi-pillar stress is likely imminent."
    if score >= AMBER_THRESHOLD:
        return "warning", "Amber: monitor closely and prepare mitigation steps."
    return "success", "Green: conditions stable, continue routine monitoring."


st.set_page_config(
    page_title="Litani WEFE Stress Cockpit",
    page_icon="🌊",
    layout="wide",
)

st.title("Litani WEFE Stress Cockpit")
st.caption(
    "Interface built on the research paper's WEFE stress framework: rainfall (Water), "
    "VIIRS radiance (Energy), NDVI (Food), and NDTI (Ecosystems) feed predictive models "
    "to flag emerging municipal stress."
)

st.sidebar.header("Data Inputs")
uploaded_csv = st.sidebar.file_uploader("Upload updated WEFE CSV (optional)", type="csv")
dataset = None

if uploaded_csv is not None:
    dataset = validate_dataset(pd.read_csv(uploaded_csv))
    if dataset is not None:
        st.sidebar.success("Using uploaded dataset.")
else:
    dataset = load_dataset(DATA_PATH)
    if dataset is None:
        st.sidebar.warning(f"Place `{DATA_PATH.name}` in the project folder to enable model training.")


models = None
if dataset is not None:
    try:
        models = train_models(dataset)
    except ValueError as exc:
        st.error(f"Unable to train models: {exc}")

if models is not None:
    _, _, scores, n_train, n_test = models
    st.sidebar.subheader("Model performance")
    st.sidebar.caption(f"Trained on {n_train} rows, scored on {n_test} held out.")
    for name, metrics in scores.items():
        st.sidebar.write(f"**{name}** — R² {metrics['r2']:.3f}, RMSE {metrics['rmse']:.4f}")
    st.sidebar.caption(
        "The stress index is itself derived from these four indicators, so high "
        "R² shows the models recovering that relationship, not forecasting an "
        "independent outcome."
    )


with st.expander("Why these signals?"):
    st.write(
        "Per the research paper, the Litani municipal stress index relies on remote-sensing pillars:\n"
        "• **NDVI (Food pillar):** Tracks vegetation health and irrigation pressure.\n"
        "• **NDTI (Ecosystem pillar):** Captures turbidity and ecological degradation.\n"
        "• **Rainfall (Water pillar):** Summarizes CHIRPS precipitation anomalies.\n"
        "• **VIIRS Night Lights (Energy proxy):** Signals electrical supply shortfalls.\n"
        "The models detect compounded stress when multiple pillars show adverse trajectories."
    )


st.subheader("Monitoring Visuals")
if dataset is not None:
    st.write("Latest processed Litani WEFE observations.")
    indicator_tabs = st.tabs(["NDVI vs Stress", "NDTI vs Stress", "Rainfall vs Stress", "VIIRS vs Stress"])
    pairs = [
        ("ndvi", "NDVI"),
        ("ndti", "NDTI"),
        ("rainfall", "Rainfall (mm/day)"),
        ("viirs_proxy", "VIIRS radiance"),
    ]
    for tab, (column, label) in zip(indicator_tabs, pairs):
        with tab:
            chart_df = dataset[[column, TARGET_COLUMN]].rename(
                columns={column: label, TARGET_COLUMN: "WEFE Stress Index"}
            )
            st.line_chart(chart_df)
else:
    st.info("Dataset unavailable — showing static figures from the paper instead.")
    cols = st.columns(2)
    for idx, (graph_path, caption) in enumerate(GRAPH_FILES):
        target = Path(graph_path)
        if not target.exists():
            st.warning(f"Missing figure `{graph_path}`.")
            continue
        cols[idx % 2].image(str(target), caption=caption, use_container_width=True)


st.subheader("Scenario Builder")

if dataset is None or models is None:
    st.info("Add the processed Litani dataset to unlock live predictions.")
else:
    st.write(
        "Stress projections update live as you adjust the indicators. Each slider "
        "spans the range observed in the dataset and starts at its median."
    )

    inputs = {}
    col1, col2 = st.columns(2)
    for idx, column in enumerate(FEATURE_COLUMNS):
        low, high, default, step = slider_bounds(dataset, column)
        target_col = col1 if idx % 2 == 0 else col2
        inputs[column] = target_col.slider(
            f"{SLIDER_LABELS[column]}  ({low:.4g} – {high:.4g})",
            min_value=low,
            max_value=high,
            value=default,
            step=step,
        )

    input_vector = pd.DataFrame([[inputs[c] for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)

    linear_model, rf_model, _, _, _ = models
    lr_pred = float(linear_model.predict(input_vector)[0])
    rf_pred = float(rf_model.predict(input_vector)[0])
    blended_pred = (lr_pred + rf_pred) / 2

    st.success("Predicted WEFE Stress Index")
    metrics = st.columns(3)
    metrics[0].metric("Linear Regression", f"{lr_pred:.3f}")
    metrics[1].metric("Random Forest", f"{rf_pred:.3f}")
    metrics[2].metric("Blended Alert Score", f"{blended_pred:.3f}")

    level, message = classify(blended_pred)
    getattr(st, level)(message)
    st.caption(
        f"Bands follow the dataset's own labels: Green below {AMBER_THRESHOLD:.2f}, "
        f"Amber {AMBER_THRESHOLD:.2f}–{RED_THRESHOLD:.2f}, Red at or above {RED_THRESHOLD:.2f}."
    )
