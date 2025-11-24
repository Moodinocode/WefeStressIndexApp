import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

DATA_PATH = Path("Litani_WEFE_Stressed_Data.csv")
FEATURE_COLUMNS = ["ndvi", "ndti", "rainfall", "viirs_proxy"]
TARGET_COLUMN = "wefe_stress_index"
GRAPH_FILES = [
    ("download.png", "Composite WEFE stress signals discussed in the paper."),
    ("download (1).png", "Seasonal NDVI departures across Litani farms."),
    ("download (3).png", "NDTI-derived turbidity hotspots."),
    ("download (4).png", "CHIRPS rainfall anomalies during 2025 drought."),
    ("download (7).png", "Night-time radiance declines used as an energy proxy."),
]


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
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    lr_model = LinearRegression()
    lr_model.fit(X, y)

    rf_model = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    rf_model.fit(X, y)
    return lr_model, rf_model


def format_prediction(label: str, value: float):
    st.metric(label, f"{value:.3f}", help="WEFE stress index (0 = low stress, 1 = high stress).")


st.set_page_config(
    page_title="Litani WEFE Stress Cockpit",
    page_icon="🌊",
    layout="wide",
)

st.title("Litani WEFE Stress Cockpit")
st.caption(
    "Interface inspired by the research paper's WEFE stress framework: rainfall (Water), "
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
st.write("Stress projections update live as you adjust remote-sensing indicators.")

col1, col2 = st.columns(2)
with col1:
    ndvi = st.slider("NDVI (0–1)", 0.0, 1.0, 0.45, 0.01)
    rainfall = st.slider("Rainfall (mm/day)", 0.0, 60.0, 5.0, 0.5)
with col2:
    ndti = st.slider("NDTI (-1–1)", -1.0, 1.0, 0.1, 0.01)
    viirs = st.slider("VIIRS radiance (nW/cm²·sr)", 0.0, 60.0, 12.0, 0.5)

input_vector = np.array([[ndvi, ndti, rainfall, viirs]])

if models is not None:
    linear_model, rf_model = models
    lr_pred = float(linear_model.predict(input_vector)[0])
    rf_pred = float(rf_model.predict(input_vector)[0])
    blended_pred = (lr_pred + rf_pred) / 2

    st.success("Predicted WEFE Stress Index")
    metrics = st.columns(3)
    metrics[0].metric("Linear Regression", f"{lr_pred:.3f}")
    metrics[1].metric("Random Forest", f"{rf_pred:.3f}")
    metrics[2].metric("Blended Alert Score", f"{blended_pred:.3f}")

    if blended_pred >= 0.66:
        st.error("Red alert: multi-pillar stress is likely imminent.")
    elif blended_pred >= 0.33:
        st.warning("Amber alert: monitor closely and prepare mitigation steps.")
    else:
        st.success("Green: conditions stable, continue routine monitoring.")
else:
    st.info("Add the processed Litani dataset to unlock live predictions.")


