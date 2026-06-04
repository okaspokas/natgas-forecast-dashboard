import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Natural Gas Price Forecasting",
    page_icon="🔥",
    layout="wide"
)

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# ── Load data ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_DIR / "raw_prices.csv", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

@st.cache_data
def load_forecast():
    df = pd.read_csv(DATA_DIR / "forecast.csv", parse_dates=["date"])
    return df

@st.cache_resource
def load_model():
    with open(MODEL_DIR / "prophet_model.pkl", "rb") as f:
        return pickle.load(f)

# ── Load everything ───────────────────────────────────────────
try:
    df = load_data()
    forecast = load_forecast()
    model = load_model()
    data_loaded = True
except FileNotFoundError as e:
    data_loaded = False
    missing = str(e)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔥 Gas Price Dashboard")
    st.markdown("**Prepared by:** Ojas Khetarpal  \n**Role:** Summer Trainee")
    st.divider()

    if data_loaded:
        st.markdown("### ⚙️ Settings")
        history_years = st.slider("History to display (years)", 1, int((df.date.max() - df.date.min()).days / 365), 3)
        forecast_days = st.slider("Forecast horizon (days)", 30, 90, 90)
        show_ci = st.toggle("Show confidence interval", value=True)
        show_rolling = st.toggle("Show 30-day rolling avg", value=True)

        st.divider()
        st.markdown("### 📊 Dataset Info")
        st.metric("Total records", f"{len(df):,}")
        st.metric("Date range", f"{df.date.min().year} – {df.date.max().year}")
        st.metric("Avg price", f"${df.price.mean():.2f}/MMBtu")

# ── Main content ──────────────────────────────────────────────
if not data_loaded:
    st.error(f"❌ Could not load data: `{missing}`")
    st.info("Make sure you have the following files in your project:\n- `data/raw_prices.csv`\n- `data/forecast.csv`\n- `models/prophet_model.pkl`")
    st.stop()

# ── Header ────────────────────────────────────────────────────
st.title("🔥 Natural Gas Price Forecasting Dashboard")
st.caption("Henry Hub Natural Gas Spot Price — Powered by Facebook Prophet")
st.divider()

# ── KPI Cards ─────────────────────────────────────────────────
hist_end = df.date.max()
hist_start = hist_end - pd.DateOffset(years=history_years)
df_view = df[df.date >= hist_start].copy()
df_view["rolling"] = df_view["price"].rolling(30).mean()

future_fc = forecast[forecast.date > df.date.max()].head(forecast_days)
latest_price = df.price.iloc[-1]
prev_price = df.price.iloc[-2]
price_change = latest_price - prev_price
price_pct = (price_change / prev_price) * 100
forecast_avg = future_fc["forecast"].mean()
forecast_end = future_fc["forecast"].iloc[-1]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest Price", f"${latest_price:.2f}/MMBtu", f"{price_change:+.2f} ({price_pct:+.1f}%)")
col2.metric("30-Day Avg", f"${df.price.tail(30).mean():.2f}/MMBtu")
col3.metric(f"{forecast_days}-Day Forecast Avg", f"${forecast_avg:.2f}/MMBtu")
col4.metric(f"Forecast End Price", f"${forecast_end:.2f}/MMBtu", f"{forecast_end - latest_price:+.2f} vs today")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecast", "📊 Historical Analysis", "🔍 Seasonality", "📋 Data Table"])

# ── TAB 1: Forecast ───────────────────────────────────────────
with tab1:
    st.subheader(f"Price History + {forecast_days}-Day Forecast")

    fig = go.Figure()

    if show_ci:
        ci_dates = pd.concat([future_fc["date"], future_fc["date"][::-1]])
        ci_values = pd.concat([future_fc["upper"], future_fc["lower"][::-1]])
        fig.add_trace(go.Scatter(
            x=ci_dates, y=ci_values,
            fill="toself", fillcolor="rgba(231,76,60,0.1)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% Confidence Interval"
        ))

    fig.add_trace(go.Scatter(
        x=df_view["date"], y=df_view["price"],
        mode="lines", name="Historical Price",
        line=dict(color="#2E75B6", width=1.2), opacity=0.8
    ))

    if show_rolling:
        fig.add_trace(go.Scatter(
            x=df_view["date"], y=df_view["rolling"],
            mode="lines", name="30-Day Rolling Avg",
            line=dict(color="#1F4E79", width=2)
        ))

    fig.add_trace(go.Scatter(
        x=future_fc["date"], y=future_fc["forecast"],
        mode="lines", name="Forecast",
        line=dict(color="#E74C3C", width=2.5, dash="dash")
    ))

    fig.add_vline(
        x=df.date.max(), line_dash="dot",
        line_color="gray", annotation_text="Forecast start",
        annotation_position="top right"
    )

    fig.update_layout(
        height=480,
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Price ($/MMBtu)",
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=10, r=10, t=20, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

# ── TAB 2: Historical Analysis ────────────────────────────────
with tab2:
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Yearly Average Price")
        df["year"] = df["date"].dt.year
        yearly = df.groupby("year")["price"].mean().reset_index()
        fig2 = px.bar(
            yearly, x="year", y="price",
            color="price", color_continuous_scale="Blues",
            labels={"price": "Avg Price ($/MMBtu)", "year": "Year"}
        )
        fig2.update_layout(height=360, coloraxis_showscale=False, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        st.subheader("Price Distribution")
        fig3 = px.histogram(
            df, x="price", nbins=60,
            color_discrete_sequence=["#2E75B6"],
            labels={"price": "Price ($/MMBtu)", "count": "Frequency"}
        )
        fig3.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Year-on-Year Price Comparison")
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.strftime("%b")
    recent_years = sorted(df["year"].unique())[-5:]
    df_recent = df[df["year"].isin(recent_years)]
    monthly_yearly = df_recent.groupby(["year", "month", "month_name"])["price"].mean().reset_index()
    monthly_yearly = monthly_yearly.sort_values("month")

    fig4 = px.line(
        monthly_yearly, x="month_name", y="price", color="year",
        markers=True,
        labels={"price": "Avg Price ($/MMBtu)", "month_name": "Month", "year": "Year"},
        color_discrete_sequence=px.colors.sequential.Blues_r[:5]
    )
    fig4.update_layout(height=380, hovermode="x unified", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig4, use_container_width=True)

# ── TAB 3: Seasonality ────────────────────────────────────────
with tab3:
    st.subheader("Prophet Model Components")
    st.caption("Trend, weekly, and yearly seasonality extracted by the model")

    try:
        import matplotlib.pyplot as plt
        fig_comp = model.plot_components(forecast)
        fig_comp.suptitle("")
        st.pyplot(fig_comp, use_container_width=True)
        plt.close()
    except Exception as e:
        st.warning(f"Could not render components: {e}")

    st.divider()
    st.subheader("Average Price by Month (All Years)")
    df["month_name"] = df["date"].dt.strftime("%b")
    df["month_num"] = df["date"].dt.month
    seasonal = df.groupby(["month_num", "month_name"])["price"].mean().reset_index().sort_values("month_num")

    fig5 = go.Figure()
    fig5.add_trace(go.Bar(
        x=seasonal["month_name"], y=seasonal["price"],
        marker_color="#2E75B6",
        name="Avg Price"
    ))
    fig5.update_layout(
        height=360, yaxis_title="Avg Price ($/MMBtu)",
        margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig5, use_container_width=True)

# ── TAB 4: Data Table ─────────────────────────────────────────
with tab4:
    st.subheader("Raw Price Data")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        date_from = st.date_input("From", value=df.date.max() - pd.Timedelta(days=365))
    with col_f2:
        date_to = st.date_input("To", value=df.date.max())

    df_filtered = df[
        (df.date >= pd.Timestamp(date_from)) &
        (df.date <= pd.Timestamp(date_to))
    ][["date", "price"]].copy()
    df_filtered.columns = ["Date", "Price ($/MMBtu)"]
    df_filtered = df_filtered.sort_values("Date", ascending=False)

    st.dataframe(df_filtered, use_container_width=True, height=400)

    csv = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download filtered data as CSV", csv, "gas_prices_filtered.csv", "text/csv")
