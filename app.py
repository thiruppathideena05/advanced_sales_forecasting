from datetime import timedelta
import io, zipfile
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline

st.set_page_config(page_title="Sales Forecasting Lab", page_icon="🛍️", layout="wide")
st.title("🛍️ Sales Forecasting & Demand Analytics")
st.caption("Public retail transactions · time-series features · chronological model evaluation")

DATA_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"

@st.cache_data(ttl=3600)
def download_uci():
    import requests
    r = requests.get(DATA_URL, timeout=60)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        names = [n for n in z.namelist() if n.lower().endswith((".xlsx", ".xls"))]
        if not names:
            raise ValueError("No Excel workbook found in UCI archive.")
        return pd.read_excel(io.BytesIO(z.read(names[0])))

def daily_sales(tx):
    d = tx.copy()
    d.columns = [str(c).strip() for c in d.columns]
    required = {"InvoiceDate", "Quantity", "UnitPrice"}
    if not required.issubset(d.columns):
        raise ValueError("Required columns: InvoiceDate, Quantity, UnitPrice.")
    d["InvoiceDate"] = pd.to_datetime(d["InvoiceDate"], errors="coerce")
    d["Quantity"] = pd.to_numeric(d["Quantity"], errors="coerce")
    d["UnitPrice"] = pd.to_numeric(d["UnitPrice"], errors="coerce")
    d = d.dropna(subset=["InvoiceDate", "Quantity", "UnitPrice"])
    if "InvoiceNo" in d:
        d = d[~d["InvoiceNo"].astype(str).str.startswith("C", na=False)]
    d = d[(d["Quantity"] > 0) & (d["UnitPrice"] > 0)]
    d["Revenue"] = d["Quantity"] * d["UnitPrice"]
    s = d.set_index("InvoiceDate")["Revenue"].resample("D").sum().asfreq("D").fillna(0)
    s.name = "Sales"
    s.index.name = "Date"
    return s

def features_for(series):
    f = pd.DataFrame({"Sales": series.astype(float)})
    ix = f.index
    for name, values in {
        "dow": ix.dayofweek, "day": ix.day, "month": ix.month,
        "quarter": ix.quarter, "year": ix.year, "dayofyear": ix.dayofyear,
        "weekend": (ix.dayofweek >= 5).astype(int)
    }.items():
        f[name] = values
    for lag in [1, 2, 3, 7, 14, 21, 28]:
        f[f"lag_{lag}"] = f["Sales"].shift(lag)
    for w in [7, 14, 28]:
        f[f"rollmean_{w}"] = f["Sales"].shift(1).rolling(w).mean()
        f[f"rollstd_{w}"] = f["Sales"].shift(1).rolling(w).std()
    return f.replace([np.inf, -np.inf], np.nan)

def fit_evaluate(s):
    f = features_for(s)
    cols = [c for c in f.columns if c != "Sales"]
    d = f.dropna(subset=cols + ["Sales"])
    if len(d) < 100:
        raise ValueError("Need at least 100 usable daily observations; choose a longer dataset.")
    split = int(len(d) * .8)
    tr, te = d.iloc[:split], d.iloc[split:]
    model = make_pipeline(SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(max_iter=250, learning_rate=.06,
            max_leaf_nodes=15, l2_regularization=1.0, random_state=42))
    model.fit(tr[cols], tr["Sales"])
    pred = np.maximum(0, model.predict(te[cols]))
    base = te["lag_7"].to_numpy()
    actual = te["Sales"].to_numpy()
    def metric(y, p):
        return {"MAE": mean_absolute_error(y,p),
                "RMSE": np.sqrt(mean_squared_error(y,p)),
                "WAPE_%": np.abs(y-p).sum()/max(np.abs(y).sum(),1e-9)*100}
    scores = pd.DataFrame({"HistGradientBoosting":metric(actual,pred),
                           "Weekly seasonal naive":metric(actual,base)}).T
    results = pd.DataFrame({"Actual":actual,"Model":pred,"Weekly_naive":base}, index=te.index)
    return results, scores, model, cols

def forecast_future(s, n, model, cols):
    hist = s.astype(float).copy()
    dates = pd.date_range(hist.index[-1] + pd.Timedelta(days=1), periods=n, freq="D")
    vals = []
    for dt in dates:
        extended = pd.concat([hist, pd.Series(index=[dt], dtype=float)])
        row = features_for(extended).loc[[dt], cols]
        value = max(0.0, float(model.predict(row)[0]))
        hist.loc[dt] = value
        vals.append(value)
    return pd.DataFrame({"Forecast":vals}, index=dates)

with st.sidebar:
    source = st.radio("Dataset source", ["Download UCI public dataset", "Upload CSV/Excel"])
    uploaded = st.file_uploader("Transaction file", type=["csv","xlsx","xls"]) if source.startswith("Upload") else None
    horizon = st.slider("Forecast horizon (calendar days)", 7, 60, 30)

try:
    if source.startswith("Download"):
        with st.spinner("Downloading UCI Online Retail dataset..."):
            tx = download_uci()
    else:
        if uploaded is None:
            st.info("Upload a transaction file to continue.")
            st.stop()
        tx = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
    sales = daily_sales(tx)
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.info("Try the UCI dataset page or upload a file with InvoiceDate, Quantity, and UnitPrice.")
    st.stop()

if len(sales) < 100:
    st.error("Not enough daily observations.")
    st.stop()
c1,c2,c3,c4 = st.columns(4)
c1.metric("Total revenue", f"{sales.sum():,.2f}")
c2.metric("Average daily revenue", f"{sales.mean():,.2f}")
c3.metric("Peak sales day", sales.idxmax().strftime("%Y-%m-%d"))
c4.metric("Daily observations", f"{len(sales):,}")

t1,t2,t3,t4 = st.tabs(["Overview","Seasonality","Forecast & evaluation","Data"])
with t1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sales.index,y=sales.values,name="Daily revenue"))
    fig.add_trace(go.Scatter(x=sales.index,y=sales.rolling(7).mean(),name="7-day mean"))
    fig.update_layout(title="Daily revenue",height=430,xaxis_title="Date",yaxis_title="Revenue")
    st.plotly_chart(fig,use_container_width=True)
with t2:
    weekday = sales.groupby(sales.index.day_name()).mean().reindex(
        ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    fig = go.Figure(go.Bar(x=weekday.index,y=weekday.values))
    fig.update_layout(title="Average revenue by weekday",height=350,yaxis_title="Revenue")
    st.plotly_chart(fig,use_container_width=True)
    month = sales.groupby(sales.index.month).mean()
    fig = go.Figure(go.Bar(x=month.index,y=month.values))
    fig.update_layout(title="Average daily revenue by month number",height=350)
    st.plotly_chart(fig,use_container_width=True)
with t3:
    with st.spinner("Training and evaluating model..."):
        try:
            results, scores, model, cols = fit_evaluate(sales)
            future = forecast_future(sales,horizon,model,cols)
        except Exception as e:
            st.error(str(e)); st.stop()
    a,b,c = st.columns(3)
    a.metric("Model MAE",f"{scores.loc['HistGradientBoosting','MAE']:,.2f}")
    b.metric("Weekly baseline MAE",f"{scores.loc['Weekly seasonal naive','MAE']:,.2f}")
    delta = (1-scores.loc["HistGradientBoosting","MAE"]/max(scores.loc["Weekly seasonal naive","MAE"],1e-9))*100
    c.metric("MAE improvement",f"{delta:.1f}%")
    st.dataframe(scores.style.format("{:,.3f}"),use_container_width=True)
    fig = go.Figure()
    for col in results.columns:
        fig.add_trace(go.Scatter(x=results.index,y=results[col],name=col))
    fig.update_layout(title="Chronological holdout results",height=400,yaxis_title="Revenue")
    st.plotly_chart(fig,use_container_width=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sales.tail(60).index,y=sales.tail(60),name="Recent actual"))
    fig.add_trace(go.Scatter(x=future.index,y=future["Forecast"],name="Recursive forecast",line=dict(dash="dash")))
    fig.update_layout(title=f"Next {horizon} days forecast",height=400,yaxis_title="Revenue")
    st.plotly_chart(fig,use_container_width=True)
    st.warning("Recursive predictions can compound error and do not include future promotions, holidays, pricing, or external drivers.")
    st.download_button("Download holdout predictions",results.to_csv().encode(), "holdout_predictions.csv","text/csv")
    st.download_button("Download future forecast",future.to_csv().encode(),"future_sales_forecast.csv","text/csv")
with t4:
    st.dataframe(sales.to_frame().tail(100),use_container_width=True)
    st.download_button("Download daily sales CSV",sales.to_csv().encode(),"daily_sales.csv","text/csv")
