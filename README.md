# Advanced Sales Forecasting with Python and Streamlit

An advanced Data Science project using public retail transactions to aggregate daily revenue, explore seasonality, engineer lag features, evaluate a gradient-boosting model, and forecast future sales.

## Features
- Public UCI Online Retail dataset
- Data cleaning and daily revenue aggregation (`Quantity × UnitPrice`)
- Excludes cancelled invoices and non-positive quantity/price rows for gross-sales analysis
- Trend and weekday seasonality analysis
- Calendar, lag, rolling mean, and rolling standard deviation features
- Chronological 80/20 holdout validation
- HistGradientBoostingRegressor vs weekly seasonal-naive baseline
- MAE, RMSE, WAPE metrics
- Streamlit dashboard with recursive 7–60 day forecast and CSV downloads

> Forecasts are uncertain and for educational purposes. This project does not guarantee future sales.

## Dataset
UCI Machine Learning Repository — Online Retail:
https://archive.ics.uci.edu/dataset/352/online+retail

The code downloads `https://archive.ics.uci.edu/static/public/352/online+retail.zip`. If automatic download fails, download the Excel workbook from the UCI page and save it as `data/Online Retail.xlsx`.

Revenue is defined as `Quantity × UnitPrice`. Cancelled invoices (InvoiceNo starts with `C`) and non-positive quantities/prices are excluded. Days without transactions are set to zero.

## Project structure
```text
advanced_sales_forecasting/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── notebooks/
│   └── advanced_sales_forecasting.ipynb
├── data/
│   └── (downloaded dataset and daily_sales.csv)
└── outputs/
    └── (metrics, holdout predictions, future forecast)
```

## Installation (Windows)
Open Command Prompt or Anaconda Prompt in the extracted project folder:
```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
These are terminal commands, not bare Python code for a notebook cell. In Jupyter use `%pip install pandas numpy matplotlib seaborn scikit-learn openpyxl requests` if necessary.

## Run Jupyter Notebook
From the project root:
```bash
jupyter notebook
```
Open `notebooks/advanced_sales_forecasting.ipynb` and run cells from top to bottom.

## Run Streamlit
From the project root:
```bash
python -m streamlit run app.py
```
Open the local URL, usually http://localhost:8501. The app can download the UCI data or accept CSV/Excel uploads with `InvoiceDate`, `Quantity`, and `UnitPrice` columns.

## Model evaluation
The first 80% of usable observations are training data and the last 20% are holdout data. The model uses date features and lag/rolling statistics from prior days. It is compared with a weekly seasonal-naive forecast (`sales[t-7]`). Lower MAE, RMSE, and WAPE are generally better. A single holdout split is only a starting point; advanced validation should use rolling-origin backtesting and compare multiple baselines.

Future forecasts are recursive, so errors can compound. Promotions, holidays, price changes, stock-outs, and external drivers are not modeled.

## Troubleshooting
- **ModuleNotFoundError:** activate `.venv` and run `python -m pip install -r requirements.txt`.
- **SyntaxError for `python -m venv`:** run it in Command Prompt/Anaconda Prompt; in Jupyter prefix shell commands with `!`.
- **app.py not found:** run Streamlit from the folder containing `app.py`, or pass its full path.
- **Dataset download failed:** check internet, retry, or download the Excel workbook manually from UCI.
- **Missing columns:** the app expects `InvoiceDate`, `Quantity`, and `UnitPrice`.
- **Too few rows:** use a longer dataset; at least 100 usable daily observations are recommended.

## Upload to GitHub
Create a repository named `advanced-sales-forecasting`, then run from the project root:
```bash
git init
git add app.py requirements.txt README.md .gitignore notebooks/
git commit -m "Build advanced sales forecasting project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/advanced-sales-forecasting.git
git push -u origin main
```
Replace `YOUR-USERNAME` with your GitHub username. Do not upload virtual environments.

## Resume bullet template
**Advanced Sales Forecasting | Python, Pandas, Scikit-learn, Streamlit**
- Aggregated public retail transactions into daily revenue and engineered calendar, lag, and rolling statistical features.
- Evaluated a HistGradientBoosting model against a weekly seasonal-naive benchmark using chronological holdout MAE, RMSE, and WAPE.
- Built an interactive Streamlit dashboard for trend analysis, weekday seasonality, model evaluation, and forecast export.

Only report measured performance after running the project and checking the metrics.
