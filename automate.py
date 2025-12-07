#!/usr/bin/env python3
"""
automation.py

Monthly automation for:
 - appending completed weekly sales (simulated),
 - fetching FRED economic indicators (GASREGW, CPIAUCSL),
 - aligning, merging, basic checks/anomaly detection,
 - sending email alerts when issues are detected,
 - saving outputs and logs.

Usage:
    python automation.py

Requirements:
    - data/synthetic_weekly_sales.csv must exist (canonical 2024 dataset)
    - api.env in project root with FRED_API_KEY=... and optional SMTP settings
    - pip install pandas numpy requests python-dotenv scipy
"""
import os
from datetime import datetime, timedelta
import logging
import smtplib
from email.message import EmailMessage

import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv
from scipy import stats  # optional; used in anomaly flow (you can remove if not installed)

# -----------------------
# Paths & logging
# -----------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

SALES_FILE = os.path.join(DATA_DIR, "synthetic_weekly_sales.csv")
MERGED_FILE = os.path.join(DATA_DIR, "merged_sales_econ_weekly.csv")
ANOMALIES_FILE = os.path.join(DATA_DIR, "anomalies_units_sold.csv")
LOG_FILE = os.path.join(LOG_DIR, "automation.log")

logging.basicConfig(level=logging.INFO, filename=LOG_FILE, filemode="a",
                    format="%(asctime)s %(levelname)s:%(message)s")

# -----------------------
# Load env
# -----------------------
ENV_PATH = os.path.join(PROJECT_ROOT, "api.env")
load_dotenv(ENV_PATH if os.path.exists(ENV_PATH) else None)

FRED_API_KEY = os.getenv("FRED_API_KEY")
if not FRED_API_KEY:
    logging.error("FRED_API_KEY not set in api.env or environment. Exiting.")
    raise SystemExit("FRED_API_KEY missing. Put it in api.env")

# SMTP / Email settings (optional)
ALERT_EMAIL_FROM = os.getenv("ALERT_EMAIL_FROM")
ALERT_EMAIL_TO   = os.getenv("ALERT_EMAIL_TO")
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT") or 0)
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")

# -----------------------
# Helpers
# -----------------------
def fetch_fred_series(series_id, start_date=None, end_date=None, api_key=FRED_API_KEY):
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json"}
    if start_date:
        params["observation_start"] = start_date
    if end_date:
        params["observation_end"] = end_date
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    obs = resp.json().get("observations", [])
    df = pd.DataFrame(obs)
    if df.empty:
        return pd.DataFrame(columns=["date", series_id])
    df["date"] = pd.to_datetime(df["date"])
    df[series_id] = pd.to_numeric(df["value"].replace('.', np.nan))
    return df[["date", series_id]]

def generate_sales_for_weeks(weeks, num_products=50, seed=42):
    np.random.seed(seed)
    product_ids = [f"P{i:03d}" for i in range(1, num_products + 1)]
    product_names = [f"Product_{i}" for i in range(1, num_products + 1)]
    base_prices = np.random.randint(10, 101, size=num_products)
    base_units = np.random.randint(50, 501, size=num_products)

    rows = []
    for week in weeks:
        week = pd.to_datetime(week)
        week_num = week.isocalendar()[1]
        for pid, pname, price, units_base in zip(product_ids, product_names, base_prices, base_units):
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * week_num / 52)
            discount = np.random.choice([0,5,10,15,20,25,30], p=[0.4,0.1,0.1,0.1,0.1,0.1,0.1])
            units_sold = int(units_base * seasonal_factor * (1 + discount / 100 * 0.5))
            revenue = units_sold * price * (1 - discount / 100)
            rows.append({
                "Week_Start_Date": week.normalize(),
                "Product_ID": pid,
                "Product_Name": pname,
                "Units_Sold": units_sold,
                "Price": float(price),
                "Discount_Percentage": int(discount),
                "Revenue": round(revenue, 2),
                "Region": "USA"
            })
    return pd.DataFrame(rows)

def append_new_sales(existing_df, new_df):
    if new_df.empty:
        return existing_df.copy()
    if existing_df.empty:
        return new_df.copy()
    existing_keys = set(map(tuple, existing_df[["Week_Start_Date","Product_ID"]].astype(str).values))
    rows = []
    for _, r in new_df.iterrows():
        key = (str(r["Week_Start_Date"]), str(r["Product_ID"]))
        if key not in existing_keys:
            rows.append(r)
    if rows:
        df_append = pd.DataFrame(rows)
        return pd.concat([existing_df, df_append], ignore_index=True)
    return existing_df.copy()

def run_checks_and_anomalies(df_merged):
    alerts = []
    # Critical missing values
    critical = ['Week_Start_Date','Product_ID','Units_Sold','Price','Revenue','Avg_Gas_Price_US','CPI']
    missing_summary = df_merged[critical].isna().sum()
    if missing_summary.any():
        msg = f"Missing critical values: {missing_summary.to_dict()}"
        logging.error(msg)
        alerts.append(msg)
    else:
        logging.info("No missing values in critical columns.")

    # Missing product-week combos
    all_products = df_merged['Product_ID'].unique()
    expected_weeks = pd.date_range(df_merged['Week_Start_Date'].min(), df_merged['Week_Start_Date'].max(), freq='W-SUN')
    expected = pd.MultiIndex.from_product([all_products, expected_weeks], names=['Product_ID','Week_Start_Date'])
    observed = pd.MultiIndex.from_frame(df_merged[['Product_ID','Week_Start_Date']])
    missing = expected.difference(observed)
    if len(missing) > 0:
        msg = f"Missing product-week records count: {len(missing)}"
        logging.warning(msg)
        alerts.append(msg)
    else:
        logging.info("No missing product-week rows.")

    # Simple anomaly detection (z-score per product)
    def safe_z(s):
        if s.std(ddof=0) == 0 or np.isnan(s.std(ddof=0)):
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - s.mean()) / s.std(ddof=0)
    df_merged['Units_Sold_z'] = df_merged.groupby('Product_ID')['Units_Sold'].transform(safe_z)
    anomalies = df_merged[np.abs(df_merged['Units_Sold_z']) > 3]
    if not anomalies.empty:
        msg = f"Detected {len(anomalies)} anomalies (|z|>3)"
        logging.warning(msg)
        alerts.append(msg)
    else:
        logging.info("No extreme anomalies detected.")

    # Econ missing check
    econ_missing = df_merged[['Avg_Gas_Price_US','CPI']].isna().sum()
    if econ_missing.any():
        msg = f"Economic indicators missing counts: {econ_missing.to_dict()}"
        logging.error(msg)
        alerts.append(msg)
    else:
        logging.info("Economic indicators present for all sales weeks.")

    anomalies.to_csv(ANOMALIES_FILE, index=False)
    return alerts

# -----------------------
# Email alert helper
# -----------------------
def send_email_alert(subject, body, attachments=None):
    """
    Send an email alert. attachments is an optional list of file paths to attach.
    Requires SMTP config variables present in environment.
    """
    if not all([ALERT_EMAIL_FROM, ALERT_EMAIL_TO, SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS]):
        logging.warning("Incomplete SMTP/email settings in env; skipping email alert.")
        return False

    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = ALERT_EMAIL_FROM
        msg["To"] = ALERT_EMAIL_TO
        msg.set_content(body)

        # attach files if provided
        if attachments:
            for path in attachments:
                try:
                    with open(path, "rb") as f:
                        data = f.read()
                    maintype = "application"
                    subtype = "octet-stream"
                    msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=os.path.basename(path))
                except Exception as e:
                    logging.warning("Failed to attach file %s: %s", path, e)

        # send via SMTP
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=60) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)

        logging.info("Email alert sent to %s", ALERT_EMAIL_TO)
        return True

    except Exception as e:
        logging.exception("Failed to send email alert: %s", e)
        return False

# -----------------------
# Main automation flow
# -----------------------
def main():
    logging.info("=== Automation run started ===")
    # Load existing sales (expected to be 2024 canonical initially)
    if not os.path.exists(SALES_FILE):
        logging.error("Sales file missing: %s", SALES_FILE)
        raise SystemExit(f"Sales file missing: {SALES_FILE}")
    df_sales = pd.read_csv(SALES_FILE, parse_dates=["Week_Start_Date"])
    df_sales['Week_Start_Date'] = pd.to_datetime(df_sales['Week_Start_Date'])

    # Determine completed-week cutoff: last Sunday of previous month
    today = datetime.now().date()
    first_of_month = today.replace(day=1)
    last_day_prev_month = first_of_month - timedelta(days=1)
    last_sunday_prev_month = last_day_prev_month - timedelta(days=(last_day_prev_month.weekday() + 1) % 7)

    # Determine next-week to create
    last_week = df_sales['Week_Start_Date'].max()
    start_candidate = pd.to_datetime(last_week) + pd.Timedelta(days=7)
    if start_candidate > pd.to_datetime(last_sunday_prev_month):
        target_weeks = []
    else:
        target_weeks = pd.date_range(start=start_candidate, end=last_sunday_prev_month, freq="W-SUN")

    logging.info("New weeks to generate (completed weeks only): %s", [str(w.date()) for w in target_weeks])

    # Generate and append sales
    df_new_sales = generate_sales_for_weeks(target_weeks)
    df_combined_sales = append_new_sales(df_sales, df_new_sales)
    df_combined_sales['Week_Start_Date'] = pd.to_datetime(df_combined_sales['Week_Start_Date']).dt.normalize()
    df_combined_sales.to_csv(SALES_FILE, index=False)
    logging.info("Saved sales file with shape: %s", df_combined_sales.shape)

    # FRED fetch window (from earliest sales to completed-week cutoff)
    fetch_start = df_combined_sales['Week_Start_Date'].min().strftime("%Y-%m-%d")
    fetch_end_dt = min(pd.to_datetime('today').normalize(), pd.to_datetime(last_sunday_prev_month))
    fetch_end = fetch_end_dt.strftime("%Y-%m-%d")
    logging.info("Fetching FRED data for %s -> %s", fetch_start, fetch_end)

    # Fetch FRED series
    df_gas = fetch_fred_series("GASREGW", start_date=fetch_start, end_date=fetch_end)
    df_cpi = fetch_fred_series("CPIAUCSL", start_date=fetch_start, end_date=fetch_end)

    # Align econ time series to weekly Sundays
    weekly_index = pd.date_range(start=fetch_start, end=fetch_end, freq="W-SUN")

    # Gas: resample & interpolate
    if not df_gas.empty:
        df_gas_weekly = df_gas.set_index("date").resample("W-SUN").mean().reindex(weekly_index)
        if 'GASREGW' in df_gas_weekly.columns:
            df_gas_weekly['GASREGW'] = df_gas_weekly['GASREGW'].interpolate(method='time')
    else:
        df_gas_weekly = pd.DataFrame(index=weekly_index)
    df_gas_weekly = df_gas_weekly.rename_axis('Week_Start_Date').reset_index().rename(columns={'GASREGW':'Avg_Gas_Price_US'})

    # CPI: forward-fill monthly to daily then resample weekly
    if not df_cpi.empty:
        df_cpi_idx = df_cpi.set_index("date").sort_index()
        daily = df_cpi_idx.reindex(pd.date_range(df_cpi_idx.index.min(), df_cpi_idx.index.max(), freq="D")).fillna(method='ffill')
        df_cpi_weekly = daily.resample("W-SUN").last().reindex(weekly_index).ffill()
        df_cpi_weekly = df_cpi_weekly.rename_axis('Week_Start_Date').reset_index().rename(columns={'CPIAUCSL':'CPI'})
    else:
        df_cpi_weekly = pd.DataFrame({'Week_Start_Date': weekly_index, 'CPI': [np.nan]*len(weekly_index)})

    # Merge sales + econ
    df_merged = df_combined_sales.merge(df_gas_weekly[['Week_Start_Date','Avg_Gas_Price_US']], on='Week_Start_Date', how='left')
    df_merged = df_merged.merge(df_cpi_weekly[['Week_Start_Date','CPI']], on='Week_Start_Date', how='left')

    # Run checks & anomalies
    alerts = run_checks_and_anomalies(df_merged)

    # Save merged file & anomalies
    df_merged.to_csv(MERGED_FILE, index=False)
    logging.info("Saved merged file: %s", MERGED_FILE)

    # If alerts exist, print/log and email stakeholders
    if alerts:
        alert_text = "\n\n".join(alerts)
        print("ALERTS FOUND:\n", alert_text)
        logging.warning("Alerts found during data checks. Sending email (if configured).")
        # Attempt to attach anomalies file if exists
        attachments = [ANOMALIES_FILE] if os.path.exists(ANOMALIES_FILE) else None
        sent = send_email_alert(
            subject="DATA PIPELINE ALERT - Issues Detected",
            body=alert_text,
            attachments=attachments
        )
        if sent:
            print("Alert email sent.")
        else:
            print("Email alert not sent (see logs).")
    else:
        print("All checks passed. Files saved:")
        print(" -", SALES_FILE)
        print(" -", MERGED_FILE)

    logging.info("=== Automation run finished ===")

if __name__ == "__main__":
    main()
