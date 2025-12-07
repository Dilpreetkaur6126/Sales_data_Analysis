

# **Retail Sales Data Automation Pipeline**

This project simulates weekly retail sales for 50 products, enriches the dataset using real economic indicators from the **FRED API**, performs data quality checks, detects anomalies, and automates monthly refreshes.
An  **Airflow DAG** is also included for workflow scheduling.

---

##  **Features**

* Synthetic sales dataset generation (weekly, 2024)
* Economic data ingestion from FRED (Gas Prices + CPI)
* Time-series alignment & weekly interpolation
* Automated monthly refresh (new completed weeks only)
* Data quality checks:

  * Missing values
  * Missing product–week combinations
  * Anomaly detection using z-score
* Logging to `logs/automation.log`
* Email alerts for detected issues
* Airflow DAG for scheduling

---

# 📁 **Repository Structure**

```
project_root/
│
├── automation.py                    # Main pipeline script
├── monthly_sales_automation_dag.py  # Airflow DAG 
├── api.env.example                  # Example env file 
│
├── data/
│   ├── synthetic_weekly_sales.csv    # Base 2024 dataset
│   └── merged_sales_econ_weekly.csv  # MergedData(output)
│
├── logs/
│   └── automation.log
│
└── README.md
```

---

#  **1. Execution Instructions**

## **Prerequisites**

Install required Python packages:

```bash
pip install pandas numpy requests python-dotenv scipy
```

---

#  **2. Environment Setup**

Create a file named **api.env** in the project root:

```
FRED_API_KEY=your_fred_api_key

# Email Alerts (Optional)
ALERT_EMAIL_FROM=youremail@gmail.com
ALERT_EMAIL_TO=alerts_recipient@gmail.com
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=youremail@gmail.com
SMTP_PASS=your_app_password
```

>  if using gmail, please use **App password**.


---

#  **3. Running the Pipeline Manually**

Run:

```bash
python automation.py
```

Upon running, the script will:

1. Load existing sales data(already created data for 2024)
2. Detect newly completed calendar weeks
3. Generate synthetic sales for those weeks
4. Fetch updated FRED economic indicators
5. Align data and merge
6. Run quality checks + anomaly detection
7. Save updated output
8. Write logs
9. Send alert email (if issues detected)

---

# 📊 **4. How the Anomaly Detection Works**

Anomalies are detected using **z-score**:

[
z = \frac{x - \mu}{\sigma}
]

Where:

| Term       | Meaning                                |
| ---------- | -------------------------------------- |
| ( x )      | Units sold in a given week             |
| ( \mu )    | Mean weekly units sold for the product |
| ( \sigma ) | Standard deviation                     |

Any record where:

```
|z| > 3
```

is considered **statistically abnormal** and added to:
`data/anomalies_units_sold.csv`

Common examples:

* Sudden sales spike
* Unexpected drop
* Incorrectly entered units

---

#  **5. Scheduling Options**



 Airflow 

A DAG file is provided:

```
monthly_sales_automation_dag.py
```

Place it in:

```
~/airflow/dags/
```

The DAG:

* Runs on the **first day of every month**
* Executes `automation.py`

---

#  **6. Overview of the Approach**

### **1. Data Generation**

* 50 products
* Weekly sales for all Sundays in 2024
* Seasonality modeled using sinusoidal variation
* Discounts randomly applied
* Revenue computed per row

### **2. Economic Data Acquisition**

* FRED series used:

  * `GASREGW` → Weekly regular gasoline price
  * `CPIAUCSL` → Consumer Price Index
* Gas data interpolated weekly
* CPI (monthly) → expanded to daily → resampled weekly

### **3. Data Refresh Logic**

* Identify last Sunday of the *previous month*
* Generate only the newly completed weeks
* Append to existing CSV without duplicates

### **4. Integration & Checks**

* Merge sales + economic indicators
* Validate data completeness
* Detect missing product-week combinations
* Detect anomalies using z-score
* Save merged dataset and anomalies file

### **5. Alerting**

* Log all issues
* Email alerts for missing data or anomalies

---

#  **7. Assumptions**

* Sales occur weekly and always on Sundays
* Only completed past weeks should be generated
* Discount impact on sales follows a fixed multiplier (0.5)
* CPI monthly data can be forward-filled for weekly alignment
* FRED API availability is stable
* Email alerts are optional depending on SMTP setup


