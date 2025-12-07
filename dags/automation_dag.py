"""
monthly_sales_automation_dag.py

Airflow DAG to run the monthly sales + economic data automation pipeline.
This DAG executes the automate.py script on the 1st of every month at 06:00.
"""

from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator
import os

# ---------------------------------------------------------------------------
# Project Paths (adjust if r Airflow environment uses another directory)
# ---------------------------------------------------------------------------
PROJECT_DIR = "/opt/airflow/folder_project"  
# If running locally with Docker, map your project into this path
# Example docker-compose mount:
#   - ./FOLDER_PROJECT:/opt/airflow/folder_project

SCRIPT_PATH = os.path.join(PROJECT_DIR, "automate.py")

# ---------------------------------------------------------------------------
# Default DAG arguments
# ---------------------------------------------------------------------------
default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
}

# ---------------------------------------------------------------------------
# DAG Definition
# ---------------------------------------------------------------------------
with DAG(
    dag_id="monthly_sales_automation",
    default_args=default_args,
    schedule_interval="0 6 1 * *",  # Run at 06:00 on the 1st of every month
    catchup=False,
    description="Runs monthly sales + FRED economic data automation pipeline",
) as dag:

    run_pipeline = BashOperator(
        task_id="run_automation_script",
        bash_command=f"python {SCRIPT_PATH}",
    )

    run_pipeline
