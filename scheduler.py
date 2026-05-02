"""
scheduler.py - Automatic batch prediction scheduler

Uses APScheduler to run batch_predict.run_batch() every 5 minutes.

Run with:
    python scheduler.py
"""

from apscheduler.schedulers.blocking import BlockingScheduler
from batch_predict import run_batch
from datetime import datetime

INTERVAL_MINUTES = 5

scheduler = BlockingScheduler()

@scheduler.scheduled_job("interval", minutes=INTERVAL_MINUTES)
def scheduled_job():
    run_batch()

if __name__ == "__main__":
    print(f"Scheduler started — running batch prediction every {INTERVAL_MINUTES} minutes.")
    print(f"First run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Press Ctrl+C to stop.\n")

    # Run once immediately on start
    run_batch()

    # Then run on schedule
    scheduler.start()
