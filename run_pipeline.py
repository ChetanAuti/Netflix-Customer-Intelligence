"""Run the whole pipeline end-to-end:  python run_pipeline.py [--skip-eda] [--skip-leakage]"""
import argparse
import time

from database.load_data import load_sqlite
from src.data_processing import clean
from src.features import build_features
from src.models import leakage_check, train
from src.utils import eda
from src.utils.config import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--skip-eda", action="store_true")
ap.add_argument("--skip-leakage", action="store_true")
a = ap.parse_args()

for name, fn in [("1/6 clean data", clean.run), ("2/6 build features", build_features.run), ("3/6 train models", train.run),
                 ("4/6 load SQLite demo DB", lambda: load_sqlite(ROOT / "data" / "netflix.db")),
                 ("5/6 EDA figures", (lambda: None) if a.skip_eda else eda.run),
                 ("6/6 forward-looking leakage check", (lambda: None) if a.skip_leakage else leakage_check.run)]:
    t0 = time.time()
    print(f"==> {name}")
    fn()
    print(f"    done in {time.time() - t0:.0f}s")
print("\nNext:  streamlit run app/dashboard.py")
