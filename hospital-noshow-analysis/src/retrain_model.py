import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import data_prep as dp

SEED = 42
df = pd.read_csv(ROOT / "data" / "processed" / "appointments_clean.csv")
NUM = dp.NUMERIC_FEATURES + dp.BINARY_FEATURES
CAT = dp.CATEGORICAL_FEATURES
FEATURES = NUM + CAT
X, y = df[FEATURES], df["missed"]

metrics = json.loads((ROOT / "reports" / "model_metrics.json").read_text())
params = metrics["best_params"]
thr = metrics["threshold_table"]["Best F1"]["threshold"]

pre = ColumnTransformer(
    [("nbh", TargetEncoder(target_type="binary", smooth="auto", random_state=SEED), CAT),
     ("num", "passthrough", NUM)],
    verbose_feature_names_out=False,
).set_output(transform="pandas")

xgb = XGBClassifier(**params, scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
                    eval_metric="logloss", n_jobs=1, random_state=SEED)
pipe = Pipeline([("pre", pre), ("m", xgb)])

print("Training (this takes a minute or two) ...")
model = CalibratedClassifierCV(pipe, method="isotonic", cv=5).fit(X, y)

out = ROOT / "models" / "noshow_model.joblib"
out.parent.mkdir(exist_ok=True)
joblib.dump({"model": model, "features": FEATURES, "num": NUM, "cat": CAT, "threshold_f1": float(thr)}, out)
print("Saved", out)