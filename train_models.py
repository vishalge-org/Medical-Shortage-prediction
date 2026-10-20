"""Command-line training:  python train_models.py [path/to/shortages_raw.csv]
With no argument it uses data/shortages_raw.csv, or downloads live from openFDA if that file is absent."""
import sys
from pathlib import Path
import pipeline as pl

HERE = Path(__file__).parent
if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "data" / "shortages_raw.csv"
    try:
        if src.exists():
            raw, name = pl.load_raw(src), src.name
        else:
            print(f"{src} not found - downloading from openFDA ...")
            raw, name = pl.fetch_live(lambda f, m: print(f"  {m}")), "openFDA live download"
            raw.to_csv(HERE / "data" / "shortages_raw.csv", index=False)
        rep = pl.run_all(raw, HERE / "models", name, progress=lambda f, m: print(f"[{f:4.0%}] {m}"))
    except pl.DataError as e:
        sys.exit(f"ERROR: {e}")
    print("\nBest early-warning model :", rep["early_warning"]["best"])
    print("Best status classifier   :", rep["status_classification"]["best"])
    print("Best forecaster          :", rep["forecast"].get("best"))
    print("Saved artefacts to", HERE / "models")
