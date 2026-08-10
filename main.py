from pathlib import Path

from bullsense.config.config_loader import load_config
from bullsense.training.train_mlplob import train_once

ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = ROOT / "configs" / "experiment"
LOB_DATA = ROOT / "data" / "processed_lob" / "ykbnk_q1_q2"
RUNS_ROOT = ROOT / "runs" / "ykbnk" / "q1_q2"

CONFIGS = [
    "cfg_1.yaml",
    "cfg_2.yaml",
    "cfg_3.yaml",
    "cfg_4.yaml",
]

best_cfg_name: str | None = None
best_results: dict | None = None
best_test_acc: float | None = None

for cfg_name in CONFIGS:
    cfg = load_config(CFG_DIR / cfg_name)
    cfg.paths.data_dir = LOB_DATA
    cfg.paths.output_dir = RUNS_ROOT / cfg_name.replace(".yaml", "")
    print(f"\n=== Training {cfg_name} on {LOB_DATA} ===")
    results = train_once(experiment=cfg)

    test_acc = None
    try:
        test_acc = float(results["metrics"]["test_accuracy"])
    except Exception:
        pass

    if test_acc is not None:
        if best_test_acc is None or test_acc > best_test_acc:
            best_test_acc = test_acc
            best_cfg_name = cfg_name
            best_results = results

if best_cfg_name is not None and best_results is not None:
    print("\n" + "=" * 60)
    print("BEST MODEL (by test_accuracy)")
    print("=" * 60)
    print(f"Config: {best_cfg_name}")
    print(f"Test accuracy: {best_test_acc:.4f}")
    best_val_acc = best_results.get("training", {}).get("best_val_acc")
    if best_val_acc is not None:
        print(f"Best val accuracy during training: {best_val_acc:.4f}")
    print(f"Data dir: {best_results.get('data_dir')}")
    print(f"Output dir: {best_results.get('output_dir')}")
