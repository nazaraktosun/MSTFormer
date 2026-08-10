# MFTFormer

Research code for limit-order-book movement classification with MFTFormer.
The repository contains the preprocessing, feature engineering, model training,
evaluation, and experiment configuration used for the accompanying paper.

The task is three-class classification of the future price movement:

| Class | Label | Meaning |
|---:|---|---|
| 0 | `STABLE` | The forward return remains inside the configured threshold. |
| 1 | `UP` | The forward return is above the positive threshold. |
| 2 | `DOWN` | The forward return is below the negative threshold. |

The importable Python package is named `bullsense`, which predates the MFTFormer
name and is kept to avoid churning every import path and invalidating committed
experiment configs. Read `bullsense/` as the MFTFormer source tree.

Paper results, final experiment configurations, and citation metadata will be
added with the paper release. Raw market data and generated artifacts are not
distributed in this repository.

## Features

- NumPy and Parquet ingestion for limit-order-book data
- Optional ITCH/message-flow inputs
- LOB-only, engineered-feature, and fused input pipelines
- Fixed-horizon, volatility-scaled, time-based, and barrier classification labels
- MFTFormer/TLOB, MLP-LOB, DeepLOB, original TLOB, and BiN-CTABL model variants
- Class weighting, focal loss, early stopping, and gradient clipping
- Configurable input normalization, including BiN, DAIN, RevIN, and Dish-TS
- Evaluation metrics, confusion matrices, visualizations, and markout backtests
- Optional MLflow experiment tracking

## Installation

Python 3.10 or newer is required. CUDA is recommended for training but is not
required for preprocessing or CPU experiments.

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` also installs the bundled microstructure feature package at
`libs/bullsense_features`.

## Data

The code accepts either:

- a NumPy LOB5 array containing a timestamp followed by five bid and five ask
  price/quantity levels; or
- local Parquet LOB data, optionally paired with message Parquet data.

Data locations and market-specific settings belong in the experiment YAML.
Important settings include the session timezone and hours, price scaling,
sampling interval, sequence length, forecast horizon, and class threshold.

Use `price_scale: 1.0` when Parquet prices are already expressed in normal
price units. Some source datasets store integer prices and require another
scale, such as `1000.0`.

### Preprocessing modes

| Mode | Model input |
|---|---|
| `lob` | Raw order-book levels only |
| `feature` | Engineered microstructure features only |
| `fusion` | Raw order-book levels plus configured features/message flow |

Prepared tensors and metadata are written beneath `data/processed_*`. These
outputs are ignored by Git.

## Microstructure features

Feature functions register themselves by name when their module is imported.
`bullsense/features/external_hft_features.py` holds the canonical import list,
and that list determines which names a configuration may reference. Nothing is
imported by wildcard, so experimental and non-causal features cannot reach a
pipeline by accident.

Configurations name features and their parameters:

```yaml
feature:
  orderbook_features:
    - { name: microprice }
    - { name: micro_return_features, params: { windows: [1, 5, 10] } }
    - { name: obi_cumulative, params: { max_level: 5 } }
```

| Feature block | Registry name | Emits |
|---|---|---|
| Order-book levels | — (raw ingest) | `pb1..pbL`, `qb1..qbL`, `pa1..paL`, `qa1..qaL` |
| ITCH order-lifecycle events | `event_lifecycle_features` | `f_evt_sqty_*`, `f_evt_sratio_*`, `f_evt_cnt_*`, `f_evt_freq_*`, `f_evt_ctr_*`, `f_evt_cancel_rate_*` |
| Depth imbalance, L ∈ {1,3,5} | `obi_cumulative` | `f_obi_1` … `f_obi_5` |
| Order-flow imbalance | `ofi_hybrid`, `f_rolling_ofi_hybrid` | `ofi_hybrid`, `f_rolling_ofi_hybrid` |
| Microprice and its returns | `microprice`, `micro_return_features` | `f_microprice`, `f_micro_ret_bps_{W}` |
| Time of day, cyclic | `time_of_day_cyclical` | `f_tod_sin`, `f_tod_cos` |
| Book-state staleness | `time_since_update` | `f_dt_since_update`, `f_dt_since_update_log` |
| Order-queue pressure | `queue_features_all` | `f_qdepl_exec_{bid,ask}_N`, `f_qdepl_cxl_{bid,ask}_N`, `f_qdepl_imb_net_N`, `f_qtouch_flip_N` |
| Schema bridge | `msgf_to_message_schema` | `message_type`, `side`, `timestamp`, `price`, `quantity` |

Implementations live in `libs/bullsense_features/hft_features/categories/`.

### Order-lifecycle reconstruction

`event_lifecycle_features` rebuilds the five native categories
`{add, cancel, delete, replace, execute}`. Per category and window it emits the
side-signed quantity `Q_bid − Q_ask` (bid-heavy is positive), a scale-free ratio
form of it, the event count, and that count's share of all retained events. The
cancellation-to-execution ratio is computed on unsigned quantities, both as the
raw ratio and as a bounded rate for windows where executed quantity is zero.

Partial size reductions (`cancel`) are separated from complete removals
(`delete`) with a causal per-order remaining-size ledger: an order's resting size
before an event is everything added to it minus everything removed from it,
strictly at or before that event. A withdrawal that takes the remaining size to
zero is a `delete`; one that leaves size behind is a `cancel`. The tempting test,
whether an order id reappears later, is a look-ahead and is not used. Orders
whose opening add predates the ingest window are treated as complete removals.

`replace` comes from the feed's native `OrderModify` action. Reconstructing it by
pairing delete-and-add sequences is available via `pair_delete_add=True`, but
labelling the delete leg requires reading one event ahead, so it is off by
default.

### Event-grid requirement

`event_lifecycle_features` requires `order_id` and `queue_features_all` requires
per-event rows. Both raw per-event fields are dropped on resample, so these
blocks must run before resampling; handed a resampled frame they raise rather
than silently producing wrong values. `queue_features_all` and
`order_activity_features` were written against a standalone message frame, so
insert `msgf_to_message_schema` ahead of them to supply the columns they expect.

### Causality

Every rolling feature aggregates a trailing window ending at the current row, and
every window is scoped to a calendar day, so no value depends on future
information and none reaches across the overnight gap. Lookback returns shift
within a day, yielding null at a session's opening rows rather than a return
measured against the previous close. Train, validation, and test splits are
day-aware, so a single day never lands on both sides of a split. The upstream
`lead_lag_features` module, whose `lead_ret_1s` and `lead_ret_5s` read forward in
time, is deliberately not vendored.

## Prepare a classification dataset

The following example uses the committed INTC fusion-classification config.
The command-line input path overrides the path stored in the YAML:

```bash
python scripts/prepare_data.py \
  --config configs/experiment/us_intc_fusion_cls.yaml \
  --mode fusion \
  --lob-parquet /path/to/INTC \
  --tag intc
```

To include message data, enable the relevant message features in the selected
configuration and provide the paired input:

```bash
python scripts/prepare_data.py \
  --config path/to/classification_config.yaml \
  --mode fusion \
  --lob-parquet /path/to/lob \
  --msg-parquet /path/to/messages \
  --tag experiment_name
```

Run the following command for all preprocessing options:

```bash
python scripts/prepare_data.py --help
```

## Train a classifier

```bash
python scripts/train_mlplob.py \
  --config configs/experiment/us_intc_fusion_cls.yaml \
  --data-dir data/processed_fusion/intc \
  --output-dir runs/us_intc_fusion
```

The YAML configuration is the experiment definition. Classification configs
must set:

```yaml
task: classification

model:
  num_classes: 3
```

The same file controls the random seed, chronological train/validation/test
split, labeling rule, feature set, architecture, optimizer, class balancing,
device, and output locations. Keep paper experiments as committed YAML files
under `configs/experiment/` so runs can be reconstructed without shell history.

## Experiment tracking

Enable MLflow in an experiment config:

```yaml
tracking:
  enable_mlflow: true
  tracking_uri: file:./mlruns
  experiment_name: mftformer
```

Inspect local runs with:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Checkpoints, metrics, plots, and MLflow state are generated artifacts and are
excluded from version control.

## Repository layout

```text
.
├── bullsense/
│   ├── config/       # Typed experiment configuration
│   ├── data/         # Dataset preparation and sequence construction
│   ├── eval/         # Markout evaluation
│   ├── features/     # Feature registry and pipeline integration
│   ├── io/           # NumPy, Parquet, and database ingestion
│   ├── labeling/     # Classification label generation
│   ├── model/        # MFTFormer/TLOB and baseline models
│   ├── tracking/     # MLflow integration
│   └── training/     # Training and evaluation loops
├── configs/experiment/      # Versioned experiment definitions
├── libs/bullsense_features/ # Bundled microstructure feature library
├── scripts/                 # Preprocessing and training entry points
└── tests/                   # Automated tests
```

## Tests

```bash
pytest -q
```

## Paper repository policy

Commit source code, tests, frozen classification configs, final manuscript
source, and release-ready figures. Do not commit raw or processed data,
checkpoints, experiment logs, notebooks, personal notes, working drafts, or
LaTeX build products; these are covered by `.gitignore`.

## Citation

Citation information will be added when the MFTFormer paper is available.
