# A Clustering-based Framework for Assessing Emotional Separability in Combined Galvanic Skin Response and Photoplethysmography Feature Representations

This repository contains two related Python workflows:

- `data/` contains the CASE-data preprocessing scripts and the feature/label pickle files for participants 1–30.
- `scripts/` contains the clustering and population-level analysis scripts.

The included `data/xf*.pkl` and `data/y_median*.pkl` files can be used to test the clustering method without downloading the raw CASE dataset. To preprocess the raw data yourself, the CASE Dataset is publicly available at [https://gitlab.com/karan-shr/case_dataset](https://gitlab.com/karan-shr/case_dataset).

The pickle files contain extracted physiological features and valence/arousal labels. Before making this repository public or reusing the files, make sure sharing and redistribution are permitted by the dataset's terms and participant-consent requirements.

## Requirements

Use a Python version supported by TensorFlow and the other dependencies in your environment. A virtual environment is recommended:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy pandas scipy scikit-learn imbalanced-learn matplotlib seaborn neurokit2 heartpy tensorflow umap-learn scikit-posthocs openpyxl
```

TensorFlow installation can depend on your operating system and hardware; use the TensorFlow installation instructions for your platform if the command above is not suitable.

## CASE preprocessing

[`data/main.py`](data/main.py) calls `process_data()` for participants 1–30, reads CASE CSVs, and writes `xf<participant>.pkl` and `y_median<participant>.pkl` into `data/`.

1. Download/extract the CASE data. The expected files include:
   - `interpolated/physiological/sub_<participant>.csv`
   - `interpolated/annotations/sub_<participant>.csv`
   - `metadata/seqs_order_num_csv.csv`
2. In [`data/processdataf.py`](data/processdataf.py), edit the dataset configuration near the top:
   - Lines 14–19: Windows default (`WINDOWS_DATASET_CONFIG`).
   - Lines 20–25: Linux default (`LINUX_DATASET_CONFIG`).
   - Lines 26–36: optional Windows computer-name overrides (`HOST_DATASET_CONFIGS`).
3. Set `root` to the directory containing the dataset folders. Set `data_subdir`, `interpolated_subdir`, and `metadata_subdir` to match your directory layout. For example, if the files are under `D:\Datasets\CASE\interpolated\...` and `D:\Datasets\CASE\metadata\...`, use `root = Path(r"D:\Datasets\CASE")`, an empty `data_subdir`, `interpolated_subdir = Path("interpolated")`, and `metadata_subdir = Path("metadata")`.
4. Run from the repository root:

   ```powershell
   cd data
   python main.py
   ```

To change the participant range, edit the `participants` assignment in `data/main.py`. Generated pickle files are written into `data/` and can replace existing files with the same names.

## Clustering and population analysis

The scripts under `scripts/` read the `xf<participant>.pkl` and `y_median<participant>.pkl` files from `data/`. In [`scripts/processdataf.py`](scripts/processdataf.py), `PICKLE_DIR` near the top points to this folder by default. Change it if you keep the pickle files elsewhere.

Run commands from `scripts`:

```powershell
cd scripts
python main.py
```

`main.py` analyzes participants 1–30 with the cluster count from the `NUMBER_OF_CLUSTERS` environment variable (3 by default; see `definitions.py`). To run the cluster sweep from 3 through 12 and aggregate its results, run:

```powershell
python run_cluster_sweep.py
```

Alternatively, run `batch.bat` from this folder. It starts the sweep, writes a log, then runs `show_global_results.py`. Sweep outputs are saved under `cluster_sweep/`.

The Python files and `batch.bat` are in `scripts/`; the `cluster_sweep/` output folder and its logs are not included.
