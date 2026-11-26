# Clustering-based framework

This repository contains two related Python workflows:

- The repository root contains CASE-data preprocessing scripts and the generated feature/label pickle files for participants 1–30.
- `seminal_script/in_progress/` contains the clustering and population-level analysis scripts.

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

The root-level `main.py` calls `process_data()` for participants 1 and 2, reads CASE CSVs, and writes `xf<participant>.pkl` and `y_median<participant>.pkl` next to the scripts.

1. Download/extract the CASE data. The expected files include:
   - `interpolated/physiological/sub_<participant>.csv`
   - `interpolated/annotations/sub_<participant>.csv`
   - `metadata/seqs_order_num_csv.csv`
2. In the root-level [`processdataf.py`](processdataf.py), edit the dataset configuration near the top:
   - Lines 14–19: Windows default (`WINDOWS_DATASET_CONFIG`).
   - Lines 20–25: Linux default (`LINUX_DATASET_CONFIG`).
   - Lines 26–36: optional Windows computer-name overrides (`HOST_DATASET_CONFIGS`).
3. Set `root` to the directory containing the dataset folders. Set `data_subdir`, `interpolated_subdir`, and `metadata_subdir` to match your directory layout. For example, if the files are under `D:\Datasets\CASE\interpolated\...` and `D:\Datasets\CASE\metadata\...`, use `root = Path(r"D:\Datasets\CASE")`, an empty `data_subdir`, `interpolated_subdir = Path("interpolated")`, and `metadata_subdir = Path("metadata")`.
4. Run from the repository root:

   ```powershell
   python main.py
   ```

To select other participants, edit line 26 (`participants = np.arange(1, 3)`) in the root-level `main.py`. Generated pickle files are written into the repository root and can replace existing files with the same names.

## Clustering and population analysis

The scripts under `seminal_script/in_progress/` read the `xf<participant>.pkl` and `y_median<participant>.pkl` files. In [`seminal_script/in_progress/processdataf.py`](seminal_script/in_progress/processdataf.py), update the pickle directory in both Windows branches if your files are elsewhere:

- Lines 76–77: the `LAPTOP-R7AHG17P` branch.
- Lines 83–84: the other Windows-computer branch.

Change the directory string in each `pd.read_pickle(os.path.join(...))` call to the folder containing the matching `xf*.pkl` and `y_median*.pkl` files. The repository's copies are in its root folder. This is separate from the raw CASE CSV location configured in the root-level `processdataf.py`.

Run commands from `seminal_script\in_progress`:

```powershell
cd seminal_script\in_progress
python main.py
```

`main.py` analyzes participants 1–30 with the cluster count from the `NUMBER_OF_CLUSTERS` environment variable (3 by default; see `definitions.py`). To run the cluster sweep from 3 through 12 and aggregate its results, run:

```powershell
python run_cluster_sweep.py
```

Alternatively, run `batch.bat` from this folder. It starts the sweep, writes a log, then runs `show_global_results.py`. Sweep outputs are saved under `cluster_sweep/`.

The Python files in `seminal_script/in_progress/` are included as requested; the `cluster_sweep/` output folder and its logs are not included.
