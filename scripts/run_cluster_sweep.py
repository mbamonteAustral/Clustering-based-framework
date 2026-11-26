"""Run main.py for cluster counts 3 through 12 and aggregate the results."""

import os
from pathlib import Path
import subprocess
import sys

import pandas as pd

CLUSTER_COUNTS = range(3, 13)
# CLUSTER_COUNTS = range(8, 10) # For testing purposes
SIGNIFICANCE_THRESHOLD = 0.05
SUMMARY_TITLE = " Summarizing Separability Analysis"


def print_run_summary(log_path):
    with log_path.open("r", encoding="utf-8", errors="replace") as log_file:
        lines = log_file.read().splitlines()

    title_index = next(
        (index for index, line in enumerate(lines) if SUMMARY_TITLE in line),
        None
    )
    if title_index is None:
        raise RuntimeError(
            f"Separability summary was not found in {log_path}"
        )

    start_index = title_index - 1
    end_index = next(
        (
            index for index in range(title_index + 2, len(lines))
            if lines[index] == "-----------------------------------------------------------"
        ),
        title_index + 1
    )
    print("\n".join(lines[start_index:end_index + 1]))


def read_pair_medians(run_dir, cluster_count):
    path = run_dir / "median_p_values_by_emotion_pair.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Expected run output was not created: {path}")

    medians = pd.read_csv(path, index_col=0)
    medians.index.name = "emotion_pair"
    medians.index = medians.index.str.replace(", ", ",", regex=False)
    medians = medians.reset_index()
    medians.insert(0, "n_clusters", cluster_count)
    return medians


def write_aggregated_results(output_dir, run_results):
    pair_tables = []
    matrix_tables = []

    for cluster_count, run_dir in run_results:
        pair_tables.append(read_pair_medians(run_dir, cluster_count))

        for dimension in ("valence", "arousal"):
            matrix_path = run_dir / f"population_level_esm_{dimension}.csv"
            if not matrix_path.is_file():
                raise FileNotFoundError(
                    f"Expected run output was not created: {matrix_path}"
                )

            matrix = pd.read_csv(matrix_path, index_col=0)
            matrix.index.name = "row_label"
            matrix_tables.append(
                matrix.reset_index().melt(
                    id_vars="row_label",
                    var_name="column_label",
                    value_name="p_value"
                ).assign(
                    n_clusters=cluster_count,
                    dimension=dimension.capitalize()
                )
            )

    pair_results = pd.concat(pair_tables, ignore_index=True)
    pair_results.to_csv(
        output_dir / "median_p_values_by_cluster.csv",
        index=False
    )
    pd.concat(matrix_tables, ignore_index=True).to_csv(
        output_dir / "population_level_esm_all_clusters.csv",
        index=False
    )

    pair_results = pair_results.set_index(
        ["n_clusters", "emotion_pair"]
    )
    pairs = pair_results.index.get_level_values("emotion_pair").unique()
    frequency = pd.DataFrame(index=pairs)
    valence_flags = pair_results["Valence"] < SIGNIFICANCE_THRESHOLD
    arousal_flags = pair_results["Arousal"] < SIGNIFICANCE_THRESHOLD
    frequency["Valence"] = valence_flags.groupby(
        level="emotion_pair"
    ).sum().reindex(pairs, fill_value=0)
    frequency["Arousal"] = arousal_flags.groupby(
        level="emotion_pair"
    ).sum().reindex(pairs, fill_value=0)
    both_flags = valence_flags & arousal_flags
    frequency["Both"] = both_flags.groupby(
        level="emotion_pair"
    ).sum().reindex(pairs, fill_value=0)
    frequency.index.name = "Emotion pair"
    frequency.to_csv(output_dir / "significant_pair_frequency.csv")

    print("\nSignificant-pair frequency across cluster counts:")
    print(frequency.to_string())
    print(
        "\nSaved aggregate outputs:\n"
        f"  {output_dir / 'population_level_esm_all_clusters.csv'}\n"
        f"  {output_dir / 'median_p_values_by_cluster.csv'}\n"
        f"  {output_dir / 'significant_pair_frequency.csv'}"
    )


def main():
    script_dir = Path(__file__).resolve().parent
    main_script = script_dir / "main.py"
    output_dir = script_dir / "cluster_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_results = []
    for cluster_count in CLUSTER_COUNTS:
        run_dir = output_dir / f"clusters_{cluster_count}"
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "main.log"
        environment = os.environ.copy()
        environment["NUMBER_OF_CLUSTERS"] = str(cluster_count)
        environment["CLUSTER_RUN_DIR"] = str(run_dir)

        print(f"\nRunning main.py with {cluster_count} clusters...")
        with log_path.open("wb") as log_file:
            completed = subprocess.run(
                [sys.executable, "-W", "ignore", "-u", str(main_script)],
                cwd=str(script_dir),
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False
            )

        if completed.returncode != 0:
            raise RuntimeError(
                f"Run for {cluster_count} clusters failed with exit code "
                f"{completed.returncode}. See {log_path}"
            )
        print_run_summary(log_path)
        run_results.append((cluster_count, run_dir))

    write_aggregated_results(output_dir, run_results)


if __name__ == "__main__":
    main()
