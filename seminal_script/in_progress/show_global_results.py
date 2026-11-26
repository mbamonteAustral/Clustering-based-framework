"""
Show global simulated results
"""
#%% --------------------------------------------------
from itertools import combinations_with_replacement
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations_with_replacement, combinations

# Sources (Authors' access only)

# 1) ML experiments planning - pc (Mind map)
# https://drive.mindmup.com/map/1hmYMXRqlrz2mcqbli2P8Ys0-OxRV-5rz

# 2) ML experiments planning - pc.xlsx (summary of results - Authors' access only)


#%%----------------------------------------------------------------

# Study the effects of different window duration sizes and percentages of overlap.

# bipartite scheme
# classic (threshold = 5)

# Data sources
# ML experiments planning - pc.xlsx (summary of results - Authors' access only)

# runs:
# c129_1
# c129_2
# c129_3
# c129_4
# c129_5
# c129_6
# c129_7
# c129_8
# c129_9
# c129_9_1
# c129_10
# c129_11
# c130
# c130_1
# c130_2
# c130_3
# c130_4
# c130_5
# c130_6
# c130_6_1
# c130_7
# c130_7_1
# c130_8
# c130_9
# c130_10
# c130_11
# c130_12
# c130_13
# c130_14

# results form clusters from 3 to 12 (inclusive):

R1_R2_val_bin5 = [4.293653e-05, 2.761327e-01, 8.320290e-01, 5.362160e-01, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]
R1_R2_aro_bin5 = [2.622981e-15, 4.921977e-09, 3.611562e-08, 2.470890e-05, 6.067984e-05, 2.177934e-04, 3.149164e-05, 2.270000e-04, 1.290000e-04, 5.000000e-06]

R1_R3_val_bin5 = [4.018670e-02, 1.803485e-02, 1.003820e-01, 5.065360e-01, 5.800000e-05, 1.229130e-01, 1.013090e-01, 1.000000e+00, 2.546500e-01, 4.116930e-01]
R1_R3_aro_bin5 = [1.000000e+00, 5.623809e-01, 1.000000e+00, 7.753450e-01, 2.619263e-01, 7.553669e-01, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]

R1_R4_val_bin5 = [5.113867e-06, 1.228456e-10, 2.287000e-03, 1.724300e-02, 2.613510e-01, 4.621190e-01, 5.062360e-01, 3.904740e-01, 6.162510e-01, 5.008480e-01]
R1_R4_aro_bin5 = [1.115543e-02, 1.562119e-09, 1.139893e-02, 6.253263e-02, 2.298057e-01, 5.003091e-01, 5.000428e-01, 5.044590e-01, 4.354870e-01, 5.143970e-01]

R1_R5_val_bin5 = [1.000000e+00, 6.078311e-01, 1.000000e+00, 1.000000e+00, 8.017820e-01, 1.000000e+00, 1.000000e+00, 9.950070e-01, 1.000000e+00, 1.000000e+00]
R1_R5_aro_bin5 = [5.412065e-04, 6.422391e-04, 5.000000e-01, 7.507366e-01, 5.009958e-01, 2.686880e-01, 3.363017e-01, 5.575170e-01, 1.190270e-01, 3.604380e-01]

R2_R3_val_bin5 = [1.230069e-09, 2.410978e-08, 5.100000e-05, 6.800000e-05, 1.700000e-05, 4.700000e-05, 2.980000e-04, 2.929000e-03, 4.286000e-03, 2.063300e-02]
R2_R3_aro_bin5 = [1.169531e-12, 6.902462e-11, 2.206088e-09, 6.674165e-07, 1.546323e-08, 7.913209e-08, 4.519936e-07, 2.000000e-06, 4.000000e-06, 1.000000e-05]

R2_R4_val_bin5 = [4.063406e-08, 1.479750e-06, 4.121000e-03, 2.370000e-04, 1.780000e-04, 3.200000e-04, 4.645000e-03, 5.065000e-03, 2.831000e-03, 9.306000e-03]
R2_R4_aro_bin5 = [1.387884e-07, 3.159457e-04, 4.537914e-02, 6.611998e-02, 1.441998e-01, 1.658623e-01, 5.991841e-01, 5.825630e-01, 7.857240e-01, 1.000000e+00]

R2_R5_val_bin5 = [3.296347e-05, 2.006514e-02, 3.768000e-02, 3.772260e-01, 5.333180e-01, 5.635860e-01, 7.060060e-01, 6.880330e-01, 7.224600e-01, 1.000000e+00]
R2_R5_aro_bin5 = [1.950261e-08, 1.707611e-03, 3.192486e-03, 6.851033e-04, 2.122624e-03, 1.544152e-03, 1.226756e-02, 2.675000e-02, 4.197900e-02, 4.978800e-02]

R3_R4_val_bin5 = [2.392555e-01, 5.041036e-01, 4.396670e-01, 7.228400e-01, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]
R3_R4_aro_bin5 = [1.409109e-11, 3.266098e-05, 5.302045e-04, 1.256864e-03, 9.472906e-03, 1.511599e-02, 2.501974e-02, 7.269500e-02, 1.904360e-01, 1.172300e-01]

R3_R5_val_bin5 = [1.995491e-01, 6.927786e-01, 6.045980e-01, 1.820120e-01, 5.236940e-01, 6.709540e-01, 7.126890e-01, 1.000000e+00, 1.000000e+00, 1.000000e+00]
R3_R5_aro_bin5 = [5.003965e-01, 1.705866e-01, 5.000000e-01, 7.249525e-01, 4.965181e-01, 1.000000e+00, 6.135254e-01, 7.381920e-01, 8.548530e-01, 1.000000e+00]

R4_R5_val_bin5 = [7.536572e-02, 2.584255e-03, 3.559900e-02, 4.402030e-01, 5.887880e-01, 5.366540e-01, 9.354160e-01, 5.868580e-01, 7.479610e-01, 1.000000e+00]
R4_R5_aro_bin5 = [2.780583e-02, 9.178084e-02, 6.720578e-01, 1.000000e+00, 7.092493e-01, 5.894500e-01, 8.348972e-01, 1.000000e+00, 1.000000e+00, 1.000000e+00]


#---------------
# está mal cargado el orden...

# results form clusters from 4 to 12 (inclusive):
A_B_val_bin9 = [2.516349e-01, 5.447510e-02, 0.345986, 3.670003e-01, 0.311043, 0.242747, 0.529203, 0.786214, 1.000000]
A_R_val_bin9 = [1.419855e-01, 5.000164e-01, 0.455720, 3.486260e-01, 0.679520, 0.630907, 1.000000, 1.000000, 1.000000]
A_S_val_bin9 = [3.340099e-10, 1.355186e-07, 0.000026, 8.244955e-04, 0.000105, 0.000047, 0.000161, 0.001201, 0.001759]
B_R_val_bin9 = [5.000000e-01, 5.000000e-01, 0.000079, 1.486827e-02, 0.056115, 0.253381, 0.500677, 0.505624, 0.202765]
B_S_val_bin9 = [7.504138e-01, 7.521294e-01, 0.933469, 7.112682e-01, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000]
R_S_val_bin9 = [1.700637e-12, 1.580985e-11, 0.000001, 4.097342e-07, 0.000002, 0.000002, 0.000018, 0.000096, 0.000104]

A_B_aro_bin9 = [1.112431e-09, 2.086399e-19, 3.289111e-08, 1.567156e-03, 3.423531e-05, 4.744679e-02, 6.722462e-04, 1.988511e-06, 6.498764e-08]
A_R_aro_bin9 = [2.035619e-01, 2.689329e-03, 3.761693e-07, 1.235060e-06, 4.851642e-04, 1.202642e-03, 9.883518e-03, 3.035167e-02, 5.404791e-02]
A_S_aro_bin9 = [7.263956e-06, 1.942815e-04, 4.530898e-03, 7.030167e-02, 8.539843e-03, 1.624359e-02, 7.951928e-02, 7.203766e-02, 3.928538e-01]
B_R_aro_bin9 = [9.621523e-02, 2.763565e-01, 5.000000e-01, 5.018043e-01, 1.000000e+00, 8.235819e-01, 1.000000e+00, 1.000000e+00, 1.000000e+00]
B_S_aro_bin9 = [2.600566e-16, 8.891339e-02, 1.121427e-11, 1.182418e-07, 3.003994e-09, 1.036573e-09, 5.644429e-10, 5.717496e-09, 5.977417e-08]
R_S_aro_bin9 = [7.425833e-13, 2.465490e-14, 1.184448e-10, 2.360098e-10, 2.971808e-08, 2.157348e-07, 2.497046e-08, 2.161612e-08, 3.315778e-05]


#------------
# results form clusters from 4 to 12 (inclusive):
R1_R2_val_bin4 = [1.000000e+00, 9.408807e-01, 5.458282e-01, 2.610525e-01, 9.669228e-01, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]
R1_R2_aro_bin4 = [5.929980e-18, 2.414294e-10, 6.492007e-11, 1.241124e-12, 1.027171e-10, 1.250559e-09, 4.352235e-09, 1.629267e-08, 3.317667e-08]

R1_R3_val_bin4 = [4.217761e-29, 4.887842e-24, 1.123949e-07, 2.500000e-01, 3.529119e-03, 1.568625e-07, 6.250000e-04, 1.380500e-02, 8.931794e-03]
R1_R3_aro_bin4 = [1.000000e+00, 5.833313e-01, 5.000010e-01, 4.022811e-01, 4.951545e-01, 6.443930e-01, 1.000000e+00, 1.000000e+00, 1.000000e+00]

R1_R4_val_bin4 = [2.182273e-25, 5.943622e-17, 2.505666e-05, 9.875872e-03, 3.765015e-02, 7.379789e-02, 4.548130e-01, 5.000030e-01, 3.137368e-01]
R1_R4_aro_bin4 = [2.443148e-02, 7.524014e-13, 1.925117e-01, 2.333153e-04, 1.098458e-03, 3.591899e-04, 7.371162e-03, 4.568444e-03, 1.746094e-03]

R2_R3_val_bin4 = [1.308285e-13, 1.503412e-11, 1.662106e-10, 1.983961e-10, 1.207620e-08, 1.572415e-08, 3.000000e-06, 4.000000e-06, 2.198693e-07]
R2_R3_aro_bin4 = [1.169899e-18, 7.065928e-17, 2.179497e-22, 9.759122e-23, 3.946877e-16, 1.425780e-14, 2.031238e-13, 1.059463e-13, 6.947619e-13]

R2_R4_val_bin4 = [2.444577e-12, 4.293237e-06, 8.649183e-07, 5.016830e-06, 3.146473e-05, 6.354053e-05, 4.350000e-04, 1.733000e-03, 2.054394e-03]
R2_R4_aro_bin4 = [2.544960e-06, 1.875593e-05, 3.121538e-04, 2.080576e-04, 7.473294e-03, 3.953350e-02, 8.353845e-03, 8.392691e-02, 9.870408e-03]

R3_R4_val_bin4 = [9.193512e-02, 2.609371e-01, 1.396855e-01, 4.757950e-01, 8.799413e-01, 8.604215e-01, 1.000000e+00, 1.000000e+00, 1.000000e+00]
R3_R4_aro_bin4 = [5.702161e-05, 4.244465e-06, 2.524326e-11, 7.498730e-08, 1.606342e-06, 2.036255e-05, 3.803957e-05, 3.322632e-04, 5.220901e-03]

# ---------------------------------------------------------
# PARAMETERS
# ---------------------------------------------------------

# nbr_clusters = [4, 5, 6, 7, 8, 9, 10, 11, 12]
# nbr_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] % with 2 clusters, the are no significant emotion pairs (p < 0.05) for most of the emotion pairs, so we can exclude it from the plots.
nbr_clusters = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


# Select bin size
bin_size = 5

if bin_size == 5:
    emotion_pairs = [
    'R1_R2', 'R1_R3', 'R1_R4', 'R1_R5',
    'R2_R3', 'R2_R4', 'R2_R5',
    'R3_R4', 'R3_R5', 'R4_R5'
]
elif bin_size == 9:

    labels = ['A', 'B', 'R', 'S']

    emotion_pairs = [
        f"{a}_{b}"
        for a, b in combinations(labels, 2)
    ]  # all combinations with repetition (A,A, A,B, …, S,S),

elif bin_size == 4:
    labels = ['R1', 'R2', 'R3', 'R4'] # R1 = LVLA, R2 = LVHA, R3 = HVLA, R4 = HVHA

    emotion_pairs = [
        f"{a}_{b}"
        for a, b in combinations(labels, 2)
    ]  # all combinations with repetition (LVHA,LVHA, LVHA,LVLA, …, HVLA,HVLA),

selected_pairs = emotion_pairs
# selected_pairs = ['R1_R4', 'R2_R3']   
# selected_pairs = ['R1_R5', 'R2_R5', 'R3_R5', 'R4_R5']  # pairs containing R5

# ---------------------------------------------------------
# DATA DICTIONARIES
# ---------------------------------------------------------

valence = {
    pair: globals()[f'{pair}_val_bin{bin_size}'] for pair in selected_pairs
}

arousal = {
    pair: globals()[f'{pair}_aro_bin{bin_size}'] for pair in selected_pairs
}

cluster_sweep_dir = Path(
    os.environ.get(
        "CLUSTER_SWEEP_DIR",
        str(Path(__file__).resolve().parent / "cluster_sweep")
    )
).expanduser()
cluster_sweep_results = cluster_sweep_dir / "median_p_values_by_cluster.csv"
figure_output_dir = Path.cwd()
if not cluster_sweep_results.is_file():
    raise FileNotFoundError(
        f"Cluster-sweep results not found: {cluster_sweep_results}. "
        "Run run_cluster_sweep.py first or set CLUSTER_SWEEP_DIR."
    )

sweep_data = pd.read_csv(cluster_sweep_results)
required_columns = {"n_clusters", "emotion_pair", "Valence", "Arousal"}
missing_columns = required_columns.difference(sweep_data.columns)
if missing_columns:
    raise ValueError(
        f"{cluster_sweep_results} is missing columns: "
        f"{', '.join(sorted(missing_columns))}"
    )
if sweep_data.empty:
    raise ValueError(f"No cluster-sweep results found in {cluster_sweep_results}")

if sweep_data.duplicated(["n_clusters", "emotion_pair"]).any():
    raise ValueError(
        f"Duplicate cluster/emotion-pair rows found in {cluster_sweep_results}"
    )

nbr_clusters = sorted(sweep_data["n_clusters"].unique().tolist())
valence = {}
arousal = {}
for emotion_pair, pair_data in sweep_data.groupby("emotion_pair"):
    pair_labels = str(emotion_pair).strip("()").split(",")
    if pair_labels[0] == pair_labels[1]:
        continue
    pair_name = "_".join(pair_labels)
    pair_data = pair_data.set_index("n_clusters").reindex(nbr_clusters)
    valence[pair_name] = pair_data["Valence"].to_numpy()
    arousal[pair_name] = pair_data["Arousal"].to_numpy()
figure_output_dir = cluster_sweep_results.parent / "global_figures"
figure_output_dir.mkdir(parents=True, exist_ok=True)
print(f"Loaded cluster-sweep results from {cluster_sweep_results}")
print(f"Cluster counts: {nbr_clusters}")


# ---------------------------------------------------------
# PLOTTING FUNCTION
# ---------------------------------------------------------

def plot_p_values(data, title, ymin=None, ymax=None, fig_name='p_values_vs_n_clusters'):

    plt.figure(figsize=(10, 6))

    for pair, values in data.items():

        # Explicitly mask NaN values
        values = np.ma.masked_invalid(values)

        plt.plot(
            nbr_clusters,
            values,
            'o:',
            label=pair.replace('_', '-')
        )

    # Significance threshold
    plt.axhline(
        y=0.05,
        linestyle='--',
        linewidth=1.5,
        label='p = 0.05'
    )

    plt.xlabel('Number of clusters')
    plt.ylabel('Median p-value')
    plt.title(title)

    plt.yscale('log')
    plt.xticks(nbr_clusters)

    plt.grid(
        True,
        which='both',
        linestyle='--',
        alpha=0.4
    )

    plt.legend()
    plt.tight_layout()

    if ymin is not None or ymax is not None:
        plt.ylim(ymin, ymax)

    plt.savefig(fig_name +'.jpg', dpi=600)



# ---------------------------------------------------------
# AROUSAL
# ---------------------------------------------------------

# plot_p_values(
#     arousal,
#     'Arousal: Median p-value vs. Number of Clusters',
#     fig_name='arousal_median_p_values_vs_n_clusters_bin' + str(bin_size)
# )

plot_p_values(
    arousal,
    '',
    fig_name=str(
        figure_output_dir
        / f'arousal_median_p_values_vs_n_clusters_bin{bin_size}'
    )
)


# ---------------------------------------------------------
# VALENCE
# ---------------------------------------------------------

plot_p_values(
    valence,
    '',
    fig_name=str(
        figure_output_dir
        / f'valence_median_p_values_vs_n_clusters_bin{bin_size}'
    )
)

# plot_p_values(
#     valence,
#     'Valence: Median p-value vs. Number of Clusters',
#     fig_name='valence_median_p_values_vs_n_clusters_bin' + str(bin_size)
# )

# ---------------------------------------------------------
# BINARIZE p-VALUES AND COUNT SIGNIFICANT EMOTION PAIRS
# ---------------------------------------------------------

# p < 0.05 -> 1
# p >= 0.05 (including NaN) -> 0

def binarize_p_values(data):
    return {
        pair: (np.asarray(values) < 0.05).astype(int)
        for pair, values in data.items()
    }


# Binarized dictionaries for the currently selected bin size
valence_binary = binarize_p_values(valence)
arousal_binary = binarize_p_values(arousal)


# # Also create binarized versions of every p-value list defined above.
# # The original p-value lists are kept unchanged.
# for variable_name, variable_value in list(globals().items()):
#     if (
#         (variable_name.endswith('_val_bin5') or
#          variable_name.endswith('_aro_bin5') or
#          variable_name.endswith('_val_bin9') or
#          variable_name.endswith('_aro_bin9') or
#          variable_name.endswith('_val_bin4') or
#          variable_name.endswith('_aro_bin4'))
#         and isinstance(variable_value, (list, np.ndarray))
#     ):
#         globals()[variable_name + '_binary'] = (
#             np.asarray(variable_value) < 0.05
#         ).astype(int)


# Number of significant emotion pairs at each number of clusters
# (i.e., number of 1s across all selected emotion pairs)
valence_n_ones = np.sum(
    np.array(list(valence_binary.values())),
    axis=0
)

arousal_n_ones = np.sum(
    np.array(list(arousal_binary.values())),
    axis=0
)


# Number of emotion pairs for which BOTH valence and arousal
# are significant at the same number of clusters.
common_pairs = sorted(set(valence_binary).intersection(arousal_binary))
both_n_ones = np.sum(
    np.array([
        valence_binary[pair] * arousal_binary[pair]
        for pair in common_pairs
    ]),
    axis=0
)


# ---------------------------------------------------------
# BAR-PLOTTING FUNCTION
# ---------------------------------------------------------

def plot_n_ones(
    values,
    title,
    ylabel='Number of separable pairs',
    fig_name='number_of_ones_vs_n_clusters'
):

    plt.figure(figsize=(10, 6))

    plt.bar(
        nbr_clusters,
        values,
        width=0.7
    )

    plt.xlabel('Number of clusters')
    plt.ylabel(ylabel)
    plt.title(title)

    plt.xticks(nbr_clusters)
    plt.ylim(bottom=0)

    plt.grid(
        axis='y',
        linestyle='--',
        alpha=0.4
    )

    plt.tight_layout()
    plt.savefig(fig_name + '.jpg', dpi=600)


# ---------------------------------------------------------
# NUMBER OF 1s — AROUSAL
# ---------------------------------------------------------

plot_n_ones(
    arousal_n_ones,
    '',
    fig_name=str(
        figure_output_dir
        / f'arousal_number_of_ones_vs_n_clusters_bin{bin_size}'
    )
)


# plot_n_ones(
#     arousal_n_ones,
#     'Arousal: Number of significant emotion pairs vs. Number of Clusters',
#     fig_name='arousal_number_of_ones_vs_n_clusters_bin' + str(bin_size)
# )


# ---------------------------------------------------------
# NUMBER OF 1s — VALENCE
# ---------------------------------------------------------

plot_n_ones(
    valence_n_ones,
    '',
    fig_name=str(
        figure_output_dir
        / f'valence_number_of_ones_vs_n_clusters_bin{bin_size}'
    )
)

# plot_n_ones(
#     valence_n_ones,
#     'Valence: Number of significant emotion pairs vs. Number of Clusters',
#     fig_name='valence_number_of_ones_vs_n_clusters_bin' + str(bin_size)
# )


# ---------------------------------------------------------
# NUMBER OF 1s — VALENCE AND AROUSAL SIMULTANEOUSLY
# ---------------------------------------------------------

plot_n_ones(
    both_n_ones,
    '',
    fig_name=str(
        figure_output_dir
        / f'valence_and_arousal_number_of_ones_vs_n_clusters_bin{bin_size}'
    )
)

# plot_n_ones(
#     both_n_ones,
#     'Valence AND Arousal: Number of jointly significant emotion pairs vs. Number of Clusters',
#     fig_name='valence_and_arousal_number_of_ones_vs_n_clusters_bin' + str(bin_size)
# )

plt.close('all')
#%%------------------------------------
# find the maximum p-value for a specific number of clusters (e.g., 8) across all emotion pairs, 
# considering only those p-values that are less than 0.05.

cluster_number = 8
if cluster_number in nbr_clusters:
    index = nbr_clusters.index(cluster_number)
    significant_arousal_values = [
        values[index]
        for values in arousal.values()
        if values[index] < 0.05
    ]
    if significant_arousal_values:
        print(
            "maximum p-value for",
            cluster_number,
            "clusters:",
            max(significant_arousal_values)
        )
    else:
        print(
            "No significant arousal pairs for",
            cluster_number,
            "clusters."
        )
else:
    print(
        f"Skipping maximum p-value check: {cluster_number} clusters "
        "are not present in the loaded results."
    )

#%%-----------------------
# ---------------------------------------------------------
# (only R1-R4, R2-R3)
# ---------------------------------------------------------

#%%-------------------------
# ---------------------------------------------------------
# (only R1-R2, R1-R3)
# ---------------------------------------------------------


#%%-------------------------
# ---------------------------------------------------------
# (only R2-R4, R3-R4)
# ---------------------------------------------------------


#%%-----------

# ---------------------------------------------------------
# AROUSAL (excluding R5)
# ---------------------------------------------------------


# ---------------------------------------------------------
# AROUSAL (pairs containing R5)
# ---------------------------------------------------------

#%%-----------



#%%-----------

