#%% --------------------------------------------------
from processdataf import process_data
import os
import numpy as np
import time
import pandas as pd
from functions import * #proyect's functions / defs
from definitions import *

# from sklearn.preprocessing import RobustScaler
# import umap
# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
# from sklearn.mixture import GaussianMixture
# import seaborn as sns
# import os
# from scipy.stats import kruskal
# import scikit_posthocs as sp
# from functions import plot_trajectory_with_arrows_lin
# # from scipy.spatial.distance import cdist
# # from scipy.optimize import linear_sum_assignment
# import traceback
# from itertools import combinations_with_replacement


# Record the start time
start_time = time.time()

participants = np.arange(1, 31)
# participants = np.arange(6, 7)

print("Processing data......")

run_dir = os.environ.get("CLUSTER_RUN_DIR")
if run_dir:
    os.makedirs(run_dir, exist_ok=True)


# ------------------------------------------------------------------
# Arrays to store results across participants:
# ------------------------------------------------------------------

n_best_array = []
means_array = []
bic_scores_across_participants = []  
participant_data_list = []

# Population_level_ESM_val = []
# Population_level_ESM_aro = []

all_participant_level_ESM_val = []
all_participant_level_ESM_aro = []



big_scale_zeros_ones_matrix = []
# all_participants_emotion_matrix = []
all_participants_emotion_matrix_val = []
all_participants_emotion_matrix_aro = []


#%%-----------------------------------------------------------------


labels, emotion_pairs = build_labels_and_emotion_pairs(NUMBER_OF_BINS)

for i in participants:

    print("    ")
    print("    ")
    print("-------------------------------")
    print("participant: ", i)
    print("-------------------------------")
    print("    ")
    print("    ")

    try:
        xf, y_median, combined_df_median = process_data(
            str(i),
            available_saved_data=True
        )

        print("xf shape: ", xf.shape)

        print("Has NaNs", xf.isna().values.any())

        # Extract segment number for coloring
        segment_number = xf['segment_number'].values

        # Scale features
        xf_scaled = RobustScaler().fit_transform(
            xf.drop(
                columns=[
                    'segment_number',
                    'participant',
                    'val_chunks_number',
                    'aro_chunks_number',
                    'video'
                ]
            )
        )

        # UMAP
        umap_model = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=2,
            random_state=1234
        )

        low_dimensional_xf = umap_model.fit_transform(xf_scaled)

        plots_dir = os.path.join(
            run_dir if run_dir else os.path.dirname(__file__),
            'plots'
        )
        participant_plots_dir = os.path.join(
            plots_dir,
            f'participant_{i}'
        )

        os.makedirs(participant_plots_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        folder_name = (
            os.path.basename(os.path.abspath(run_dir))
            if run_dir
            else os.path.basename(os.path.dirname(os.path.abspath(__file__)))
        )

        colors = np.arange(combined_df_median.shape[0])

        bic_scores = []

        # Choose clustering range (or just the optimum number of clusters if known)   
        # n_components_range = range(10, 11)
        
        n_components_range = range(number_of_clusters, number_of_clusters+1)

        for n_components in n_components_range:

            # ---------------------------------------------------------
            # Gaussian Mixture
            # ---------------------------------------------------------
            gm = GaussianMixture(
                n_components=n_components,
                random_state=0
            )

            clusters = gm.fit_predict(low_dimensional_xf)

            means = gm.means_

            means_array.append(means)

            bic_scores.append(gm.bic(low_dimensional_xf))

            # ---------------------------------------------------------
            # Store cluster ids
            # ---------------------------------------------------------
            xf_plot = xf.copy()

            xf_plot['cluster_id'] = clusters

            xf_plot['cluster_id'] = pd.Categorical(
                xf_plot['cluster_id'].astype(int),
                categories=range(n_components),
                ordered=True
            )

            # ---------------------------------------------------------
            plot_clustering_and_label_distribution_per_cluster (xf, xf_plot, low_dimensional_xf, colors, clusters, means, participant_plots_dir, folder_name, n_components=n_components)

        # -------------------------------------------------------------
        # Best number of clusters according to BIC
        # -------------------------------------------------------------
        # best_n = n_components_range[np.argmin(bic_scores)]

        # print(
        #     f"Best number of components according to BIC: {best_n}"
        # )

        # n_best_array.append(best_n)

        bic_scores_across_participants.append(gm.bic(low_dimensional_xf))

        # -------------------------------------------------------------
        # Cluster statistics
        # -------------------------------------------------------------
        cluster_statistics(xf_plot, n_components)

        # dunn_combined_binary = kruskal_dunn_test(xf_plot, participant_plots_dir, folder_name, n_components, operation = 'or')
        # dunn_result_val_inverted, dunn_result_aro_inverted   = kruskal_dunn_test(xf_plot, participant_plots_dir, folder_name, n_components)
        dunn_result_val, dunn_result_aro = kruskal_dunn_test(xf_plot, participant_plots_dir, folder_name, n_components)
        
        plot_cluster_over_time(xf_plot, participant_plots_dir, folder_name, n_components)

        plot_cluster_vs_video(xf_plot, participant_plots_dir, folder_name, n_components)

        xf_plot = plot_cellID_over_time(xf_plot, participant_plots_dir, folder_name, n_components)            

        print(f"participant {i}")
        print(f"id(xf_plot) = {id(xf_plot)}")

        plot_cellID_vs_video(xf_plot, colors, participant_plots_dir, folder_name, n_components)

        # xf_plot_dom_emot is the dataset with the dominant emotion assigned to each cluster, based on the number of bins (4, 5 or 9)
        xf_plot_dom_emot = map_cluster_to_dominant_emotion(xf_plot, n_components, labels=labels, number_of_bins=NUMBER_OF_BINS)
        participant_data_list.append(xf_plot_dom_emot.copy())
        print(f"len(participant_data_list) = {len(participant_data_list)}")

        # Most dominant emotions across cluster for the participant 
        xf_plot_dom_emot['dominant_emotion'].value_counts()

        # dunn_combined_binary = build_dunn_matrix_with_emotion_labels(dunn_combined_binary, xf_plot_dom_emot)
        # print(dunn_combined_binary)
        # print_lower_triangular_red(dunn_combined_binary, include_diagonal = True)

        # emotions_per_cluster_matrix_val = build_dunn_matrix_with_emotion_labels(dunn_result_val_inverted, xf_plot_dom_emot)
        emotions_per_cluster_matrix_val = build_dunn_matrix_with_emotion_labels(dunn_result_val, xf_plot_dom_emot)
        print(emotions_per_cluster_matrix_val)
        print_lower_triangular_red(emotions_per_cluster_matrix_val, include_diagonal = True)

        # emotions_per_cluster_matrix_aro = build_dunn_matrix_with_emotion_labels(dunn_result_aro_inverted, xf_plot_dom_emot)
        emotions_per_cluster_matrix_aro = build_dunn_matrix_with_emotion_labels(dunn_result_aro, xf_plot_dom_emot)
        print(emotions_per_cluster_matrix_aro)
        print_lower_triangular_red(emotions_per_cluster_matrix_aro, include_diagonal = True)

        try:
            # matrix, matrix_zeros_ones_count = build_emotion_significance_matrix(dunn_combined_binary, verbose=True)
            # Participant_level_ESM = Participant-level Emotional Separability Matrix (ESM)
            
            Participant_level_ESM_val = build_emotion_significance_matrix(emotions_per_cluster_matrix_val, labels=labels)
            print("----------------------------------")
            print("Participant_level_ESM (valence):")
            print(Participant_level_ESM_val)
            print("----------------------------------")
            # print(matrix_zeros_ones_count)
            # Population_level_ESM_val = Population-level Emotional Separability Matrix (ESM)
            
            all_participant_level_ESM_val.append(Participant_level_ESM_val)
            # Population_level_ESM_val.append(Participant_level_ESM_val)

            # big_scale_zeros_ones_matrix.append(matrix_zeros_ones_count)
        except Exception as exc:
            print(f"Skipping participant {i}: {exc}")


        try:
            # matrix, matrix_zeros_ones_count = build_emotion_significance_matrix(dunn_combined_binary, verbose=True)
            # Participant-level Emotional Separability Matrix (ESM)
            Participant_level_ESM_aro = build_emotion_significance_matrix(emotions_per_cluster_matrix_aro, labels=labels)
            print("----------------------------------")
            print("Participant_level_ESM (arousal):")
            print(Participant_level_ESM_aro)
            print("----------------------------------")
            # print(matrix_zeros_ones_count)
            # Population_level_ESM_aro = Population-level Emotional Separability Matrix (ESM)
            all_participant_level_ESM_aro.append(Participant_level_ESM_aro)

            # Population_level_ESM_aro.append(Participant_level_ESM_aro)
            # big_scale_zeros_ones_matrix.append(matrix_zeros_ones_count)
        except Exception as exc:
            print(f"Skipping participant {i}: {exc}")


        # Group by emotion and cluster, count occurrences, and pivot the table
        emotion_table = (
            xf_plot_dom_emot.groupby(['dominant_emotion', 'cluster_id'])
            .size()
            .unstack(fill_value=0)
        )
        # Print the formatted table - 
        print(emotion_table)
        print("It counts the number of annotated window segments that fall into each emotion region, for each cluster")

        print(f"Participant {i} completed successfully.")
    
    except Exception as exc:

        print("\n====================================")
        print(f"ERROR processing participant {i}")
        traceback.print_exc()
        print("====================================\n")

        continue


#%%---------------------------------
if len(participant_data_list) > 0:

    xf_all_participants = pd.concat(
        participant_data_list,
        ignore_index=True
    )

    # Population-level Valence-Arousal plot and region counts
    population_region_data, population_region_summary, participant_region_summary = (
        plot_population_valence_arousal_regions(
            xf_all_participants,
            plots_dir=plots_dir,
            folder_name=folder_name,
            annotate_points=False
        )
    )

    print(len(participant_data_list))

    for k, df in enumerate(participant_data_list):
        print(k, df.shape)

    print([id(df) for df in participant_data_list])

    all_participants_emotion_matrix_val = np.stack(
        [matrix.to_numpy() for matrix in all_participant_level_ESM_val],
        axis=0
    )
    all_participants_emotion_matrix_aro = np.stack(
        [matrix.to_numpy() for matrix in all_participant_level_ESM_aro],
        axis=0
    )

    
    print("----------------------------------")
    print("Matrix with all participant-level ESM (Valence):")
    print(all_participant_level_ESM_val)
    
    print("----------------------------------")

    print("----------------------------------")
    print("Population_level_ESM - PoL (Valence) :")
    Population_level_ESM_withoutLabels_val = np.nanmedian(all_participants_emotion_matrix_val, axis=0)
    print("PoL as array :")
    print(Population_level_ESM_withoutLabels_val)
    Population_level_ESM_df_val = pd.DataFrame(Population_level_ESM_withoutLabels_val)

    Population_level_ESM_val = add_labels_to_matrix(
        Population_level_ESM_df_val,
        labels=('R1', 'R2', 'R3', 'R4', 'R5')
    )
    print("PoL as dataFrame :")
    print(Population_level_ESM_val)
    if run_dir:
        Population_level_ESM_val.to_csv(
            os.path.join(run_dir, "population_level_esm_valence.csv"),
            index=True
        )
    print("----------------------------------")

    print("----------------------------------")
    print("Matrix with all participant-level ESM (Arousal):")
    print(all_participant_level_ESM_aro)
    print("----------------------------------")

    print("----------------------------------")
    print("Population_level_ESM - PoL (Arousal) :")
    Population_level_ESM_withoutLabels_aro = np.nanmedian(all_participants_emotion_matrix_aro, axis=0)
    print("PoL as array :")
    print(Population_level_ESM_withoutLabels_aro)
    Population_level_ESM_df_aro = pd.DataFrame(Population_level_ESM_withoutLabels_aro)

    Population_level_ESM_aro = add_labels_to_matrix(
        Population_level_ESM_df_aro,
        labels=('R1', 'R2', 'R3', 'R4', 'R5')
    )
    print("PoL as dataFrame :")
    print(Population_level_ESM_aro)
    if run_dir:
        Population_level_ESM_aro.to_csv(
            os.path.join(run_dir, "population_level_esm_arousal.csv"),
            index=True
        )
    print("----------------------------------")


else:
    print("No participant data available.")


#%%---------------------------------

plot_cellID_per_participant_per_videoID(xf_all_participants, plots_dir, folder_name)


#%%---------------------------------------------------------------
# all_participants_emotion_matrix
# Stack into a 3D array: shape (n_participants, 4, 4)

# Population_level_ESM_val:
# is a list of the emotion significance matrices (pandas DataFrames)
# for each participant, where each matrix is of shape (N, N), 
# corresponding to the significance p-values for each emotion pair for a specific
# participant. N is the number of emotions or regions in the valence-arousal 
# space (e.g., 4 for A, B, R, S; 5 for R1, R2, R3, R4, R5, 4 for R1, R2, R3, R4). 


# all_participants_emotion_matrix_aro:
# is Population_level_ESM_aro, but converted to 3D numpy array:
# without the emotion labels. It is a 3D array of shape (n_participants, N, N) 
# containing the significance p-values for each emotion pair across all participants.


# all_participants_emotion_matrix_val = np.stack([m.to_numpy() for m in Population_level_ESM_val], axis=0)
# all_participants_emotion_matrix_aro = np.stack([m.to_numpy() for m in Population_level_ESM_aro], axis=0)


# emotion_pairs_data_val (or emotion_pairs_data_aro):
# is a list of R arrays, each containing the significance p-values for a 
# specific emotion pair across all participants. 
# R is the combination of N emotions taken 2 at a time (with repetition): math.comb(N+2-1, 2). 
# e.g., if N = 4 (e.g., A, B, R, S), then R = 10. 
# The order of the pairs is, e.g.: (A, A), (A, B), (A, R), (A, S), (B, B), (B, R), (B, S), (R, R), (R, S), (S, S). 
# Each array has a length equal to the number of participants.

emotion_pairs_data_val, xtick_labels = emotion_significance_box_plot(all_participants_emotion_matrix_val, labels=labels, plots_dir=plots_dir, folder_name=folder_name, emotion_type = "Valence")
emotion_pairs_data_aro, xtick_labels = emotion_significance_box_plot(all_participants_emotion_matrix_aro, labels=labels, plots_dir=plots_dir, folder_name=folder_name, emotion_type = "Arousal")

#%%--------------------------------------------------------------------

#-----------------------------------------------------------------------
# Separability analysis (from participants_medians_val_aro_binary) for Valence and Arousal:
#-----------------------------------------------------------------------

print("-----------------------------------------------------------")
print("Separability analysis")
print("-----------------------------------------------------------")


def compute_emotion_pair_medians(emotion_pairs_data, emotion_pairs, data_name="emotion_pairs_data"):
    
    n_pairs = len(emotion_pairs)

    # One median for each emotion pair
    participants_medians = np.full(n_pairs, np.nan)

    for pair_idx in range(n_pairs):

        x = emotion_pairs_data[pair_idx]

        # Check if x contains at least one non-NaN value
        if np.all(np.isnan(x)):
            q1 = np.nan
            median = np.nan
            q3 = np.nan
            lower_whisker = np.nan
            upper_whisker = np.nan

        else:
            q1 = np.nanpercentile(x, 25)
            median = np.nanpercentile(x, 50)
            q3 = np.nanpercentile(x, 75)

            iqr = q3 - q1

            lower_whisker = np.nanmin(
                x[x >= q1 - 1.5 * iqr]
            )

            upper_whisker = np.nanmax(
                x[x <= q3 + 1.5 * iqr]
            )

        print("------------------------------------------------------------")
        print(data_name, "- pair:", emotion_pairs[pair_idx])
        print("------------------------------------------------------------")

        print("Upper whisker:", upper_whisker)
        print("Q3:", q3)
        print("Median:", median)
        print("Q1:", q1)
        print("Lower whisker:", lower_whisker)

        # Median across the 30 participants
        participants_medians[pair_idx] = (
            np.nanmedian(x) if np.any(~np.isnan(x)) else np.nan
        )

    # Create labeled DataFrame
    participants_medians_df = pd.DataFrame(
        [participants_medians],
        columns=emotion_pairs
    )

    print("\n", data_name, 'Medians for each emotion pair across all participants:')
    print(participants_medians_df)

    return participants_medians, participants_medians_df


# participants_medians_val:
# has the median p-value for each emotion pair across all participants for Valence.

participants_medians_val, participants_medians_val_df = \
    compute_emotion_pair_medians(
        emotion_pairs_data_val,
        emotion_pairs,
        "Valence"
    )


# participants_medians_aro:
# has the median p-value for each emotion pair across all participants for Arousal.

participants_medians_aro, participants_medians_aro_df = \
    compute_emotion_pair_medians(
        emotion_pairs_data_aro,
        emotion_pairs,
        "Arousal"
    )

# participants_medians_val_aro:
# has the median p-value for each emotion pair across all participants for Valence and Arousal. 
participants_medians_val_aro = pd.DataFrame(
    {
        "Valence": participants_medians_val,
        "Arousal": participants_medians_aro
    },
    index=emotion_pairs
)

print("--------------------------")
print("Medians of p-values for each emotion pair across all participants:")
print(participants_medians_val_aro)
if run_dir:
    median_p_values_for_export = participants_medians_val_aro.copy()
    median_p_values_for_export.index = [
        pair.replace(", ", ",")
        for pair in median_p_values_for_export.index
    ]
    median_p_values_for_export.to_csv(
        os.path.join(run_dir, "median_p_values_by_emotion_pair.csv"),
        index=True
    )

# participants_medians_val_aro_binary:
# has a binary representation of the median p-values for each emotion pair 
# across all participants for Valence and Arousal, 
# where 1 indicates a statistically significant difference (p-value < P_VALUE) 
# and 0 indicates no significant difference (p-value >= P_VALUE).
participants_medians_val_aro_binary = (participants_medians_val_aro < P_VALUE).astype(int)

print("--------------------------")
print(f"Binarized medians of p-values (p-value < {P_VALUE} -> 1, otherwise -> 0):")
print(participants_medians_val_aro_binary)

#-----------------------------------------------------------------------
# Summarizing Separability analysis (from participants_medians_val_aro_binary) for Valence and Arousal:
#-----------------------------------------------------------------------

print("-----------------------------------------------------------")
print(" Summarizing Separability Analysis")
print("-----------------------------------------------------------")

print("How many statistically significant differences were found?")
print(participants_medians_val_aro_binary.sum())

# Pairs with a 1 in each dimension
valence_pairs = participants_medians_val_aro_binary.index[participants_medians_val_aro_binary["Valence"] == 1].tolist()
arousal_pairs = participants_medians_val_aro_binary.index[participants_medians_val_aro_binary["Arousal"] == 1].tolist()

# Pairs with 1 in both Valence and Arousal
both_pairs = participants_medians_val_aro_binary.index[
    (participants_medians_val_aro_binary["Valence"] == 1) &
    (participants_medians_val_aro_binary["Arousal"] == 1)
].tolist()

# Counts
n_valence = len(valence_pairs)
n_arousal = len(arousal_pairs)
n_both = len(both_pairs)

print(f"Valence = 1: {n_valence} pairs")
print(valence_pairs)

print(f"\nArousal = 1: {n_arousal} pairs")
print(arousal_pairs)

print(f"\nValence = 1 AND Arousal = 1: {n_both} pairs")
print(both_pairs)

print("-----------------------------------------------------------")



#%%---------------------------------------------------------------
# Valence / Arousal significance p-values, binary matrix of statistically 
# significant difference and proportions of statistically significant
# comparisons for each emotion pair across all participants

# emotion_pairs_data_df_binary_val (or emotion_pairs_data_df_binary_aro):
# is a DataFrame of shape (n_participants, R), where each column 
# corresponds to an emotion pair and each row corresponds to a participant. 
# The values are 1 if the comparison is statistically significant (p < P_VALUE), 0 if not, 
# and <NA> if the comparison was not applicable for that participant.

# emotion_pairs_data_df_binary_val is the binarized Emotional Separability Matrix (ESM) 

emotion_pairs_data_df_binary_val = print_p_values_binary_matrix_proportions(emotion_pairs_data_val, xtick_labels, emotion_type="Valence")
emotion_pairs_data_df_binary_aro = print_p_values_binary_matrix_proportions(emotion_pairs_data_aro, xtick_labels, emotion_type="Arousal")

#%%---------------------------------
# ---- Stacked bar plots (counts and proportions) for each emotion pair across all participants

def plot_significance_summary(
    emotion_pairs_data_df_binary,
    plots_dir,
    folder_name, 
    emotion_type = "Valence",
    dpi=150,
    figsize=(10, 5)
):
    """
    Create and save two stacked bar plots from a binary DataFrame
    (0, 1, <NA>):

    1. Counts of statistically significant (1) and non-significant (0)
       comparisons.
    2. Proportions of statistically significant (1) and
       non-significant (0) comparisons.

    Parameters
    ----------
    emotion_pairs_data_df_binary : pd.DataFrame
        DataFrame containing 0, 1 and <NA>.
    plots_dir : str
        Directory where the figures will be saved.
    folder_name : str
        Prefix used in the output filenames.
    dpi : int, optional
        Resolution of the saved figures.
    figsize : tuple, optional
        Figure size.
    """

    # --------------------------------------------------------
    # Compute counts
    # --------------------------------------------------------
    n_ones = emotion_pairs_data_df_binary.sum(skipna=True)
    n_zeros = emotion_pairs_data_df_binary.count() - n_ones
    n_valid = emotion_pairs_data_df_binary.count()

    # ========================================================
    # Counts plot
    # ========================================================
    fig, ax = plt.subplots(figsize=figsize)

    bars0 = ax.bar(n_zeros.index, n_zeros, label='0')
    bars1 = ax.bar(n_ones.index, n_ones, bottom=n_zeros, label='1')

    ax.bar_label(
        bars0,
        labels=[str(int(v)) if v > 0 else "" for v in n_zeros],
        label_type='center',
        fontsize=10
    )

    ax.bar_label(
        bars1,
        labels=[str(int(v)) if v > 0 else "" for v in n_ones],
        label_type='center',
        fontsize=10
    )

    ax.set_xlabel("Emotion pair")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Number of statistically significant (1) and "
        f"non-significant (0) comparisons ({emotion_type})"
    )
    ax.legend(title="Binary value")

    plt.xticks(rotation=45)
    plt.tight_layout()

    fig_path = os.path.join(
        plots_dir,
        f"{folder_name}_significance_count_{emotion_type}.jpg"
    )

    try:
        fig.savefig(fig_path, dpi=dpi)
    except Exception as exc:
        print(f"Error saving {fig_path}")
        print(exc)
    finally:
        plt.close(fig)

    # ========================================================
    # Proportions plot
    # ========================================================
    prop_ones = n_ones / n_valid
    prop_zeros = n_zeros / n_valid

    fig, ax = plt.subplots(figsize=figsize)

    bars0 = ax.bar(prop_zeros.index, prop_zeros, label='0')
    bars1 = ax.bar(prop_ones.index, prop_ones, bottom=prop_zeros, label='1')

    ax.bar_label(
        bars0,
        labels=[f"{100*v:.1f}%" if v > 0 else "" for v in prop_zeros],
        label_type='center',
        fontsize=10
    )

    ax.bar_label(
        bars1,
        labels=[f"{100*v:.1f}%" if v > 0 else "" for v in prop_ones],
        label_type='center',
        fontsize=10
    )

    ax.set_ylim(0, 1)
    ax.set_xlabel("Emotion pair")
    ax.set_ylabel("Proportion")
    ax.set_title(
        f"Proportion of statistically significant (1) and "
        f"non-significant (0) comparisons ({emotion_type})"
    )
    ax.legend(title="Binary value")

    plt.xticks(rotation=45)
    plt.tight_layout()

    fig_path = os.path.join(
        plots_dir,
        f"{folder_name}_significance_proportion_{emotion_type}.jpg"
    )

    try:
        fig.savefig(fig_path, dpi=dpi)
    except Exception as exc:
        print(f"Error saving {fig_path}")
        print(exc)
    finally:
        plt.close(fig)


plot_significance_summary(
    emotion_pairs_data_df_binary_val,
    plots_dir=plots_dir,
    folder_name=folder_name,
    emotion_type="Valence"
)

plot_significance_summary(
    emotion_pairs_data_df_binary_aro,
    plots_dir=plots_dir,
    folder_name=folder_name,
    emotion_type="Arousal"
)

#%% -------------------------------------------------------------
# print("Best n for each participant: ", n_best_array)

# print(
#     "Average best n across participants: ",
#     np.mean(n_best_array)
# )

print(f"BIC score for each participant: {bic_scores_across_participants}")
print(f"mean BIC score across partipants: {np.mean(bic_scores_across_participants)}")
print(f"std BIC score across partipants: {np.std(bic_scores_across_participants)}")
print(f"std/mean [%] BIC score across partipants: {100*np.std(bic_scores_across_participants)/np.mean(bic_scores_across_participants)}")

fig_significance = plt.figure()
sns.boxplot(
    y=bic_scores_across_participants,
    medianprops={
        "color": "r",
        "linewidth": 2
    },
    notch=True,
)

# plt.xlabel('Cluster ID')
plt.ylabel('BIC score')

plt.title(
    f'BIC score across all participants'
)

fig_signific_ratio_path = os.path.join(
        plots_dir,
    f'{folder_name}_bic_score_{n_components}.jpg'
)

try:
    fig_significance.savefig(fig_signific_ratio_path, dpi=150)
except Exception as exc:
    print(f"Error saving figure for participant {i}")
    print(exc)

plt.close(fig_significance)

# Record the end time
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time

print("Elapsed time:", elapsed_time / 60, "minutes")
print("Elapsed time:", elapsed_time / (60 * 60), "hours")


#%%-------------------------------------------------------------


