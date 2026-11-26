#%% --------------------------------------------------
from processdataf import process_data
import numpy as np
import time
import pandas as pd
from sklearn.preprocessing import RobustScaler
import umap
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.mixture import GaussianMixture
import seaborn as sns
import os
from scipy.stats import kruskal
import scikit_posthocs as sp
from functions import plot_trajectory_with_arrows_lin
# from scipy.spatial.distance import cdist
# from scipy.optimize import linear_sum_assignment
import traceback
from itertools import combinations_with_replacement

# Record the start time
start_time = time.time()

participants = np.arange(1, 31)
# participants = np.arange(23, 25)

print("Processing data......")

# ------------------------------------------------------------------
# Method configuration parameters:
# ------------------------------------------------------------------
number_of_clusters = 8
NUMBER_OF_BINS = 5 # Number of bins for mapping valence/arousal to 2D space (e.g., 4 or 9)
P_VALUE = 0.05
#------------------------------------------------------------------

# ------------------------------------------------------------------
# Valence / Arousal thresholds for mapping to X cells/bins in the 2D space
# ------------------------------------------------------------------
# This thresholds are used by the function map_label_to_cell to map 
# 9 bins thresholds
# valence/arousal values to a cell/bin in the 2D space

LOW_MED_THRESHOLD = 3.66
MED_HIGH_THRESHOLD = 6.33

if MED_HIGH_THRESHOLD < LOW_MED_THRESHOLD:
    raise ValueError(
        "MED_HIGH_THRESHOLD must be greater than LOW_MED_THRESHOLD"
    )


# 4 bins thresholds
MIDDLE_THRESHOLD = 5.0

# 5 bins thresholds
UPPER_MIDDLE_THRESHOLD = 5.5
MIDDLE_THRESHOLD = 5.0
LOWER_MIDDLE_THRESHOLD = 4.5

if MIDDLE_THRESHOLD < LOWER_MIDDLE_THRESHOLD or MIDDLE_THRESHOLD > UPPER_MIDDLE_THRESHOLD:
    raise ValueError(
        "MIDDLE_THRESHOLD must be between LOWER_MIDDLE_THRESHOLD and UPPER_MIDDLE_THRESHOLD"
    )


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

# ------------------------------------------------------------------
# Functions section
# ------------------------------------------------------------------


def build_labels_and_emotion_pairs(NUMBER_OF_BINS):

    if NUMBER_OF_BINS == 4:
        print("Mapping to 4 bins...: R1 (LVLA), R2 (LVHA), R3 (HVLA), R4 (HVHA)")

        # labels = ['LVHA', 'LVLA', 'HVHA', 'HVLA']
        # labels = ['LVLA', 'LVHA', 'HVLA', 'HVHA']
        labels = ['R1', 'R2', 'R3', 'R4'] # R1 = LVLA, R2 = LVHA, R3 = HVLA, R4 = HVHA

        emotion_pairs = [
            f"({a}, {b})"
            for a, b in combinations_with_replacement(labels, 2)
        ]  # all combinations with repetition (LVHA,LVHA, LVHA,LVLA, …, HVLA,HVLA),
    elif NUMBER_OF_BINS == 9:
        print("Mapping to 9 bins...: A, B, R, S")

        labels = ['A', 'B', 'R', 'S']

        emotion_pairs = [
            f"({a}, {b})"
            for a, b in combinations_with_replacement(labels, 2)
        ]  # all combinations with repetition (A,A, A,B, …, S,S),

        # e.g., if labels = ['A', 'B', 'R', 'S'] then 
        # emotion_pairs = [
        #     "(A, A)",
        #     "(A, B)",
        #     "(A, R)",
        #     "(A, S)",
        #     "(B, B)",
        #     "(B, R)",
        #     "(B, S)",
        #     "(R, R)",
        #     "(R, S)",
        #     "(S, S)"
        # ]
    elif NUMBER_OF_BINS == 5:
        print("Mapping to 5 bins...: R1 (LVLA), R2 (LVHA), R3 (HVLA), R4 (HVHA), R5 (Neutral)")

        labels = ['R1', 'R2', 'R3', 'R4', 'R5'] # R1 = LVLA, R2 = LVHA, R3 = HVLA, R4 = HVHA, R5 = Neutral

        emotion_pairs = [
            f"({a},{b})"
            for a, b in combinations_with_replacement(labels, 2)
        ]
    print("Emotions pairs: ", emotion_pairs)
    print("number of pairs: ", len(emotion_pairs))

    return labels, emotion_pairs



# ------------------------------------------------------------------

def map_label_to_4_bins(val, aro):
    # cell_id is the bin_id in the 2D space defined by
    # valence and arousal thresholds
    # cell and bin are synonymous in this context

    if val < MIDDLE_THRESHOLD and aro < MIDDLE_THRESHOLD:
        cell_id = 1

    elif val < MIDDLE_THRESHOLD and aro >= MIDDLE_THRESHOLD:
        cell_id = 2

    elif val >= MIDDLE_THRESHOLD and aro < MIDDLE_THRESHOLD:
        cell_id = 3

    elif val >= MIDDLE_THRESHOLD and aro >= MIDDLE_THRESHOLD:
        cell_id = 4

    return cell_id


def map_label_to_5_bins(val, aro):
    # cell_id is the bin_id in the 2D space defined by
    # valence and arousal thresholds
    # cell and bin are synonymous in this context

    if (val <= LOWER_MIDDLE_THRESHOLD and aro < MIDDLE_THRESHOLD) or (val < MIDDLE_THRESHOLD and aro <= LOWER_MIDDLE_THRESHOLD):
        cell_id = 1

    elif (val <= LOWER_MIDDLE_THRESHOLD and aro >= MIDDLE_THRESHOLD) or (val < MIDDLE_THRESHOLD and aro >= UPPER_MIDDLE_THRESHOLD):
        cell_id = 2

    elif (val >= UPPER_MIDDLE_THRESHOLD and aro < MIDDLE_THRESHOLD) or (val >= MIDDLE_THRESHOLD and aro <= LOWER_MIDDLE_THRESHOLD):
        cell_id = 3
 
    elif (val >= UPPER_MIDDLE_THRESHOLD and aro >= MIDDLE_THRESHOLD) or (val >= MIDDLE_THRESHOLD and aro >= UPPER_MIDDLE_THRESHOLD):
        cell_id = 4

    elif (val > LOWER_MIDDLE_THRESHOLD and val < UPPER_MIDDLE_THRESHOLD) and (aro > LOWER_MIDDLE_THRESHOLD and aro < UPPER_MIDDLE_THRESHOLD):
        cell_id = 5

    return cell_id


def map_label_to_9_bins (val, aro):
    # cell_id is the bin_id in the 2D space defined by valence and arousal thresholds
    # cell and bin are synonymous in this context

    if val < LOW_MED_THRESHOLD and aro < LOW_MED_THRESHOLD:
        cell_id = 1
    elif LOW_MED_THRESHOLD <= val < MED_HIGH_THRESHOLD and aro < LOW_MED_THRESHOLD:
        cell_id = 4
    elif MED_HIGH_THRESHOLD <= val and aro < LOW_MED_THRESHOLD:
        cell_id = 7
    elif val < LOW_MED_THRESHOLD and LOW_MED_THRESHOLD <= aro < MED_HIGH_THRESHOLD:
        cell_id = 2
    elif LOW_MED_THRESHOLD <= val < MED_HIGH_THRESHOLD and LOW_MED_THRESHOLD <= aro < MED_HIGH_THRESHOLD:
        cell_id = 5
    elif MED_HIGH_THRESHOLD <= val and LOW_MED_THRESHOLD <= aro < MED_HIGH_THRESHOLD:
        cell_id = 8
    elif val < LOW_MED_THRESHOLD and MED_HIGH_THRESHOLD <= aro:
        cell_id = 3 
    elif LOW_MED_THRESHOLD <= val < MED_HIGH_THRESHOLD and MED_HIGH_THRESHOLD <= aro:
        cell_id = 6
    elif MED_HIGH_THRESHOLD <= val and MED_HIGH_THRESHOLD <= aro:
        cell_id = 9

    return cell_id


def map_label_to_cell (val, aro, number_of_bins=9):
    # cell_id is the bin_id in the 2D space defined by valence and arousal thresholds
    # cell and bin are synonymous in this context

    if number_of_bins == 9:
        cell_id = map_label_to_9_bins (val, aro)
    elif number_of_bins == 4:
        cell_id = map_label_to_4_bins (val, aro)
    elif number_of_bins == 5:
        cell_id = map_label_to_5_bins (val, aro)

    return cell_id



def plot_clustering_and_label_distribution_per_cluster (xf, xf_plot, low_dimensional_xf, clusters, means, participant_plots_dir, folder_name, n_components=9):


    # ---------------------------------------------------------
    # Figure layout
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(22, 12))

    gs = GridSpec(2, 6, figure=fig)

    # Top row
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax3 = fig.add_subplot(gs[0, 4:6])

    # Bottom row
    ax4 = fig.add_subplot(gs[1, 0:3])
    ax5 = fig.add_subplot(gs[1, 3:6])

    # ---------------------------------------------------------
    # Plot 1 - Clustered UMAP
    # ---------------------------------------------------------
    for cluster_id in range(n_components):

        idx = clusters == cluster_id

        ax1.scatter(
            low_dimensional_xf[idx, 0],
            low_dimensional_xf[idx, 1],
            s=50,
            label=f'Cluster {cluster_id + 1}'
        )

    ax1.scatter(
        means[:, 0],
        means[:, 1],
        c='black',
        marker='x',
        s=300,
        linewidths=3,
        label='Means'
    )

    ax1.set_xlabel('UMAP 1')
    ax1.set_ylabel('UMAP 2')

    ax1.set_title(
        f'(a) Model-Based Clusters (n_clusters={n_components})'
    )

    ax1.legend()

    # ---------------------------------------------------------
    # Plot 2 - Temporal progression
    # ---------------------------------------------------------
    scatter2 = ax2.scatter(
        low_dimensional_xf[:, 0],
        low_dimensional_xf[:, 1],
        c=segment_number,
        cmap='viridis',
        s=50
    )

    ax2.set_xlabel('UMAP 1')
    ax2.set_ylabel('UMAP 2')
    ax2.set_title('(b) Temporal Progression')

    fig.colorbar(
        scatter2,
        ax=ax2,
        label='Segment Number'
    )

    # ---------------------------------------------------------
    # Plot 3 - Valence vs arousal
    # ---------------------------------------------------------
    scatter3 = ax3.scatter(
        xf['val_chunks_number'],
        xf['aro_chunks_number'],
        c=colors,
        cmap='plasma',
        s=50
    )

    ax3.set_xlabel('Valence')
    ax3.set_ylabel('Arousal')

    ax3.set_title('(c) Participant Valence vs Arousal')

    fig.colorbar(
        scatter3,
        ax=ax3,
        label='Segment Number'
    )

    # ---------------------------------------------------------
    # Plot 4 - Valence boxplot
    # ---------------------------------------------------------
    sns.boxplot(
        x=xf_plot['cluster_id'].astype(int) + 1,
        y=xf_plot['val_chunks_number'],
        medianprops={
            "color": "r",
            "linewidth": 2
        },
        notch=True,
        ax=ax4
    )

    ax4.set_xlabel('Cluster ID')
    ax4.set_ylabel('Valence')

    ax4.set_title(
        f'(d) Valence by Cluster (1-{n_components})'
    )

    # ---------------------------------------------------------
    # Plot 5 - Arousal boxplot
    # ---------------------------------------------------------
    sns.boxplot(
        x=xf_plot['cluster_id'].astype(int) + 1,
        y=xf_plot['aro_chunks_number'],
        medianprops={
            "color": "r",
            "linewidth": 2
        },
        notch=True,
        ax=ax5
    )

    ax5.set_xlabel('Cluster ID')
    ax5.set_ylabel('Arousal')

    ax5.set_title(
        f'(e) Arousal by Cluster (1-{n_components})'
    )

    # ---------------------------------------------------------
    # Final layout
    # ---------------------------------------------------------
    fig.tight_layout()

    fig_path = os.path.join(
        participant_plots_dir,
        f'{folder_name}_combined_{n_components}.jpg'
    )

    # plt.show()

    try:
        fig.savefig(fig_path, dpi=800)
        plt.close(fig)
    except Exception as exc:
        print(f"Error saving figure for participant {i}")
        print(exc)
    
def cluster_statistics(xf_plot, n_components):
            
    for cluster_id in range(n_components):

        print(f"---- Cluster {cluster_id + 1}:")

        val_mean = xf_plot[
            xf_plot['cluster_id'].astype(int) == cluster_id
        ]['val_chunks_number'].mean()

        val_std = xf_plot[
            xf_plot['cluster_id'].astype(int) == cluster_id
        ]['val_chunks_number'].std()

        aro_mean = xf_plot[
            xf_plot['cluster_id'].astype(int) == cluster_id
        ]['aro_chunks_number'].mean()

        aro_std = xf_plot[
            xf_plot['cluster_id'].astype(int) == cluster_id
        ]['aro_chunks_number'].std()

        print(f"Mean Valence: {val_mean}")
        print(f"Std Valence : {val_std}")

        print("--------------------------")

        print(f"Mean Arousal: {aro_mean}")
        print(f"Std Arousal : {aro_std}")

        print("--------------------------")

def kruskal_dunn_test(xf_plot, participant_plots_dir, folder_name, n_components):

    nan_matrix = pd.DataFrame(
        np.nan,
        index=range(n_components),
        columns=range(n_components)
    )

    # Kruskal-Wallis test results
    clusters_for_kruskal_val = []
    clusters_for_kruskal_aro = []

    for cluster_id in range(n_components):
        cluster_val = xf_plot[
            xf_plot['cluster_id'].astype(int) == cluster_id
        ]['val_chunks_number']
        cluster_aro = xf_plot[
                xf_plot['cluster_id'].astype(int) == cluster_id
            ]['aro_chunks_number']
    
        clusters_for_kruskal_val.append(cluster_val)
        clusters_for_kruskal_aro.append(cluster_aro)

    stat_val, p_value_valence = kruskal(*clusters_for_kruskal_val)

    print("Kruskal-Wallis statistic (Valence):", stat_val)
    print("Kruskal-Wallis p-value (Valence):", p_value_valence)

    stat_aro, p_value_arousal = kruskal(*clusters_for_kruskal_aro)

    print("Kruskal-Wallis statistic (Arousal):", stat_aro)
    print("Kruskal-Wallis p-value (Arousal):", p_value_arousal)

    if p_value_valence < 0.05:


        print("There are significant differences between clusters (Valence).")
        print("Proceeding with Dunn's test for pairwise comparisons...")

        # -------------------------------------------------------------
        # Dunn's test results (Valence)
        # -------------------------------------------------------------

        df = pd.DataFrame({'val': xf_plot['val_chunks_number'], 'cluster': xf_plot['cluster_id']})

        # Dunn test
        dunn_result = sp.posthoc_dunn(
            df,
            val_col='val',
            group_col='cluster',
            p_adjust='bonferroni'
        )

        print("\nDunn's test results (Valence):")
        print(dunn_result)    

        fig_dunn = plt.figure()
        plt.imshow(dunn_result)
        plt.colorbar()
        plt.xticks(range(len(dunn_result)), range(1, len(dunn_result.columns) + 1))
        plt.yticks(range(len(dunn_result)), range(1, len(dunn_result.columns) + 1))
        plt.title("Dunn Test p-values (Valence)")
        # plt.show()


        fig_dunn_path = os.path.join(
            participant_plots_dir,
            f'{folder_name}_dunn_val_{n_components}.jpg'
        )

        # plt.show()


        for k in range(5):
            try:
                fig_dunn.savefig(fig_dunn_path, dpi=150)
                plt.close(fig_dunn)
                break
            except OSError:
                time.sleep(1)

        # binarize dunn results
        # dunn_result_val_binary = dunn_result < P_VALUE
        # print("dunn_result_val_binary: ",dunn_result_val_binary)

        dunn_result_val = dunn_result.copy()
    else:        
        print("There are NO significant differences between clusters (Valence)")
        print("Therefore, we cannot proceed with Dunn's test for pairwise comparisons.")
        print("-----------------------------------------------------------------------")
        dunn_result_val = nan_matrix.copy()
        

    if p_value_arousal < 0.05:


        print("There are significant differences between clusters (Arousal).")
        print("Proceeding with Dunn's test for pairwise comparisons...")
        # -------------------------------------------------------------
        # Dunn's test results (Arousal)
        # -------------------------------------------------------------

        df = pd.DataFrame({'aro': xf_plot['aro_chunks_number'], 'cluster': xf_plot['cluster_id']})

        # Dunn test
        dunn_result = sp.posthoc_dunn(
            df,
            val_col='aro',
            group_col='cluster',
            p_adjust='bonferroni'
        )

        print("\nDunn's test results (Arousal):")
        print(dunn_result)    

        fig_dunn = plt.figure()
        plt.imshow(dunn_result)
        plt.colorbar()
        plt.xticks(range(len(dunn_result)), range(1, len(dunn_result.columns) + 1))
        plt.yticks(range(len(dunn_result)), range(1, len(dunn_result.columns) + 1))
        plt.title("Dunn Test p-values (Arousal)")
        # plt.show()

        fig_dunn_path = os.path.join(
                participant_plots_dir,
            f'{folder_name}_dunn_aro_{n_components}.jpg'
        )

        # plt.show()

        for k in range(5):
            try:
                fig_dunn.savefig(fig_dunn_path, dpi=150)
                plt.close(fig_dunn)
                break
            except OSError:
                time.sleep(1)

        dunn_result_aro = dunn_result.copy()
    else:
        print("There are NO significant differences between clusters (Arousal).")
        print("Therefore, we cannot proceed with Dunn's test for pairwise comparisons.")
        print("-----------------------------------------------------------------------")
        dunn_result_aro = nan_matrix.copy()

    return dunn_result_val, dunn_result_aro

def plot_cluster_over_time(xf_plot, participant_plots_dir, folder_name, n_components):

    # -------------------------------------------------------------
    # Plot Cluster IDs over time
    # -------------------------------------------------------------    
    fig_cluster_time = plt.figure()
    plt.plot(xf_plot['segment_number'], xf_plot['cluster_id'], linestyle=(0, (1, 10)), marker='o')
    plt.xlabel('Segment Number')
    plt.ylabel('Cluster ID')
    plt.title('Cluster ID over time')

    # Shift displayed y-axis labels by +1
    y_min = int(xf_plot['cluster_id'].min())
    y_max = int(xf_plot['cluster_id'].max())

    plt.ylim(y_min, y_max)

    ticks = np.arange(y_min, y_max + 1, 1)
    plt.yticks(ticks, [int(t) + 1 for t in ticks])
    
    # # Shift displayed y-axis labels by +1
    # ticks, labels = plt.yticks()
    # plt.yticks(ticks, [int(t) + 1 for t in ticks])
    # plt.ylim(0,10)

    fig_cluster_path = os.path.join(
        participant_plots_dir,
        f'{folder_name}_cluster_time_{n_components}.jpg'
    )

    # plt.show()

    for k in range(5):
        try:
            fig_cluster_time.savefig(fig_cluster_path, dpi=150)
            plt.close(fig_cluster_time)
            break
        except OSError:
            time.sleep(1)

def plot_cellID_per_participant_per_videoID(xf_all_participants, plots_dir, folder_name): 

    for video_id in range(1, 9):

        xp_ordered_video = xf_all_participants[
            xf_all_participants['video'] == video_id
        ].sort_values(by='participant').copy()

        fig = plt.figure(figsize=(7, 4))
        scatter4 = plt.scatter(
            xp_ordered_video['participant'],
            xp_ordered_video['cell_id'],
            # c=colors,
            # cmap='plasma',
            s=50
        )

        plt.xlabel('Participant')
        plt.ylabel('Bin ID')
        # plt.title(f'Cell ID over participant (for video ID = {video_id})')
        plt.title(f'Bin ID over participant (for video ID = {video_id})')

        # Adapt Y-axis to the minimum and maximum cell_id
        y_min = 1
        y_max = NUMBER_OF_BINS

        plt.ylim(y_min, y_max)

        # Show only integer values on the Y-axis
        plt.yticks(np.arange(y_min, y_max + 1, 1))

        plt.grid(True)
    
        fig_cell_time_path = os.path.join(
            plots_dir,
            # f'{folder_name}_cell_vs_video_part_{video_id}.jpg'
            f'{folder_name}_bin_vs_video_part_{video_id}.jpg'
        )

        # plt.show()

        try:
            fig.savefig(fig_cell_time_path, dpi=150)
            plt.close(fig)
        except Exception as exc:
            print(f"Error saving figure for participant {i}")
            print(exc)

def plot_cellID_over_time(xf_plot, participant_plots_dir, folder_name, n_components):            

    xf_plot['cell_id'] = xf_plot.apply(lambda row: map_label_to_cell(row['val_chunks_number'], row['aro_chunks_number'], number_of_bins=NUMBER_OF_BINS), axis=1)
    # print(xf_plot['cell_id'])
    # xf_plot.iloc[831]

    # participant_data_list.append(xf_plot.copy())

    # print(f"participant {i}")
    # print(f"len(participant_data_list) = {len(participant_data_list)}")
    # print(f"id(xf_plot) = {id(xf_plot)}")

    fig_cell_time = plt.figure()
    plt.plot(xf_plot['segment_number'], xf_plot['cell_id'], linestyle=(0, (1, 10)), marker='o')
    plt.xlabel('Segment Number')
    plt.ylabel('Cell ID')
    plt.title('Cell ID over time')
    # # Shift displayed y-axis labels by +1
    # ticks, labels = plt.yticks()
    # plt.yticks(ticks, [int(t) + 1 for t in ticks])
    plt.ylim(1,16)

    fig_cell_time_path = os.path.join(
        participant_plots_dir,
        f'{folder_name}_cell_time_{n_components}.jpg'
    )

    # plt.show()

    for k in range(5):
        try:
            fig_cell_time.savefig(fig_cell_time_path, dpi=150)
            plt.close(fig_cell_time)
            break
        except OSError:
            time.sleep(1)
        # except Exception as exc:
        #     print(f"Error saving figure for participant {i}")
        #     print(exc)

    return xf_plot.copy()

    # -------------------------------------------------------------
    # Plot Cell IDs over participant (per video ID)
    # -------------------------------------------------------------    

    for video_id in range(1,9):

        xp_ordered_video = xf_all_participants[xf_all_participants['video']==video_id].sort_values(by='participant').copy()

        fig = plt.figure(figsize=(7, 4))
        scatter4 = plt.scatter(
            xp_ordered_video['participant'],
            xp_ordered_video['cell_id'],
            # c=colors,
            # cmap='plasma',
            s=50
        )

        plt.xlabel('Participant')
        plt.ylabel('Cell ID')
        plt.title(f'Cell ID over participant (for video ID = {video_id})')
        plt.ylim(1,9)
        plt.grid(True)
        
        fig_cell_time_path = os.path.join(
            plots_dir,
            f'{folder_name}_cell_vs_video_part_{video_id}.jpg'
        )

        # plt.show()

        try:
            fig.savefig(fig_cell_time_path, dpi=150)
            plt.close(fig_cell_time)
        except Exception as exc:
            print(f"Error saving figure for participant {i}")
            print(exc)

def plot_cluster_vs_video(xf_plot, participant_plots_dir, folder_name, n_components):

    # -------------------------------------------------------------
    # Plot Cluster IDs over video ID
    # -------------------------------------------------------------    
    xp_ordered_video = xf_plot.sort_values(by='video')

    fig = plot_trajectory_with_arrows_lin(
        xp_ordered_video['video'],
        xp_ordered_video['cluster_id'],
        n_arrows=50
    )

    # -------------------------------------------------------------
    # Adapt Y-axis to cluster_id min/max
    # -------------------------------------------------------------
    y_min = int(xp_ordered_video['cluster_id'].min())
    y_max = int(xp_ordered_video['cluster_id'].max())

    ax = fig.gca()

    ax.set_ylim(y_min, y_max)

    ticks = np.arange(y_min, y_max + 1, 1)
    ax.set_yticks(ticks)
    ax.set_yticklabels([int(t) + 1 for t in ticks])

    # -------------------------------------------------------------
    # Save figure
    # -------------------------------------------------------------
    fig_cluster_video_path = os.path.join(
        participant_plots_dir,
        f'{folder_name}_cluster_video_{n_components}.jpg'
    )

    for k in range(5):
        try:
            fig.savefig(fig_cluster_video_path, dpi=150)
            plt.close(fig)
            break
        except OSError:
            time.sleep(1)
            
def plot_cellID_vs_video(xf_plot, participant_plots_dir, folder_name, n_components):

    # -------------------------------------------------------------
    # Plot Cell IDs over video ID (scatter)
    # -------------------------------------------------------------    
    xp_ordered_video = xf_plot.sort_values(by='video').copy()

    fig_cell_time = plt.figure(figsize=(7, 4))
    scatter4 = plt.scatter(
        xp_ordered_video['video'],
        xp_ordered_video['cell_id'],
        c=colors,
        cmap='plasma',
        s=50
    )

    plt.xlabel('Video ID')
    plt.ylabel('Cell ID')
    plt.title('Cell ID over video ID')
    plt.grid(True)
    # # Shift displayed y-axis labels by +1
    # ticks, labels = plt.yticks()
    # plt.yticks(ticks, [int(t) + 1 for t in ticks])
    # plt.ylim(0,10)

    # fig.colorbar(
    #     scatter4,
    #     label='?'
    # )

    fig_cell_time_path = os.path.join(
        participant_plots_dir,
        f'{folder_name}_cell_time_{n_components}.jpg'
    )

    # plt.show()
    for k in range(5):
        try:
            fig_cell_time.savefig(fig_cell_time_path, dpi=150)
            plt.close(fig_cell_time)
            break
        except Exception as exc:
            print(f"Error saving figure for participant {i}")
            print(exc)

def map_cluster_to_dominant_emotion_4_overlapping_bins(xf_plot, n_components):

    for cluster_id in range(n_components):
        
        scary_bin = 0
        amusing_bin = 0
        boring_bin = 0
        relaxing_bin = 0

        # compute cell_id counts:
        counts = xf_plot[xf_plot['cluster_id'] == cluster_id].loc[:, 'cell_id'].value_counts()
        # Access the cell_id values
        unique_ids = counts.index

        # fill emotion bins with cell_id counts:
        for idx in range (0, unique_ids.shape[0]):
            
            if unique_ids[idx] == 1:
                boring_bin += counts[unique_ids[idx]]
            if unique_ids[idx] == 2:
                boring_bin += counts[unique_ids[idx]]
                scary_bin += counts[unique_ids[idx]]
            if unique_ids[idx] == 3:
                scary_bin += counts[unique_ids[idx]]
            if unique_ids[idx] == 4:
                boring_bin += counts[unique_ids[idx]]
                relaxing_bin += counts[unique_ids[idx]]
            if unique_ids[idx] == 5:
                boring_bin += counts[unique_ids[idx]]
                relaxing_bin += counts[unique_ids[idx]]
                scary_bin += counts[unique_ids[idx]]
                amusing_bin += counts[unique_ids[idx]]
            if unique_ids[idx] == 6:
                scary_bin += counts[unique_ids[idx]]
                amusing_bin += counts[unique_ids[idx]]
            if unique_ids[idx] == 7:
                relaxing_bin += counts[unique_ids[idx]]
            if unique_ids[idx] == 8:
                # boring_bin += counts[unique_ids[idx]]
                relaxing_bin += counts[unique_ids[idx]]
                amusing_bin += counts[unique_ids[idx]]
            if unique_ids[idx] == 9:
                amusing_bin += counts[unique_ids[idx]]


        # print(f"scary_bin: {scary_bin}, amusing_bin: {amusing_bin}, boring_bin: {boring_bin}, relaxing_bin: {relaxing_bin}")

        # bins = {
        #     "scary_bin": scary_bin,
        #     "amusing_bin": amusing_bin,
        #     "boring_bin": boring_bin,
        #     "relaxing_bin": relaxing_bin
        # }


        bins = {
            "S": scary_bin,
            "A": amusing_bin,
            "B": boring_bin,
            "R": relaxing_bin
        }

        bins_df = pd.DataFrame(list(bins.items()), columns=['bin', 'count'])

        dominant_emotion = bins_df.loc[bins_df['count'].idxmax(), 'bin']
        
        # assign dominant emotion to all rows in xf_plot with the current cluster_id
        mask = xf_plot['cluster_id'] == cluster_id
        xf_plot.loc[mask, 'dominant_emotion'] = dominant_emotion

    # check assigned dominant emotions:
    xf_plot['dominant_emotion'].value_counts()

    # check for NaN values in the 'dominant_emotion' column
    # print("Has NaNs", xf_plot['dominant_emotion'].isna().values.any())
    
    return xf_plot.copy()


def map_cluster_to_dominant_emotion_4_non_overlapping_bins(xf_plot, n_components): 
 
    for cluster_id in range(n_components): 
         
        LVHA_bin = 0 
        HVHA_bin = 0 
        LVLA_bin = 0 
        HVLA_bin = 0 
 
        # compute cell_id counts: 
        counts = xf_plot[xf_plot['cluster_id'] == cluster_id].loc[:, 'cell_id'].value_counts() 
        # Access the cell_id values 
        unique_ids = counts.index 
 
        # fill emotion bins with cell_id counts: 
        for idx in range (0, unique_ids.shape[0]): 
             
            if unique_ids[idx] == 1: 
                LVLA_bin += counts[unique_ids[idx]] 
 
            if unique_ids[idx] == 2: 
                LVHA_bin += counts[unique_ids[idx]] 
 
            if unique_ids[idx] == 3: 
                HVLA_bin += counts[unique_ids[idx]] 
 
            if unique_ids[idx] == 4: 
                HVHA_bin += counts[unique_ids[idx]] 
 
 
        # print(f"LVHA_bin: {LVHA_bin}, HVHA_bin: {HVHA_bin}, LVLA_bin: {LVLA_bin}, HVLA_bin: {HVLA_bin}") 
 
        # bins = { 
        #     "LVHA_bin": LVHA_bin, 
        #     "HVHA_bin": HVHA_bin, 
        #     "LVLA_bin": LVLA_bin, 
        #     "HVLA_bin": HVLA_bin 
        # }     

        # bins = { 
        #     "LVLA": LVLA_bin, 
        #     "LVHA": LVHA_bin, 
        #     "HVLA": HVLA_bin, 
        #     "HVHA": HVHA_bin, 
        # } 

        bins = { 
            labels[0]: LVLA_bin, 
            labels[1]: LVHA_bin, 
            labels[2]: HVLA_bin, 
            labels[3]: HVHA_bin, 
        } 
 
        bins_df = pd.DataFrame(list(bins.items()), columns=['bin', 'count']) 
 
        dominant_emotion = bins_df.loc[bins_df['count'].idxmax(), 'bin'] 
         
        # assign dominant emotion to all rows in xf_plot with the current cluster_id 
        mask = xf_plot['cluster_id'] == cluster_id 
        xf_plot.loc[mask, 'dominant_emotion'] = dominant_emotion 
 
    # check assigned dominant emotions: 
    xf_plot['dominant_emotion'].value_counts() 
 
    # check for NaN values in the 'dominant_emotion' column 
    # print("Has NaNs", xf_plot['dominant_emotion'].isna().values.any()) 
     
    return xf_plot.copy()


def map_cluster_to_dominant_emotion_5_non_overlapping_bins(xf_plot, n_components): 
 
    for cluster_id in range(n_components): 
         
        LVHA_bin = 0 
        HVHA_bin = 0 
        LVLA_bin = 0 
        HVLA_bin = 0
        NEUTRAL_bin = 0 
        
 
        # compute cell_id counts: 
        counts = xf_plot[xf_plot['cluster_id'] == cluster_id].loc[:, 'cell_id'].value_counts() 
        # Access the cell_id values 
        unique_ids = counts.index 
 
        # fill emotion bins with cell_id counts: 
        for idx in range (0, unique_ids.shape[0]): 
             
            if unique_ids[idx] == 1: 
                LVLA_bin += counts[unique_ids[idx]] 
 
            if unique_ids[idx] == 2: 
                LVHA_bin += counts[unique_ids[idx]] 
 
            if unique_ids[idx] == 3: 
                HVLA_bin += counts[unique_ids[idx]] 
 
            if unique_ids[idx] == 4: 
                HVHA_bin += counts[unique_ids[idx]] 
 
            if unique_ids[idx] == 5: 
                NEUTRAL_bin += counts[unique_ids[idx]] 
 
        # print(f"LVHA_bin: {LVHA_bin}, HVHA_bin: {HVHA_bin}, LVLA_bin: {LVLA_bin}, HVLA_bin: {HVLA_bin}") 
 
        # bins = { 
        #     "LVHA_bin": LVHA_bin, 
        #     "HVHA_bin": HVHA_bin, 
        #     "LVLA_bin": LVLA_bin, 
        #     "HVLA_bin": HVLA_bin 
        # }     

        # bins = { 
        #     "LVLA": LVLA_bin, 
        #     "LVHA": LVHA_bin, 
        #     "HVLA": HVLA_bin, 
        #     "HVHA": HVHA_bin, 
        # } 

        bins = { 
            labels[0]: LVLA_bin, 
            labels[1]: LVHA_bin, 
            labels[2]: HVLA_bin, 
            labels[3]: HVHA_bin, 
            labels[4]: NEUTRAL_bin
        } 
 
        bins_df = pd.DataFrame(list(bins.items()), columns=['bin', 'count']) 
 
        dominant_emotion = bins_df.loc[bins_df['count'].idxmax(), 'bin'] 
         
        # assign dominant emotion to all rows in xf_plot with the current cluster_id 
        mask = xf_plot['cluster_id'] == cluster_id 
        xf_plot.loc[mask, 'dominant_emotion'] = dominant_emotion 
 
    # check assigned dominant emotions: 
    xf_plot['dominant_emotion'].value_counts() 
 
    # check for NaN values in the 'dominant_emotion' column 
    # print("Has NaNs", xf_plot['dominant_emotion'].isna().values.any()) 
     
    return xf_plot.copy()


def map_cluster_to_dominant_emotion(xf_plot, n_components, number_of_bins=NUMBER_OF_BINS):

    if number_of_bins == 4:
        xf_plot_nv = map_cluster_to_dominant_emotion_4_non_overlapping_bins(
            xf_plot,
            n_components
        )

    elif number_of_bins == 5:
        xf_plot_nv = map_cluster_to_dominant_emotion_5_non_overlapping_bins(
            xf_plot,
            n_components
        )

    elif number_of_bins == 9:
        xf_plot_nv = map_cluster_to_dominant_emotion_4_overlapping_bins(
            xf_plot,
            n_components
        )
    else:
        raise ValueError(
            f"number_of_bins must be either 4 or 9, but got {number_of_bins}"
        )
    return xf_plot_nv

def safe_div(numerator, denominator, default=0.0):
    return numerator / denominator if denominator else default

def build_emotion_significance_matrix(df, labels=('A', 'B', 'R', 'S')):
    """
    Build a symmetric matrix whose (i,j) entry is the median value of all
    comparisons between dominant emotions i and j using the lower triangle
    of the input matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Square matrix whose rows and columns correspond to dominant emotions.
        The index and/or columns may be either a simple Index or a MultiIndex.
        Values may be floats (e.g., p-values) or binary values.

    labels : iterable of str
        Complete set of emotion labels.

    Returns
    -------
    pd.DataFrame
        Symmetric matrix of median values.

    Notes
    -----
    If one or more emotions are absent from the participant (i.e., they never
    appear as dominant emotions in the rows), then the corresponding row and
    column of the output matrix are filled with NaN.
    """

    if df.shape[0] != df.shape[1]:
        raise ValueError(
            f"df must be a square matrix, but got shape {df.shape}"
        )
    else:
        labels_list = list(labels)

        label_order = {label: i for i, label in enumerate(labels_list)}

        # -------- Row labels --------
        if isinstance(df.index, pd.MultiIndex):
            row_labels = list(df.index.get_level_values(-1))
        else:
            row_labels = list(df.index)

        # -------- Column labels --------
        if isinstance(df.columns, pd.MultiIndex):
            col_labels = list(df.columns.get_level_values(-1))
        else:
            col_labels = list(df.columns)

        # Store all values for each unordered emotion pair
        values = {
            pair: []
            for pair in combinations_with_replacement(labels_list, 2)
        }
        n_rows, n_cols = df.shape

        for col in range(n_cols):
            for row in range(col, n_rows):      # lower triangle only

                # key = tuple(sorted((row_labels[row], col_labels[col])))
                key = tuple(
                    sorted(
                        (row_labels[row], col_labels[col]),
                        key=lambda x: label_order[x]
                    )
                )

                if key in values:
                    value = df.iat[row, col]

                    if pd.notna(value):
                        values[key].append(float(value))

        # Compute medians
        medians = {
            pair: np.median(vals) if vals else np.nan
            for pair, vals in values.items()
        }

        # Build symmetric matrix
        matrix = pd.DataFrame(index=labels_list, columns=labels_list, dtype=float)

        for (a, b), value in medians.items():
            matrix.loc[a, b] = value
            matrix.loc[b, a] = value

        # ------------------------------------------------------------
        # Mask emotions that are absent from the participant
        # ------------------------------------------------------------
        present = set(row_labels)
        missing = set(labels) - present

        for emotion in missing:
            matrix.loc[emotion, :] = np.nan
            matrix.loc[:, emotion] = np.nan

        return matrix

def build_dunn_matrix_with_emotion_labels(dunn_combined, xf_plot_dom_emot):

    """
    This function takes a matrix of Dunn's test results and a DataFrame containing 
    dominant emotions for each cluster, and returns a new DataFrame where the columns 
    are labeled with the corresponding dominant emotions. 
    It also creates a MultiIndex for the rows, 
    pairing each original index value with the emotion label.

    Parameters:
    dunn_combined : pd.DataFrame
        A matrix (DataFrame) where each cell indicates whether the difference 
        between two clusters is statistically significant (p-value).
    
    Returns:
    pd.DataFrame

    """

    dunn_combined_copy= dunn_combined.copy()

    emotion_table = (
        xf_plot_dom_emot.groupby(['dominant_emotion', 'cluster_id'], observed=False)
        .size()
        .unstack(fill_value=0)
    )

    dominant_emotions_series = emotion_table.idxmax()

    single_row_table = (
        dominant_emotions_series
        .to_frame()
        .transpose()
        .rename(index={0: 'dominant_emotion: '})  # Fixes the row label from 0 to name
    )

    print(single_row_table)

    # Assign the dominant emotion label to the columns names
    dunn_combined_copy.columns=single_row_table.iloc[0,:]

    # Make sure it's int first (remember: astype returns a copy!)
    # dunn_combined_copy = dunn_combined_copy.astype(int)

    # Build a MultiIndex pairing each original index value with the emotion label
    dunn_combined_copy.index = pd.MultiIndex.from_arrays(
        [dunn_combined_copy.index, dunn_combined_copy.columns.to_numpy()],
        names=["pos", "emotion"]
    )

    return dunn_combined_copy   

def print_lower_triangular_red(df, include_diagonal=False):
    RED, RESET = "\033[91m", "\033[0m"
    col_w = 3
    # width reserved for the (pos, emotion) row labels
    idx_w = max(len(str(x)) for lvl in zip(*df.index) for x in lvl) + 4

    # header
    print(" " * idx_w + f"{df.columns.name}")
    print(" " * idx_w + "".join(f"{str(c):>{col_w}}" for c in df.columns))

    for i, (pos, emo) in enumerate(df.index):
        row_label = f"{str(pos):<3}{str(emo):<{idx_w-3}}"
        cells = []
        for j in range(df.shape[1]):
            val = str(df.iat[i, j])
            below = i >= j if include_diagonal else i > j
            cells.append(f"{RED}{val:>{col_w}}{RESET}" if below else f"{val:>{col_w}}")
        print(row_label + "".join(cells))

def print_p_values_binary_matrix_proportions(emotion_pairs_data, xtick_labels, emotion_type="Valence"):

    # emotion_pairs_data_reshaped = np.column_stack(emotion_pairs_data)
    emotion_pairs_data_reshaped = np.array(emotion_pairs_data).T
    emotion_pairs_data_df = pd.DataFrame(emotion_pairs_data_reshaped, columns=xtick_labels)

    print("--------------------------------------")
    print(f"{emotion_type} p-values for each emotion pair across all participants:")
    print(emotion_pairs_data_df)

    emotion_pairs_data_df_binary = (
        (emotion_pairs_data_df < P_VALUE)
        .where(emotion_pairs_data_df.notna())
        .astype("Int64")
    )

    print("--------------------------------------")
    print(f"Binary matrix of statistically significant comparisons (p < {P_VALUE}) for each emotion pair ({emotion_type}):")
    print(emotion_pairs_data_df_binary)

    proportions = emotion_pairs_data_df_binary.mean(skipna=True)
    print("--------------------------------------")
    print(f"Proportions of statistically significant comparisons (p < {P_VALUE}) for each emotion pair ({emotion_type}):")
    print(proportions)

    return emotion_pairs_data_df_binary

def emotion_significance_box_plot(all_participants_emotion_matrix, labels=('A', 'B', 'R', 'S'), plots_dir='plots', folder_name='folder_name', emotion_type = "Valence"):
    """
    Create a box plot of significance p-values for each emotion pair across all participants.

    Parameters
    ----------
    all_participants_emotion_matrix : np.ndarray
        A 3D array of shape (n_participants, n_emotions, n_emotions) containing significance p-values.
    
    labels : tuple of str
        The labels for the emotions.
    
    plots_dir : str
        Directory to save the plot.
    
    folder_name : str
        Name of the folder for saving the plot.

    Returns
    -------
    emotion_pairs_data: list of np.ndarray
        A list containing arrays of significance p-values for each emotion pair.
    xtick_labels: list of str
        A list of labels for the x-axis ticks in the box plot.
    """

    # {('A','A'): [v_p1, v_p2, ...], ('A','B'): [...], ...}
    pair_values = {}

    for a, b in combinations_with_replacement(labels, 2):
        i, j = labels.index(a), labels.index(b)
        pair_values[(a, b)] = all_participants_emotion_matrix[:, i, j]   # one value per participant

    pairs = list(pair_values.keys())                       # 10 pairs, ordered
    emotion_pairs_data  = [pair_values[p] for p in pairs]                # list of 30-value arrays
    xtick_labels = [f"({a}, {b})" for a, b in pairs]       # (A, A), (A, B), ...

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        emotion_pairs_data,
        # labels=xtick_labels,
        medianprops={
            "color": "r",
            "linewidth": 2
        },
        # notch=True,
        ax=ax
    )
    ax.set_xlabel('Emotion pair')
    ax.set_ylabel('Significance p-value')
    ax.set_title(f'Emotion significance per pair across all participants ({emotion_type})')
    ax.set_xticks(range(len(xtick_labels)))        # ensure one tick per box
    ax.set_xticklabels(xtick_labels, rotation=45)  # apply your (A, A), (A, B), … labels
    plt.tight_layout()
    # plt.show()

    fig_path = os.path.join(
        plots_dir,
        f'{folder_name}_significance_p_value_{emotion_type}.jpg'
    )

    # plt.show()

    try:
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"Error saving figure for participant {i}")
        print(exc)

    return emotion_pairs_data, xtick_labels



def add_labels_to_matrix(df, labels=('R1', 'R2', 'R3', 'R4', 'R5')):

    if df.shape[0] != df.shape[1]:
        raise ValueError(
            f"df must be a square matrix, but got shape {df.shape}"
        )

    if df.shape[0] != len(labels):
        raise ValueError(
            f"The number of labels ({len(labels)}) must match "
            f"the matrix dimensions ({df.shape[0]})"
        )

    df1 = df.copy()
    df1.index = labels
    df1.columns = labels

    return df1


#%%----------------------------------


def plot_population_valence_arousal_regions(
    xf_all_participants,
    plots_dir,
    folder_name,
    annotate_points=False
):
    """
    Plot all participants' valence-arousal observations and quantify
    observations assigned to R1-R5 using map_label_to_5_bins().

    Each participant has a distinct color and appears in the legend.
    Set annotate_points=True to print the participant ID beside every point.
    """
    from matplotlib.patches import Polygon

    required_columns = {
        'participant', 'val_chunks_number', 'aro_chunks_number'
    }
    missing = required_columns.difference(xf_all_participants.columns)
    if missing:
        raise KeyError(f"Missing columns required for population plot: {sorted(missing)}")

    data = xf_all_participants[
        ['participant', 'val_chunks_number', 'aro_chunks_number']
    ].copy()

    # Exclude observations without valid valence/arousal values.
    data = data.dropna(
        subset=['participant', 'val_chunks_number', 'aro_chunks_number']
    )

    region_names = {
        1: 'R1 (low valence / low arousal)',
        2: 'R2 (low valence / high arousal)',
        3: 'R3 (high valence / low arousal)',
        4: 'R4 (high valence / high arousal)',
        5: 'R5 (neutral)'
    }

    # Use the same classification rule already used by the main script.
    data['region_id'] = data.apply(
        lambda row: map_label_to_5_bins(
            row['val_chunks_number'], row['aro_chunks_number']
        ),
        axis=1
    )
    data['region'] = data['region_id'].map(region_names)

    # ---------------------------------------------------------
    # Overall counts and percentages
    # ---------------------------------------------------------
    region_order = [1, 2, 3, 4, 5]
    total_valid = len(data)

    counts = (
        data['region_id']
        .value_counts()
        .reindex(region_order, fill_value=0)
    )
    summary = pd.DataFrame({
        'region': [region_names[k] for k in region_order],
        'count': counts.values,
        'percentage': (
            100 * counts.values / total_valid if total_valid else 0
        )
    })

    unassigned = int(data['region_id'].isna().sum())
    print("\\n====================================================")
    print("POPULATION-LEVEL VALENCE-AROUSAL REGION DISTRIBUTION")
    print("====================================================")
    print(summary.to_string(index=False, formatters={
        'percentage': lambda value: f'{value:.2f}%'
    }))
    print(f"Total valid observations: {total_valid}")
    print(f"Unassigned observations: {unassigned}")

    # ---------------------------------------------------------
    # Participant-by-region counts and within-participant %
    # ---------------------------------------------------------
    participant_counts = pd.crosstab(
        data['participant'],
        data['region_id']
    ).reindex(columns=region_order, fill_value=0)

    participant_counts.columns = [f'R{k}' for k in region_order]
    participant_counts['Total'] = participant_counts.sum(axis=1)

    participant_percentages = (
        participant_counts[[f'R{k}' for k in region_order]]
        .div(participant_counts['Total'].replace(0, np.nan), axis=0)
        * 100
    )
    participant_percentages.columns = [
        f'R{k}_percent' for k in region_order
    ]

    participant_summary = pd.concat(
        [participant_counts, participant_percentages], axis=1
    )

    print("\\nCounts per participant:")
    print(participant_counts.to_string())
    print("\\nPercentages within each participant:")
    print(participant_percentages.round(2).to_string())

    os.makedirs(plots_dir, exist_ok=True)
    summary.to_csv(
        os.path.join(plots_dir, f'{folder_name}_population_region_counts.csv'),
        index=False
    )
    participant_summary.to_csv(
        os.path.join(plots_dir, f'{folder_name}_participant_region_counts.csv')
    )

    # ---------------------------------------------------------
    # Population scatter plot
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 9))

    # Region polygons follow the same 4.5 / 5.0 / 5.5
    # boundary logic used by map_label_to_5_bins().
    region_polygons = [
        ('R1', [(1, 1), (5, 1), (5, 4.5), (4.5, 4.5),
                (4.5, 5), (1, 5)], '#5B8CC0', (2.7, 2.7)),
        ('R2', [(1, 5), (4.5, 5), (4.5, 5.5), (5, 5.5),
                (5, 9), (1, 9)], '#F2C84B', (2.7, 7.2)),
        ('R3', [(5, 1), (9, 1), (9, 5), (5.5, 5),
                (5.5, 4.5), (5, 4.5)], '#F4B18F', (7.2, 2.7)),
        ('R4', [(5, 5.5), (5, 9), (9, 9), (9, 5),
                (5.5, 5), (5.5, 5.5)], '#45C5AD', (7.2, 7.2)),
        ('R5', [(4.5, 4.5), (5.5, 4.5), (5.5, 5.5), (4.5, 5.5)],
         '#F08080', (5, 5))
    ]

    for label, vertices, color, label_position in region_polygons:
        ax.add_patch(Polygon(
            vertices,
            closed=True,
            facecolor=color,
            edgecolor='white',
            linewidth=1.2,
            alpha=0.48,
            zorder=0
        ))
        ax.text(
            label_position[0], label_position[1], label,
            ha='center', va='center', fontsize=17,
            fontweight='bold', color='black', alpha=0.8, zorder=1
        )

    participants_in_data = sorted(data['participant'].unique())
    cmap = plt.get_cmap('tab20', max(len(participants_in_data), 1))

    for color_index, participant_id in enumerate(participants_in_data):
        participant_data = data[data['participant'] == participant_id]
        color = cmap(color_index)

        ax.scatter(
            participant_data['val_chunks_number'],
            participant_data['aro_chunks_number'],
            color=[color],
            s=24,
            alpha=0.75,
            edgecolors='none',
            label=f'Participant {participant_id}',
            zorder=2
        )

        if annotate_points:
            for _, point in participant_data.iterrows():
                ax.annotate(
                    str(participant_id),
                    (point['val_chunks_number'], point['aro_chunks_number']),
                    fontsize=5,
                    alpha=0.65,
                    xytext=(2, 2),
                    textcoords='offset points',
                    zorder=3
                )

    ax.set_xlim(1, 9)
    ax.set_ylim(1, 9)
    ax.set_xticks([1, 4.5, 5, 5.5, 9])
    ax.set_yticks([1, 4.5, 5, 5.5, 9])
    ax.set_xlabel('Valence', fontsize=12)
    ax.set_ylabel('Arousal', fontsize=12)
    ax.set_title('Population-level Valence vs Arousal', fontsize=14)
    ax.grid(alpha=0.2, zorder=0)

    # Put the participant legend outside the plotting area.
    ax.legend(
        title='Participant',
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        fontsize=8,
        ncol=1
    )

    fig.tight_layout()
    fig.savefig(
        os.path.join(
            plots_dir,
            f'{folder_name}_population_valence_arousal_regions.jpg'
        ),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    return data, summary, participant_summary


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

        participant_plots_dir = os.path.join(
            os.path.dirname(__file__),
            'plots',
            f'participant_{i}'
        )

        plots_dir = os.path.join(
            os.path.dirname(__file__),
            'plots',
        )

        os.makedirs(participant_plots_dir, exist_ok=True)

        folder_name = os.path.basename(
            os.path.dirname(os.path.abspath(__file__))
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
            plot_clustering_and_label_distribution_per_cluster (xf, xf_plot, low_dimensional_xf, clusters, means, participant_plots_dir, folder_name, n_components=n_components)

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
        
        participant_data_list.append(xf_plot.copy())

        print(f"participant {i}")
        print(f"len(participant_data_list) = {len(participant_data_list)}")
        print(f"id(xf_plot) = {id(xf_plot)}")

        plot_cellID_vs_video(xf_plot, participant_plots_dir, folder_name, n_components)

        # xf_plot_dom_emot is the dataset with the dominant emotion assigned to each cluster, based on the number of bins (4, 5 or 9)
        xf_plot_dom_emot = map_cluster_to_dominant_emotion(xf_plot, n_components, number_of_bins=NUMBER_OF_BINS)

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
        # Print the formatted table
        print(emotion_table)
        print("It counts the number of annotated window segments that fall into each emotion bin (A, B, R, S), for each cluster")

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


