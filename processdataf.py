import numpy as np
import random as python_random
import socket
import os
import sys
from pathlib import Path

# https://stackoverflow.com/questions/8663076/python-best-way-to-add-to-sys-path-relative-to-the-current-running-script
from definitions import *
from functions import * #proyect's functions / defs

# The Linux config is selected on Linux. On Windows, the default config covers
# the second PC; add host-specific entries below for other layouts.
WINDOWS_DATASET_CONFIG = {
    "root": Path(r"C:\Users\mbamo\Desktop\Datasets\CASE"),
    "data_subdir": Path(""),
    "interpolated_subdir": Path("interpolated"),
    "metadata_subdir": Path("metadata"),
}
LINUX_DATASET_CONFIG = {
    "root": Path("/home/marcos/datasets/case"),
    "data_subdir": Path(""),
    "interpolated_subdir": Path(""),
    "metadata_subdir": Path(""),
}
HOST_DATASET_CONFIGS = {
    "LAPTOP-R7AHG17P": {
        "root": Path(r"C:\Users\Javier\Desktop\CASE_full\CASE_full"),
        "data_subdir": Path("data"),
    },
    # Add a new PC by hostname; override subdirectory values if its layout differs.
    # "NEW-PC-HOSTNAME": {
    #     "root": Path(r"D:\Datasets\CASE"),
    #     "data_subdir": Path(""),
    # },
}

HOSTNAME = socket.gethostname()
DATASET_CONFIG = dict(
    LINUX_DATASET_CONFIG if sys.platform.startswith("linux") else WINDOWS_DATASET_CONFIG
)
DATASET_CONFIG.update(HOST_DATASET_CONFIGS.get(HOSTNAME, {}))


def process_data(participant):

    # https://keras.io/getting_started/faq/#how-to-do-hyperparameter-tuning-with-keras
    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    np.random.seed(123)

    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    python_random.seed(123)


    try:
        participant
        print("participant: ", participant)
    except NameError:
        # assign a participant in order to run the whole script:
        participant = str(24)
        

        print("           ")
        print("-------  ----------------------------------")
        print("-----------------------------------------")
        print("WARNING:")
        print("No participant was passed as argument: this file will be used instead: ", r"C:\Users\mbamo\Desktop\Datasets\K-EmoCon\data for vpn pc\E4_" + r"BVP_p" + participant + r"t.xlsx")
        # print("No participant was passed as argument: this file will be used instead: ", fname + r"BVP_p" + participant + r"t.xlsx")
        print("-----------------------------------------")
        print("-----------------------------------------")
        print("           ")

    # ---  Load participant Data

    if HOSTNAME == "LAPTOP-R7AHG17P":
        print("corriendo en PC gamer:")
        sys.stdout.reconfigure(encoding="utf-8")

    data_dir = (
        DATASET_CONFIG["root"]
        / DATASET_CONFIG["data_subdir"]
        / DATASET_CONFIG["interpolated_subdir"]
    )
    metadata_dir = DATASET_CONFIG["root"] / DATASET_CONFIG["metadata_subdir"]
    physiological_file = data_dir / "physiological" / f"sub_{participant}.csv"
    annotation_file = data_dir / "annotations" / f"sub_{participant}.csv"
    seqs_order_num_file = metadata_dir / "seqs_order_num_csv.csv"

    # Use numpy's loadtxt function to load the data
    data = pd.read_csv(physiological_file)
    seqs_order_num = pd.read_csv(seqs_order_num_file)

    # time_vector = data['daqtime']/(1000*60)  # [minutes] 
    time_vector = data['daqtime']  # [ms] 

    # load annotations
    annotations = pd.read_csv(annotation_file)
    valence = annotations['valence']
    arousal = annotations['arousal']

    stimulus_tag_list = [map_video_to_tag(i) for i in data['video']] 

    data['tag'] = stimulus_tag_list  # dataframe now contains the stimulus tag of each video

    [ gsr_mc, ppg_mc ] = baseline_mean_centered_normalization(data, show_plot=0)

    gsr_signals, gsr_info = filter_gsr(gsr_mc, show_plot=0)

    ppg_signals = filter_ppg(ppg_mc, show_plot=0)

    # test windowing
    window_size = int(3*1000)  # ms
    perc_overlap = 0
    step = int(window_size * (1-perc_overlap))
    overlap = True

    print("Window size (sec): ", window_size/1000)
    print("step (sec): ", step/1000)
    print("overlap: ", overlap)
    if overlap:
        print("perc. of overlap: ", (window_size-step)*100/window_size)

    # Ensures windows_size and step are compatible to Deep Learning arquitecture and pooling operations
    window_size = nearest_multiple(window_size, multiple=16)
    step = nearest_multiple(step, multiple=16)  #ensures perc. of overlap is exact
            
    combined_df = perform_windowing(data, ppg_signals, gsr_signals, valence, arousal, seqs_order_num, participant, window_size, step, overlap, show_plot = 0)

    combined_df_median = median_voting(combined_df)

    combined_df_mapped = map_valence_arousal(combined_df_median, scheme = 'bipartite', number_thresholds = 1, threshold = 5, L = 3.6, H = 4.4)

    X_windowed, y_median = build_dataset(combined_df_mapped)

    xf = feature_extract_GSR_PPG_non_linear(X_windowed)
    output_dir = os.path.dirname(os.path.abspath(__file__))
    xf.to_pickle(os.path.join(output_dir, f'xf{participant}.pkl'))
    y_median.to_pickle(os.path.join(output_dir, f'y_median{participant}.pkl'))

    # # JUST FOR TESTING PURPOSES:
    # xf_retrieved_pickle = pd.read_pickle('xf.pkl')
    # print("\nRetrieved DataFrame from pickle file:")
    # print(xf_retrieved_pickle)
    # xf = xf_retrieved_pickle.copy()

    #%%----------------------------------------------------------------
    # xf # feature vector (in DataFrame format)
    # y_median  # label vector (in DataFrame format)

    # gsr_ts_arousal, ppg_ts_arousal, gsr_ts_valence, ppg_ts_valence, valence, arousal = format_dataset_for_DL(X_windowed, y_median)

    # arousal_d_one_class, gsr_arousal_d_one_class, valence_d_one_class, gsr_valence_d_one_class = imbalance_test(valence, arousal, show_plots = False)

    # # ---- ALG TEST -----
    # knn = KNeighborsClassifier(n_neighbors=5)
    # dt = DecisionTreeClassifier()
    # rf = RandomForestClassifier(n_estimators=5)
    # svm = SVC(C=0.1)
    # gbm = GradientBoostingClassifier(n_estimators=5)
    # performance = alg_performance_eval(xf, y_median, knn, dt, rf, svm, gbm, arousal_d_one_class, valence_d_one_class, gsr_arousal_d_one_class, gsr_valence_d_one_class, gsr_ts_arousal, ppg_ts_arousal, arousal, gsr_ts_valence, ppg_ts_valence, valence)


    # print("-------------------------------------")
    # print("-------------------------------------")
    # print("participant: ", participant)
    # print("-------------------------------------")
    # print("----- RESULTS ------")
    # print("-------------------------------------")
    # print("Window size (sec): ", window_size/1000)
    # print("step (sec): ", step/1000)
    # print("overlap: ", overlap)
    # if overlap:
    #     print("perc. of overlap: ", (window_size-step)*100/window_size)
    # print("Number of windows / instances: ", X_windowed.shape[0])
    # print("-rows: alg : KNN = 0; DT = 1; RF = 2; SVM = 3; GBM = 4; BDDAE = 5; DUMMY = 5")
    # print("columns:")
    # print("'val_cohen','val_uar', 'val_acc', 'val_gm',  'val_f1',  'aro_cohen','aro_uar', 'aro_acc', 'aro_gm',  'aro_f1'")
    # print("-------------------------------------------------------------")
    # print("  v_c   v_u   v_a   v_g   v_f1  a_c   a_u   a_a   a_g   a_f1")
    # # print("-columns: [(val_cohen, val_uar, val_acc,val_gm, val_f1, aro_cohen, aro_uar,aro_acc,aro_gm, aro_f1)]")
    # print(np.round(performance, decimals=3))

    # return performance, window_size, step, overlap, X_windowed.shape[0]

if __name__ == "__main__":
    process_data('28')

