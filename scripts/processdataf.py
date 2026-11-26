#%%----------------------------------------------------------------

import numpy as np
import random as python_random
import socket
# import torch
import time

# ---------------  import constants and functions:
# https://stackoverflow.com/questions/15514593/importerror-no-module-named-when-trying-to-run-python-script
import sys, os

# https://stackoverflow.com/questions/8663076/python-best-way-to-add-to-sys-path-relative-to-the-current-running-script
from definitions import *
from functions import * #proyect's functions / defs

def process_data(participant, available_saved_data = False):

    # https://keras.io/getting_started/faq/#how-to-do-hyperparameter-tuning-with-keras
    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    # np.random.seed(123)

    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    # python_random.seed(123)


    try:
        participant
        print("participant: ", participant)
    except NameError:
        # assign a participant in order to run the whole script:
        participant = str(28)  # original 28

        

        print("           ")
        print("-------  ----------------------------------")
        print("-----------------------------------------")
        print("WARNING:")
        print("No participant was passed as argument: this file will be used instead: ", r"C:\Users\mbamo\Desktop\Datasets\K-EmoCon\data for vpn pc\E4_" + r"BVP_p" + participant + r"t.xlsx")
        # print("No participant was passed as argument: this file will be used instead: ", fname + r"BVP_p" + participant + r"t.xlsx")
        print("-----------------------------------------")
        print("-----------------------------------------")
        print("           ")

    # Windowing
    window_size = int(3*1000)  # ms
    perc_overlap = 0.0
    step = int(window_size * (1-perc_overlap))
    overlap = True

    print("Window size (sec): ", window_size/1000)
    print("step (sec): ", step/1000)
    print("overlap: ", overlap)
    if overlap:
        print("perc. of overlap: ", (window_size-step)*100/window_size)
        print("overlap duration (sec): ", window_size*perc_overlap/1000)

    # Ensures windows_size and step are compatible to Deep Learning arquitecture and pooling operations
    window_size = nearest_multiple(window_size, multiple=16)
    step = nearest_multiple(step, multiple=16)  #ensures perc. of overlap is exact

    # ---  Load participant Data

    if available_saved_data:

        # Add the absolute path to the system path
        # if features and labels are already saved, then load them:
        # xf = pd.read_pickle('xf' + participant +'.pkl')  

        if platform == "win32":
            if socket.gethostname() == "LAPTOP-R7AHG17P":  # pc gamer (Javier)
                print("corriendo en PC gamer:")                
                xf = pd.read_pickle(os.path.join(r'C:\Users\Javier\Dropbox\Clustering\c_sldl_1_2_1_reloaded','xf' + participant +'.pkl'))
                y_median = pd.read_pickle(os.path.join(r'C:\Users\Javier\Dropbox\Clustering\c_sldl_1_2_1_reloaded','y_median' + participant + '.pkl'))  #for testing purposes
                # xf = pd.read_pickle(os.path.join(r'C:\Users\Javier\Dropbox\pykan','xf' + participant +'.pkl'))
                # y_median = pd.read_pickle(os.path.join(r'C:\Users\Javier\Dropbox\pykan','y_median' + participant + '.pkl'))  #for testing purposes
                
            else: # MFB PC
                print("corriendo en PC MFB:")
                xf = pd.read_pickle(os.path.join(r'C:\Users\mbamo\Dropbox\Clustering\c_sldl_1_2_1_reloaded','xf' + participant +'.pkl'))
                y_median = pd.read_pickle(os.path.join(r'C:\Users\mbamo\Dropbox\Clustering\c_sldl_1_2_1_reloaded','y_median' + participant + '.pkl'))  #for testing purposes

        # add segment number to the features dataframe:
        xf['segment_number'] = range(1, len(xf) + 1)
        xf['participant'] = participant
        xf['participant'] = xf['participant'].astype('category')

        if platform == "linux" or platform == "linux2":

            # current_folder=$(pwd)
            # file_path="$current_folder/subfolder_name/file_name"

            # physiological_base = r"./physiological/sub_"
            # annotation_base = r"./annotations/sub_"
            # seqs_order_num_base = r"seqs_"
            physiological_base = r"/home/marcos/datasets/case/physiological/sub_"
            annotation_base = r"/home/marcos/datasets/case/annotations/sub_"
            seqs_order_num_base = r"//home/marcos/datasets/case/seqs_"
        elif platform == "win32":
            if socket.gethostname() == "LAPTOP-R7AHG17P":  # pc gamer (Javier)
                print("corriendo en PC gamer:")
                physiological_base = r"C:\Users\Javier\Desktop\CASE_full\CASE_full\data\interpolated\physiological\sub_"
                annotation_base = r"C:\Users\Javier\Desktop\CASE_full\CASE_full\data\\interpolated\annotations\sub_"
                seqs_order_num_base = r"C:\Users\Javier\Desktop\CASE_full\CASE_full\metadata\seqs_"
                # Set the default encoding
                sys.stdout.reconfigure(encoding='utf-8')
            else:
                physiological_base = r"C:\Users\mbamo\Datasets\CASE\interpolated\physiological\sub_"
                annotation_base = r"C:\Users\mbamo\Datasets\CASE\interpolated\annotations\sub_"
                seqs_order_num_base = r"C:\Users\mbamo\Datasets\CASE\metadata\seqs_"

        
        # Specify the path to your text file
        physiological_file = physiological_base + participant + '.csv'
        annotation_file = annotation_base + participant + '.csv'
        seqs_order_num_file = seqs_order_num_base + 'order_num_csv' + '.csv'

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
        
        combined_df = perform_windowing(data, ppg_signals, gsr_signals, valence, arousal, seqs_order_num, participant, window_size, step, overlap, show_plot = 0)

        combined_df_median = median_voting(combined_df)

        # Add two new columns with the first value of each pandas Series (valence or arousal value of the segment)
        xf['val_chunks_number'] = combined_df_median['val_chunks'].apply(
            lambda s: s.iloc[0] if isinstance(s, pd.Series) else s[0]
        ).values

        xf['aro_chunks_number'] = combined_df_median['aro_chunks'].apply(
            lambda s: s.iloc[0] if isinstance(s, pd.Series) else s[0]
        ).values

        xf['video'] = combined_df_median['video'].values

        print("-----")


    else:

        if platform == "linux" or platform == "linux2":

            # current_folder=$(pwd)
            # file_path="$current_folder/subfolder_name/file_name"

            # physiological_base = r"./physiological/sub_"
            # annotation_base = r"./annotations/sub_"
            # seqs_order_num_base = r"seqs_"
            physiological_base = r"/home/marcos/datasets/case/physiological/sub_"
            annotation_base = r"/home/marcos/datasets/case/annotations/sub_"
            seqs_order_num_base = r"//home/marcos/datasets/case/seqs_"
        elif platform == "win32":
            if socket.gethostname() == "LAPTOP-R7AHG17P":  # pc gamer (Javier)
                print("corriendo en PC gamer:")
                physiological_base = r"C:\Users\Javier\Desktop\CASE_full\CASE_full\data\interpolated\physiological\sub_"
                annotation_base = r"C:\Users\Javier\Desktop\CASE_full\CASE_full\data\\interpolated\annotations\sub_"
                seqs_order_num_base = r"C:\Users\Javier\Desktop\CASE_full\CASE_full\metadata\seqs_"
                # Set the default encoding
                sys.stdout.reconfigure(encoding='utf-8')
            else:
                physiological_base = r"C:\Users\mbamo\Datasets\CASE\interpolated\physiological\sub_"
                annotation_base = r"C:\Users\mbamo\Datasets\CASE\interpolated\annotations\sub_"
                seqs_order_num_base = r"C:\Users\mbamo\Datasets\CASE\metadata\seqs_"

        
        # Specify the path to your text file
        physiological_file = physiological_base + participant + '.csv'
        annotation_file = annotation_base + participant + '.csv'
        seqs_order_num_file = seqs_order_num_base + 'order_num_csv' + '.csv'

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
        
        combined_df = perform_windowing(data, ppg_signals, gsr_signals, valence, arousal, seqs_order_num, participant, window_size, step, overlap, show_plot = 0)

        combined_df_median = median_voting(combined_df)

        combined_df_mapped = map_valence_arousal(combined_df_median, scheme = 'bipartite', number_thresholds = 1, threshold = 5, L = 3.6, H = 4.4)

        X_windowed, y_median = build_dataset(combined_df_mapped)

        # linear features (used in DEAP and K-EmoCon)
        # xf = feature_extract_GSR_PPG_dataframe(X_windowed)

        # compute features and save in a file
        xf = feature_extract_GSR_PPG_non_linear(X_windowed)

        xf.to_pickle("./xf"+participant+".pkl") # save features to a file
        y_median.to_pickle("./y_median"+participant+".pkl") # save labels to a file
            
        time.sleep(5)       

        #%%-----------------------------------------------------

        # # # JUST FOR TESTING PURPOSES:
        # xf_retrieved_pickle = pd.read_pickle('xf.pkl')
        # # xf_retrieved_pickle = pd.read_pickle('xf'+participant+'.pkl')
        # print("\nRetrieved DataFrame from pickle file:")
        # print(xf_retrieved_pickle)
        # xf = xf_retrieved_pickle.copy()


    #%%----------------------------------------------------------------
    # ---- ALG TEST -----
    # xf

    # performance, aro_feature_score, val_feature_score = alg_performance_eval_KAN(xf, y_median)

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
    #     perc_overlap = (window_size-step)/window_size
    #     print("perc. of overlap: ", (window_size-step)*100/window_size)
    #     print("overlap duration (sec): ", window_size*perc_overlap/1000)
    # print("Number of windows / instances: ", xf.shape[0])
    # print("-rows: alg : KNN = 0; DT = 1; RF = 2; SVM = 3; GBM = 4; BDDAE = 5; DUMMY = 5")
    # print("columns:")
    # print("'val_cohen','val_uar', 'val_acc', 'val_gm',  'val_f1',  'aro_cohen','aro_uar', 'aro_acc', 'aro_gm',  'aro_f1'")
    # print("-------------------------------------------------------------")
    # print("  v_c   v_u   v_a   v_g   v_f1  a_c   a_u   a_a   a_g   a_f1")
    # # print("-columns: [(val_cohen, val_uar, val_acc,val_gm, val_f1, aro_cohen, aro_uar,aro_acc,aro_gm, aro_f1)]")
    # print(np.round(performance, decimals=3))
    # print("aro_feature_score: ", aro_feature_score)
    # print("val_feature_score: ", val_feature_score)

    #%%----------------------------------------------------------------

    # return performance, window_size, step, overlap, xf.shape[0], aro_feature_score, val_feature_score
    return xf, y_median, combined_df_median

if __name__ == "__main__":
    process_data('28', available_saved_data=True)

