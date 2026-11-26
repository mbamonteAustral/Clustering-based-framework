"""
main.py

main script. It calls processDEAPdata.py to apply different ALG to 
each participant data.
"""

#%% --------------------------------------------------
from processdataf import process_data
import numpy as np
from numpy import savetxt
from sys import platform
from openpyxl import load_workbook
import time
import tensorflow as tf
import os.path


with tf.device('/CPU:0'):

    # Record the start time
    start_time = time.time()

    participants = np.arange(1, 31)  # participant from 1 to 30

    # participants = np.arange(1, 3)  # participant from 1 to 30



    
    failed_participants = []

    for i in participants:
        
        print("    ")
        print("    ")
        print("-------------------------------")
        print("participant: ", i)
        # print(participant)
        print("-------------------------------")
        print("    ")
        print("    ")
        try:
            process_data(str(i))
            
        except Exception as exc:
            failed_participants.append(i)
            print(f"Failed to process participant {i}: {exc}")
        else:
            print(f"Saved pickle files for participant {i}.")

    output_dir = os.path.dirname(os.path.abspath(__file__))
    expected_files = [
        filename
        for i in participants
        for filename in (f"xf{i}.pkl", f"y_median{i}.pkl")
    ]
    missing_files = [
        filename
        for filename in expected_files
        if not os.path.isfile(os.path.join(output_dir, filename))
    ]

    if failed_participants or missing_files:
        raise RuntimeError(
            "Pickle generation incomplete. "
            f"Failed participants: {failed_participants}. "
            f"Missing files: {missing_files}."
        )

    print(f"Verified all {len(expected_files)} pickle files in {output_dir}.")
        

    

    # # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time  # in seconds

    print("Elapsed time:", elapsed_time/(60), "minutes")
    print("Elapsed time:", elapsed_time/(60*60), "hours")

    