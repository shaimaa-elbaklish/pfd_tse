###########################################################################
# Imports
###########################################################################
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from scipy.stats import spearmanr, pearsonr
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from TS_Laval_Maiti import compute_traffic_states
from TS_PFD import compute_PFD_traffic_states
from TS_PFD import estimate_traffic_states_GP
from TS_GroundTruth import compute_traffic_states_GT

import constants as cs
from constants import I24_TIMEZONE, I24_DATA_ROOT, I24_FREQUENCY
from constants import I24_MIN_POSITION, I24_TESTBED_LENGTH, I24_MIN_MILEMARKER, I24_MAX_SPEED
from utils import convert_to_unix
from utils import plot_vehicle_trajectories_TSD, visualize_raw_traffic_state_TSD

###########################################################################
# Parameters
###########################################################################
filename = "2022-11-22From0700To0730Processed.csv"
date = "2022-11-22"
starttime = convert_to_unix(date + " 07:00", I24_TIMEZONE)
endtime = convert_to_unix(date + " 07:30", I24_TIMEZONE)

dx, dt = 0.05*cs.MILE2FT, 10.0
seed = 1
all_penetration_rates = [0.05, 0.10, 0.25, 0.50, 0.75, 1.0]
all_lanes = [1,2,3,4]

###########################################################################
# Methods
###########################################################################
def obtain_ground_truth(trajectory_df):
    actual_traffic_states_dict = compute_traffic_states_GT(
        trajectory_df, dx, dt, starttime, endtime, lane_id=all_lanes, 
        min_position=I24_MIN_POSITION, testbed_length=I24_TESTBED_LENGTH
    )
    return actual_traffic_states_dict

def run_experiment(pen_rate, trajectory_df, actual_traffic_states_dict):
    # 1. Obtain PFD TS Estimates
    pfd_traffic_df = compute_PFD_traffic_states(trajectory_df, pen_rate=pen_rate, sampling_interval=1/I24_FREQUENCY, lane_id=all_lanes, random_seed=seed, 
                                                starttime=starttime, endtime=endtime, min_position=I24_MIN_POSITION, testbed_length=I24_TESTBED_LENGTH)
    estimated_traffic_states_dict = estimate_traffic_states_GP(
        pfd_traffic_df, dx, dt, starttime, endtime, lane_id=all_lanes, 
        min_position=I24_MIN_POSITION, testbed_length=I24_TESTBED_LENGTH
    )

    # 2. Compare
    comp_results_list = []
    for lane_id in all_lanes:
        Estimated_Density = np.round(estimated_traffic_states_dict[lane_id]["Density"], 3)
        Estimated_Flow = np.round(estimated_traffic_states_dict[lane_id]["Flow"], 3)
        Estimated_Speed = np.round(estimated_traffic_states_dict[lane_id]["Speed"], 3)
        Num_Observations = estimated_traffic_states_dict[lane_id]["Num_Observations"]

        Actual_Density = np.round(actual_traffic_states_dict[lane_id]["Density"], 3)
        Actual_Flow = np.round(actual_traffic_states_dict[lane_id]["Flow"], 3)
        Actual_Speed = np.round(actual_traffic_states_dict[lane_id]["Speed"], 3)
        Num_Vehicles = actual_traffic_states_dict[lane_id]["Num_Vehicles"]

        Percent_Observations = Num_Observations / Num_Vehicles

        mask = ~np.isnan(Actual_Density) & (Actual_Density > 1e-03) & (Actual_Speed < 80)
        k_mse = np.mean(np.square(Estimated_Density[mask] - Actual_Density[mask]))
        k_mae = np.mean(np.abs(Estimated_Density[mask] - Actual_Density[mask]))
        k_mare = np.mean(np.abs(Estimated_Density[mask] - Actual_Density[mask])/Actual_Density[mask])
        k_nrmse = np.sqrt(k_mse) / np.mean(Actual_Density[mask])

        mask = ~np.isnan(Actual_Flow) & (Actual_Flow > 1e-03) & (Actual_Speed < 80)
        q_mse = np.mean(np.square(Estimated_Flow[mask] - Actual_Flow[mask]))
        q_mae = np.mean(np.abs(Estimated_Flow[mask] - Actual_Flow[mask]))
        q_mare = np.mean(np.abs(Estimated_Flow[mask] - Actual_Flow[mask])/Actual_Flow[mask])
        q_nrmse = np.sqrt(q_mse) / np.mean(Actual_Flow[mask])

        mask = ~np.isnan(Actual_Speed) & (Actual_Speed > 1e-03) & (Actual_Speed < 80)
        v_mse = np.mean(np.square(Estimated_Speed[mask] - Actual_Speed[mask]))
        v_mae = np.mean(np.abs(Estimated_Speed[mask] - Actual_Speed[mask]))
        v_mare = np.mean(np.abs(Estimated_Speed[mask] - Actual_Speed[mask])/Actual_Speed[mask])
        v_nrmse = np.sqrt(v_mse) / np.mean(Actual_Speed[mask])

        ADiff_Density = abs(Estimated_Density - Actual_Density) / Actual_Density
        ADiff_Flow = abs(Estimated_Flow - Actual_Flow) / Actual_Flow
        ADiff_Speed = abs(Estimated_Speed - Actual_Speed) / Actual_Speed

        mask = (Actual_Density > 1e-03) & (Actual_Speed < 80) #& (Percent_Observations > 0)
        k_res = spearmanr(Num_Observations[mask].flatten(), ADiff_Density[mask].flatten(), alternative="less")
        #k_res = pearsonr(Percent_Observations[mask].flatten(), ADiff_Density[mask].flatten(), alternative="two-sided")

        mask = (Actual_Flow > 1e-03) & (Actual_Speed < 80) #& (Percent_Observations > 0)
        q_res = spearmanr(Num_Observations[mask].flatten(), ADiff_Flow[mask].flatten(), alternative="less")
        #q_res = pearsonr(Percent_Observations[mask].flatten(), ADiff_Flow[mask].flatten(), alternative="two-sided")

        mask = (Actual_Speed > 1e-03) & (Actual_Speed < 80) #& (Percent_Observations > 0)
        v_res = spearmanr(Num_Observations[mask].flatten(), ADiff_Speed[mask].flatten(), alternative="less")
        #v_res = pearsonr(Percent_Observations[mask].flatten(), ADiff_Speed[mask].flatten(), alternative="two-sided")
        
        comp_results_list.append(
            [
                lane_id, 
                np.sqrt(k_mse), k_nrmse, k_mae, k_mare, k_res.statistic, k_res.pvalue,
                np.sqrt(q_mse), q_nrmse, q_mae, q_mare, q_res.statistic, q_res.pvalue,
                np.sqrt(v_mse), v_nrmse, v_mae, v_mare, v_res.statistic, v_res.pvalue,
            ]
        )
    
    comp_results_df = pd.DataFrame(comp_results_list)
    comp_results_df.columns = [
        "Lane_ID",
        "Density_RMSE", "Density_NRMSE", "Density_MAE", "Density_MARE", "Density_SpearmanCoeff", "Density_SpearmanPval",
        "Flow_RMSE", "Flow_NRMSE", "Flow_MAE", "Flow_MARE", "Flow_SpearmanCoeff", "Flow_SpearmanPval",
        "Speed_RMSE", "Speed_NRMSE", "Speed_MAE", "Speed_MARE", "Speed_SpearmanCoeff", "Speed_SpearmanPval",
    ]
    comp_results_df["Penetration_Rate"] = pen_rate
    return comp_results_df


###########################################################################
# Main
###########################################################################
input_file_path = os.path.join(I24_DATA_ROOT, filename)
trajectory_df = pd.read_csv(input_file_path)
trajectory_df = trajectory_df.rename(
    columns ={
        "X_Position": "Position",
        "Leader_ID": "Preceeding",
        "Vehicle_Length": "v_Length",
        "Speed": "v_Vel"
    }
)
trajectory_df = trajectory_df.drop(columns=["Y_Position"])
# make Position monotonically increasing (westbound direction of I-24)
trajectory_df['Position'] = I24_TESTBED_LENGTH - (trajectory_df['Position'] - I24_MIN_MILEMARKER)

actual_traffic_states_dict = obtain_ground_truth(trajectory_df)
comp_results_df = None
for pen_rate in all_penetration_rates:
    df = run_experiment(pen_rate, trajectory_df, actual_traffic_states_dict)
    if comp_results_df is None:
        comp_results_df = df.copy()
    else:
        comp_results_df = pd.concat((comp_results_df, df))
comp_results_df = comp_results_df.reset_index().drop(columns="index")
comp_results_df.to_csv("./Comparison_Results_I24Motion.csv", index=False)
print(comp_results_df)

