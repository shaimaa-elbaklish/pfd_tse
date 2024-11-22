###########################################################################
# Imports
###########################################################################
import os
import sys
import datetime
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from TS_Laval_Maiti import compute_traffic_states
from TS_PFD import compute_PFD_traffic_states
from TS_PFD import estimate_traffic_states_GP
from TS_GroundTruth import compute_traffic_states_GT
from FD_Aggregation import aggregate_FD

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
bin_width = 0.3/(1000*cs.METER2FT*cs.FT2MILE) # from veh/km to veh/mile
penetration_rate = 0.05
seed = 1

selected_lane = 2

###########################################################################
# Read all vehicle trajectories files
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
print(trajectory_df.head())

#plot_vehicle_trajectories_TSD(trajectory_df, lane_id=selected_lane, starttime=starttime, endtime=endtime, timezone=I24_TIMEZONE, 
#                              max_speed=I24_MAX_SPEED, min_position=I24_MIN_POSITION, testbed_length=I24_TESTBED_LENGTH)
#plt.show()

###########################################################################
# Obtain Ground Truth TS
###########################################################################
#traffic_df = compute_traffic_states(trajectory_df, dx, dt, starttime, endtime, lane_id=selected_lane, 
#                                    min_position=I24_MIN_POSITION, testbed_length=I24_TESTBED_LENGTH)
#print(traffic_df.head())

#visualize_raw_traffic_state_TSD(traffic_df, "Speed", starttime, endtime, lane_id=selected_lane, timezone=I24_TIMEZONE, 
#                                min_position=I24_MIN_POSITION, testbed_length=I24_TESTBED_LENGTH, max_speed=I24_MAX_SPEED)
#plt.show()


###########################################################################
# Obtain TS Estimates with PFD according to penetration rate
###########################################################################
pfd_traffic_df = compute_PFD_traffic_states(trajectory_df, pen_rate=penetration_rate, sampling_interval=1/I24_FREQUENCY, lane_id=selected_lane, random_seed=seed, 
                                            starttime=starttime, endtime=endtime, min_position=I24_MIN_POSITION, testbed_length=I24_TESTBED_LENGTH)
print(pfd_traffic_df.head())

#visualize_raw_traffic_state_TSD(pfd_traffic_df, "Speed", starttime, endtime, lane_id=selected_lane, timezone=I24_TIMEZONE, 
#                                min_position=I24_MIN_POSITION, testbed_length=I24_TESTBED_LENGTH, max_speed=I24_MAX_SPEED)
#plt.show()

###########################################################################
# Obtain Fundamental Diagrams
###########################################################################
#fig = aggregate_FD(traffic_df, lane_id=selected_lane, bin_width=bin_width, FD_mode="Exponential")
#fig.suptitle("Laval-Maiti")
#plt.show()

fig = aggregate_FD(pfd_traffic_df, lane_id=selected_lane, bin_width=bin_width, FD_mode="Exponential")
#fig.suptitle("PFD")

plt.show()

###########################################################################
# Obtain TS Estimates at unobserved time-space locations via GP
###########################################################################
Estimated_Density, Estimated_Flow, Estimated_Speed, Num_Observations_Mat = estimate_traffic_states_GP(
    pfd_traffic_df, dx, dt, starttime, endtime, lane_id=selected_lane, 
    min_position=I24_MIN_POSITION, testbed_length=I24_TESTBED_LENGTH
)


###########################################################################
# Compare TS Estimates with respect to Ground Truth 
# Ground Truth is obtained by Edie's Generalized Definitions
###########################################################################
Density_Mat, Flow_Mat, Speed_Mat, Num_Vehicles_Mat = compute_traffic_states_GT(
    trajectory_df, dx, dt, starttime, endtime, lane_id=selected_lane, 
    min_position=I24_MIN_POSITION, testbed_length=I24_TESTBED_LENGTH
)

k_max = 150 # cs.PFD_MAX_DENSITY
q_max = 2000
v_max = 65

n_x_cols, n_t_rows = Estimated_Density.shape
jet = plt.cm.jet
colors = [jet(x) for x in np.linspace(1, 0.5, 256)]
cmap = LinearSegmentedColormap.from_list('GreenToRed', colors, N=256)

fs1, fs2 = 12, 14
space_ticks = np.linspace(I24_MIN_MILEMARKER, (I24_MIN_MILEMARKER+I24_TESTBED_LENGTH), 5)
space_ticks = space_ticks*cs.FT2MILE
start_time = datetime.datetime.strptime(datetime.datetime.fromtimestamp(starttime, tz=I24_TIMEZONE).strftime("%H:%M"), "%H:%M")
time_ticks = list(range(0, endtime-starttime + 1, 300))
time_ticks = [(start_time + datetime.timedelta(seconds=tick)).strftime("%H:%M") for tick in time_ticks]

mask = (Speed_Mat >= 80.0)
Density_Mat[mask] = np.nan
Estimated_Density[mask] = np.nan

fig = plt.figure(num="Actual Density", figsize=(6,4))
ax = plt.gca()
ax.matshow(Density_Mat, cmap=cmap, vmin=0, vmax=k_max, aspect='auto')
ax.set_yticks(np.linspace(0, n_x_cols, len(space_ticks)))
ax.set_yticklabels(np.round(space_ticks, 2), fontsize=fs1)
ax.set_ylabel('Position $\it{[mile]}$', fontsize=fs2)
ax.set_xticks(np.linspace(0, n_t_rows, len(time_ticks)))
ax.set_xticklabels(time_ticks, rotation=45, fontsize=fs1)
ax.xaxis.set_ticks_position('bottom')
fig.tight_layout()

fig = plt.figure(num="Estimated Density", figsize=(6,4))
ax = plt.gca()
sc = ax.matshow(Estimated_Density, cmap=cmap, vmin=0, vmax=k_max, aspect='auto')
ax.set_yticks(np.linspace(0, n_x_cols, len(space_ticks)))
ax.set_yticklabels(np.round(space_ticks, 2), fontsize=fs1)
ax.set_ylabel('Position $\it{[mile]}$', fontsize=fs2)
ax.set_xticks(np.linspace(0, n_t_rows, len(time_ticks)))
ax.set_xticklabels(time_ticks, rotation=45, fontsize=fs1)
ax.xaxis.set_ticks_position('bottom')
plt.colorbar(sc, ax=ax, pad=0.04).set_label('Density $\it{[veh/mile]}$', rotation=90, labelpad=20, fontsize=fs2) #fraction=0.015
fig.tight_layout()

mask = (Speed_Mat >= 80.0)
Flow_Mat[mask] = np.nan
Estimated_Flow[mask] = np.nan

fig = plt.figure(num="Actual Flow", figsize=(6,4))
ax = plt.gca()
ax.matshow(Flow_Mat, cmap=cmap, vmin=0, vmax=q_max, aspect='auto')
ax.set_yticks(np.linspace(0, n_x_cols, len(space_ticks)))
ax.set_yticklabels(np.round(space_ticks, 2), fontsize=fs1)
ax.set_ylabel('Position $\it{[mile]}$', fontsize=fs2)
ax.set_xticks(np.linspace(0, n_t_rows, len(time_ticks)))
ax.set_xticklabels(time_ticks, rotation=45, fontsize=fs1)
ax.xaxis.set_ticks_position('bottom')
fig.tight_layout()

fig = plt.figure(num="Estimated Flow", figsize=(6,4))
ax = plt.gca()
sc = ax.matshow(Estimated_Flow, cmap=cmap, vmin=0, vmax=q_max, aspect='auto')
ax.set_yticks(np.linspace(0, n_x_cols, len(space_ticks)))
ax.set_yticklabels(np.round(space_ticks, 2), fontsize=fs1)
ax.set_ylabel('Position $\it{[mile]}$', fontsize=fs2)
ax.set_xticks(np.linspace(0, n_t_rows, len(time_ticks)))
ax.set_xticklabels(time_ticks, rotation=45, fontsize=fs1)
ax.xaxis.set_ticks_position('bottom')
plt.colorbar(sc, ax=ax, pad=0.04).set_label('Flow $\it{[veh/h]}$', rotation=90, labelpad=20, fontsize=fs2)
fig.tight_layout()

mask = (Speed_Mat >= 80.0)
Speed_Mat[mask] = np.nan
Estimated_Speed[mask] = np.nan

fig = plt.figure(num="Actual Speed", figsize=(6,4))
ax = plt.gca()
ax.matshow(Speed_Mat, cmap=cmap, vmin=0, vmax=v_max, aspect='auto')
ax.set_yticks(np.linspace(0, n_x_cols, len(space_ticks)))
ax.set_yticklabels(np.round(space_ticks, 2), fontsize=fs1)
ax.set_ylabel('Position $\it{[mile]}$', fontsize=fs2)
ax.set_xticks(np.linspace(0, n_t_rows, len(time_ticks)))
ax.set_xticklabels(time_ticks, rotation=45, fontsize=fs1)
ax.xaxis.set_ticks_position('bottom')
fig.tight_layout()

fig = plt.figure(num="Estimated Speed", figsize=(6,4))
ax = plt.gca()
sc = ax.matshow(Estimated_Speed, cmap=cmap, vmin=0, vmax=v_max, aspect='auto')
ax.set_yticks(np.linspace(0, n_x_cols, len(space_ticks)))
ax.set_yticklabels(np.round(space_ticks, 2), fontsize=fs1)
ax.set_ylabel('Position $\it{[mile]}$', fontsize=fs2)
ax.set_xticks(np.linspace(0, n_t_rows, len(time_ticks)))
ax.set_xticklabels(time_ticks, rotation=45, fontsize=fs1)
ax.xaxis.set_ticks_position('bottom')
plt.colorbar(sc, ax=ax, pad=0.04).set_label('Speed $\it{[mph]}$', rotation=90, labelpad=20, fontsize=fs2)
fig.tight_layout()


#plt.show()
n_max = np.amax(Num_Observations_Mat)
mask = np.isnan(Speed_Mat)
Num_Observations_Mat[mask] = np.nan

fig = plt.figure(num="Number of Observations", figsize=(6,4))
ax = plt.gca()
sc = ax.matshow(Num_Observations_Mat, cmap=cmap, vmin=0, vmax=n_max, aspect='auto') #np.amax(Num_Observations_Mat)
ax.set_yticks(np.linspace(0, n_x_cols, len(space_ticks)))
ax.set_yticklabels(np.round(space_ticks, 2), fontsize=fs1)
ax.set_ylabel('Position $\it{[mile]}$', fontsize=fs2)
ax.set_xticks(np.linspace(0, n_t_rows, len(time_ticks)))
ax.set_xticklabels(time_ticks, rotation=45, fontsize=fs1)
ax.xaxis.set_ticks_position('bottom')
plt.colorbar(sc, ax=ax, pad=0.04).set_label('Number of Observations $\it{[veh]}$', rotation=90, labelpad=20, fontsize=fs2)
fig.tight_layout()


ADiff_Density = abs(Estimated_Density - Density_Mat)*100.0 / Density_Mat
ADiff_Density[mask] = np.nan
fig = plt.figure(num="Relative Density Error", figsize=(6,4))
ax = plt.gca()
sc = ax.matshow(ADiff_Density, cmap=cmap, vmin=0, vmax=100, aspect='auto')
ax.set_yticks(np.linspace(0, n_x_cols, len(space_ticks)))
ax.set_yticklabels(np.round(space_ticks, 2), fontsize=fs1)
ax.set_ylabel('Position $\it{[mile]}$', fontsize=fs2)
ax.set_xticks(np.linspace(0, n_t_rows, len(time_ticks)))
ax.set_xticklabels(time_ticks, rotation=45, fontsize=fs1)
ax.xaxis.set_ticks_position('bottom')
plt.colorbar(sc, ax=ax, pad=0.04).set_label('Density Relative Error $\it{[\%]}$', rotation=90, labelpad=20, fontsize=fs2)
fig.tight_layout()

ADiff_Flow = abs(Estimated_Flow - Flow_Mat)*100.0 / Flow_Mat
ADiff_Flow[mask] = np.nan
fig = plt.figure(num="Relative Flow Error", figsize=(6,4))
ax = plt.gca()
sc = ax.matshow(ADiff_Flow, cmap=cmap, vmin=0, vmax=100, aspect='auto')
ax.set_yticks(np.linspace(0, n_x_cols, len(space_ticks)))
ax.set_yticklabels(np.round(space_ticks, 2), fontsize=fs1)
ax.set_ylabel('Position $\it{[mile]}$', fontsize=fs2)
ax.set_xticks(np.linspace(0, n_t_rows, len(time_ticks)))
ax.set_xticklabels(time_ticks, rotation=45, fontsize=fs1)
ax.xaxis.set_ticks_position('bottom')
plt.colorbar(sc, ax=ax, pad=0.04).set_label('Flow Relative Error $\it{[\%]}$', rotation=90, labelpad=20, fontsize=fs2)
fig.tight_layout()

ADiff_Speed = abs(Estimated_Speed - Speed_Mat)*100.0 / Speed_Mat
ADiff_Speed[mask] = np.nan
fig = plt.figure(num="Relative Speed Error", figsize=(6,4))
ax = plt.gca()
sc = ax.matshow(ADiff_Speed, cmap=cmap, vmin=0, vmax=100, aspect='auto')
ax.set_yticks(np.linspace(0, n_x_cols, len(space_ticks)))
ax.set_yticklabels(np.round(space_ticks, 2), fontsize=fs1)
ax.set_ylabel('Position $\it{[mile]}$', fontsize=fs2)
ax.set_xticks(np.linspace(0, n_t_rows, len(time_ticks)))
ax.set_xticklabels(time_ticks, rotation=45, fontsize=fs1)
ax.xaxis.set_ticks_position('bottom')
plt.colorbar(sc, ax=ax, pad=0.04).set_label('Speed Relative Error $\it{[\%]}$', rotation=90, labelpad=20, fontsize=fs2)
fig.tight_layout()

plt.show()

print(np.median(ADiff_Density[~mask]), np.mean(ADiff_Density[~mask]))
print(np.median(ADiff_Flow[~mask]), np.mean(ADiff_Flow[~mask]))
print(np.median(ADiff_Speed[~mask]), np.mean(ADiff_Speed[~mask]))

mask = (~mask) & (Num_Observations_Mat > 0)
print(np.median(Num_Observations_Mat[mask]), np.mean(Num_Observations_Mat[mask]))