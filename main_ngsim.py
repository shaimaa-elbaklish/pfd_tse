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
from constants import NGSIM_TIMEZONE, NGSIM_DATA_ROOT, NGSIM_FREQUENCY
from constants import NGSIM_MIN_POSITION, NGSIM_TESTBED_LENGTH, NGSIM_MAX_SPEED
from utils import convert_to_unix
from utils import plot_vehicle_trajectories_TSD, visualize_raw_traffic_state_TSD

###########################################################################
# Parameters
###########################################################################
location = "US-101-LosAngeles-CA"
date = "2005-06-15"
dx, dt = 100.0, 10.0
bin_width = 0.3/(1000*cs.METER2FT*cs.FT2MILE) # from veh/km to veh/mile

seed = 1234
penetration_rate = 0.25
selected_lane = 2

###########################################################################
# Read all vehicle trajectories files
###########################################################################
vehicle_traj_path = os.path.join(NGSIM_DATA_ROOT, location, "vehicle-trajectory-data")
durations = [dir for dir in os.listdir(vehicle_traj_path) if dir[0] != '.']
trajectory_df = None

duration = durations[0]

duration_start, duration_end = duration.split('-')
starttime = convert_to_unix(date + " " + duration_start, custom_format='%Y-%m-%d %I%M%p', timezone=NGSIM_TIMEZONE)
endtime = convert_to_unix(date + " " + duration_end, custom_format='%Y-%m-%d %I%M%p', timezone=NGSIM_TIMEZONE)
    
data_files = [fn for fn in os.listdir(os.path.join(vehicle_traj_path, duration)) if os.path.splitext(fn)[1] == '.csv' and 'trajectories' in fn]
if len(data_files) == 0:
    raise ValueError
# There should only be one CSV data file in each directory.
input_filename = data_files[0]
input_file_path = os.path.join(vehicle_traj_path, duration, input_filename)

trajectory_df = pd.read_csv(input_file_path)
trajectory_df["Position"] = trajectory_df["Local_Y"]
trajectory_df.loc[trajectory_df["Preceeding"] == 0, "Preceeding"] = pd.NA
trajectory_df.loc[trajectory_df["Following"] == 0, "Following"] = pd.NA
trajectory_df["Global_Time"] = trajectory_df["Global_Time"] / 1000.0
trajectory_df = trajectory_df.drop(columns=["Frame_ID", "Local_X", "Local_Y", "Global_X", "Global_Y", "Total_Frames"])

#plot_vehicle_trajectories_TSD(trajectory_df, lane_id=selected_lane, starttime=starttime, endtime=endtime, timezone=NGSIM_TIMEZONE, 
#                              max_speed=NGSIM_MAX_SPEED, min_position=NGSIM_MIN_POSITION, testbed_length=NGSIM_TESTBED_LENGTH)
#plt.show()

###########################################################################
# Obtain Ground Truth TS
###########################################################################
traffic_df = compute_traffic_states(trajectory_df, dx, dt, starttime, endtime, lane_id=selected_lane, 
                                    min_position=NGSIM_MIN_POSITION, testbed_length=NGSIM_TESTBED_LENGTH)
print(traffic_df.head())

#visualize_raw_traffic_state_TSD(traffic_df, "Speed", starttime, endtime, lane_id=selected_lane, timezone=NGSIM_TIMEZONE, 
#                                min_position=NGSIM_MIN_POSITION, testbed_length=NGSIM_TESTBED_LENGTH, max_speed=NGSIM_MAX_SPEED)
#plt.show()

###########################################################################
# Obtain TS Estimates with PFD according to penetration rate
###########################################################################
pfd_traffic_df = compute_PFD_traffic_states(trajectory_df, pen_rate=penetration_rate, sampling_interval=1/NGSIM_FREQUENCY, lane_id=selected_lane, random_seed=seed, 
                                            starttime=starttime, endtime=endtime, min_position=NGSIM_MIN_POSITION, testbed_length=NGSIM_TESTBED_LENGTH)
print(pfd_traffic_df.head())

#visualize_raw_traffic_state_TSD(pfd_traffic_df, "Speed", starttime, endtime, lane_id=selected_lane, timezone=NGSIM_TIMEZONE, 
#                                min_position=NGSIM_MIN_POSITION, testbed_length=NGSIM_TESTBED_LENGTH, max_speed=NGSIM_MAX_SPEED)
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
    min_position=NGSIM_MIN_POSITION, testbed_length=NGSIM_TESTBED_LENGTH
)


###########################################################################
# Compare TS Estimates with respect to Ground Truth 
# Ground Truth is obtained by Edie's Generalized Definitions
###########################################################################
Density_Mat, Flow_Mat, Speed_Mat, Num_Vehicles_Mat = compute_traffic_states_GT(
    trajectory_df, dx, dt, starttime, endtime, lane_id=selected_lane, 
    min_position=NGSIM_MIN_POSITION, testbed_length=NGSIM_TESTBED_LENGTH
)

mask = (Speed_Mat >= 80.0)
Density_Mat[mask] = np.nan
Estimated_Density[mask] = np.nan
Flow_Mat[mask] = np.nan
Estimated_Flow[mask] = np.nan
Speed_Mat[mask] = np.nan
Estimated_Speed[mask] = np.nan

mask = np.isnan(Speed_Mat)
k_max = min(120, max(np.amax(Density_Mat[mask]), np.amax(Estimated_Density[mask]))) # cs.PFD_MAX_DENSITY
q_max = min(2000, max(np.amax(Flow_Mat[mask]), np.amax(Estimated_Flow[mask])))
v_max = min(65, max(np.amax(Speed_Mat[mask]), np.amax(Estimated_Speed[mask])))

n_x_cols, n_t_rows = Estimated_Density.shape
jet = plt.cm.jet
colors = [jet(x) for x in np.linspace(1, 0.5, 256)]
cmap = LinearSegmentedColormap.from_list('GreenToRed', colors, N=256)

fs1, fs2 = 12, 14
space_ticks = np.linspace(NGSIM_MIN_POSITION, (NGSIM_MIN_POSITION+NGSIM_TESTBED_LENGTH), 5)
space_ticks = space_ticks[::-1]
start_time = datetime.datetime.strptime(datetime.datetime.fromtimestamp(starttime, tz=NGSIM_TIMEZONE).strftime("%H:%M"), "%H:%M")
time_ticks = list(range(0, endtime-starttime + 1, 300))
time_ticks = [(start_time + datetime.timedelta(seconds=tick)).strftime("%H:%M") for tick in time_ticks]

fig = plt.figure(num="Actual Density", figsize=(6,4))
ax = plt.gca()
ax.matshow(Density_Mat, cmap=cmap, vmin=0, vmax=k_max, aspect='auto')
ax.set_yticks(np.linspace(0, n_x_cols, len(space_ticks)))
ax.set_yticklabels(np.round(space_ticks, 2), fontsize=fs1)
ax.set_ylabel('Position $\it{[ft]}$', fontsize=fs2)
ax.set_xticks(np.linspace(0, n_t_rows, len(time_ticks)))
ax.set_xticklabels(time_ticks, rotation=45, fontsize=fs1)
ax.xaxis.set_ticks_position('bottom')
fig.tight_layout()

fig = plt.figure(num="Estimated Density", figsize=(6,4))
ax = plt.gca()
sc = ax.matshow(Estimated_Density, cmap=cmap, vmin=0, vmax=k_max, aspect='auto')
ax.set_yticks(np.linspace(0, n_x_cols, len(space_ticks)))
ax.set_yticklabels(np.round(space_ticks, 2), fontsize=fs1)
ax.set_ylabel('Position $\it{[ft]}$', fontsize=fs2)
ax.set_xticks(np.linspace(0, n_t_rows, len(time_ticks)))
ax.set_xticklabels(time_ticks, rotation=45, fontsize=fs1)
ax.xaxis.set_ticks_position('bottom')
plt.colorbar(sc, ax=ax, pad=0.04).set_label('Density $\it{[veh/mile]}$', rotation=90, labelpad=20, fontsize=fs2) #fraction=0.015
fig.tight_layout()

fig = plt.figure(num="Actual Flow", figsize=(6,4))
ax = plt.gca()
ax.matshow(Flow_Mat, cmap=cmap, vmin=0, vmax=q_max, aspect='auto')
ax.set_yticks(np.linspace(0, n_x_cols, len(space_ticks)))
ax.set_yticklabels(np.round(space_ticks, 2), fontsize=fs1)
ax.set_ylabel('Position $\it{[ft]}$', fontsize=fs2)
ax.set_xticks(np.linspace(0, n_t_rows, len(time_ticks)))
ax.set_xticklabels(time_ticks, rotation=45, fontsize=fs1)
ax.xaxis.set_ticks_position('bottom')
fig.tight_layout()

fig = plt.figure(num="Estimated Flow", figsize=(6,4))
ax = plt.gca()
sc = ax.matshow(Estimated_Flow, cmap=cmap, vmin=0, vmax=q_max, aspect='auto')
ax.set_yticks(np.linspace(0, n_x_cols, len(space_ticks)))
ax.set_yticklabels(np.round(space_ticks, 2), fontsize=fs1)
ax.set_ylabel('Position $\it{[ft]}$', fontsize=fs2)
ax.set_xticks(np.linspace(0, n_t_rows, len(time_ticks)))
ax.set_xticklabels(time_ticks, rotation=45, fontsize=fs1)
ax.xaxis.set_ticks_position('bottom')
plt.colorbar(sc, ax=ax, pad=0.04).set_label('Flow $\it{[veh/h]}$', rotation=90, labelpad=20, fontsize=fs2)
fig.tight_layout()

fig = plt.figure(num="Actual Speed", figsize=(6,4))
ax = plt.gca()
ax.matshow(Speed_Mat, cmap=cmap, vmin=0, vmax=v_max, aspect='auto')
ax.set_yticks(np.linspace(0, n_x_cols, len(space_ticks)))
ax.set_yticklabels(np.round(space_ticks, 2), fontsize=fs1)
ax.set_ylabel('Position $\it{[ft]}$', fontsize=fs2)
ax.set_xticks(np.linspace(0, n_t_rows, len(time_ticks)))
ax.set_xticklabels(time_ticks, rotation=45, fontsize=fs1)
ax.xaxis.set_ticks_position('bottom')
fig.tight_layout()

fig = plt.figure(num="Estimated Speed", figsize=(6,4))
ax = plt.gca()
sc = ax.matshow(Estimated_Speed, cmap=cmap, vmin=0, vmax=v_max, aspect='auto')
ax.set_yticks(np.linspace(0, n_x_cols, len(space_ticks)))
ax.set_yticklabels(np.round(space_ticks, 2), fontsize=fs1)
ax.set_ylabel('Position $\it{[ft]}$', fontsize=fs2)
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
ax.set_ylabel('Position $\it{[ft]}$', fontsize=fs2)
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
ax.set_ylabel('Position $\it{[ft]}$', fontsize=fs2)
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
ax.set_ylabel('Position $\it{[ft]}$', fontsize=fs2)
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
ax.set_ylabel('Position $\it{[ft]}$', fontsize=fs2)
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

#mask = ~np.isnan(Num_Vehicles_Mat) & (Num_Vehicles_Mat > 0)
#print(np.median(Num_Vehicles_Mat[mask]), np.mean(Num_Vehicles_Mat[mask]))