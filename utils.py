###########################################################################
# IMPORTS
###########################################################################
import sys
import os
import warnings
warnings.filterwarnings("ignore")
import datetime
import pytz

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import patches
from matplotlib.collections import cm

from typing import Any, Tuple 

import constants as cs

###########################################################################
# METHODS
###########################################################################

def convert_to_unix(date_string, timezone, is_24hour_format = True, custom_format = None):
    """
    Convert a date string in the format 'YYYY-MM-DD HH:MM' to a Unix timestamp in given timezone.
    """
    # Parse the date string to a datetime object
    if custom_format is None:
        if is_24hour_format:
            date_obj = datetime.datetime.strptime(date_string, '%Y-%m-%d %H:%M')
        else:
            date_obj = datetime.datetime.strptime(date_string, '%Y-%m-%d %I:%M %p')
    else:
        date_obj = datetime.datetime.strptime(date_string, custom_format)
    
    date_obj_localized = timezone.localize(date_obj)
    # Convert to Unix timestamp
    unix_timestamp = int(date_obj_localized.timestamp())
    return unix_timestamp


def plot_vehicle_trajectories_TSD(trajectory_df: pd.DataFrame, lane_id: int, starttime: float, endtime: float, timezone: Any, min_position: float, testbed_length: float, 
                                  max_speed: float = None, fig_width: float = 8, fig_height: float = 8, minor_xtick: float = 300):
    fs_1, fs_2 = 14, 12
    subdf = trajectory_df[(trajectory_df["Lane_ID"] == lane_id) & (trajectory_df["Global_Time"] >= starttime) & (trajectory_df["Global_Time"] <= endtime)].copy()
    subdf[["Next_Global_Time", "Next_Position"]] = subdf.groupby("Vehicle_ID")[["Global_Time", "Position"]].shift(-1)
    subdf = subdf[subdf["Next_Global_Time"].notna()]
    line_segs = subdf[['Global_Time', 'Position', 'Next_Global_Time', 'Next_Position']].values.reshape((len(subdf), 2, 2))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    jet = plt.cm.jet
    colors = [jet(x) for x in np.linspace(1, 0.5, 256)]
    cmap = LinearSegmentedColormap.from_list('GreenToRed', colors, N=256)
    if max_speed is None:
        max_speed = subdf['v_Vel'].max()
    norm = plt.Normalize(vmin=0, vmax=max_speed)
    lc = LineCollection(line_segs, norm=norm, cmap=cmap)
    lc.set_array(subdf['v_Vel'].values)
    lc.set_linewidth(1)
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_ylabel('Position (ft)', fontsize=fs_1)
    ticks = list(range(starttime, endtime + 1, minor_xtick))
    xlabels = [datetime.datetime.fromtimestamp(tick, tz=timezone).strftime("%H:%M") for tick in ticks]
    plt.xticks(ticks, labels=xlabels, rotation=45, fontsize=fs_2)
    plt.yticks(fontsize=fs_2)
    plt.xlim(starttime, endtime)
    plt.ylim(min_position, min_position+testbed_length)
    plt.grid(which='both', linewidth=1, linestyle='--')
    cbar = plt.colorbar(lc, ax=ax, norm=norm)
    cbar.set_label('Velocity (ft/s)', fontsize=fs_1)
    cbar.ax.tick_params(labelsize=fs_2)
    fig.tight_layout()
    #plt.show()


def visualize_raw_traffic_state_TSD(traffic_df: pd.DataFrame, traffic_state: str, starttime: float, endtime: float, lane_id: int, timezone: Any, 
                                    min_position: float, testbed_length: float, max_speed: float, flip_yaxis: bool = False, yaxis_mile: bool = False, 
                                    fig_width: float = 8, fig_height: float = 8, minor_xtick: float = 300):
    cbar_label = {
        "Speed": "Speed (mph)", 
        "Flow": "Flow (veh/hour)",
        "Density": "Density (veh/mile)"
    }
    if traffic_state not in traffic_df.columns and traffic_state not in cbar.keys():
        raise KeyError
    
    fs_1, fs_2 = 14, 12

    df = traffic_df[traffic_df["Lane_ID"] == lane_id].copy()

    jet = plt.cm.jet
    colors = [jet(x) for x in np.linspace(1, 0.5, 256)]
    cmap = LinearSegmentedColormap.from_list('GreenToRed', colors, N=256)
    norm = plt.Normalize(vmin=0, vmax=max_speed)
    _, ax = plt.subplots(figsize=(fig_width, fig_height))

    for _, row in df.iterrows():
        x = [row['A_t00'], row['A_t01'], row['A_t11'], row['A_t10'], row['A_t00']]
        y = [row['A_x00'], row['A_x01'], row['A_x11'], row['A_x10'], row['A_x00']]
        ax.add_patch(
            patches.Polygon(xy=list(zip(x, y)), fill=True, color=cmap(norm(row[traffic_state])))
        )
    
    # Customize the axes ticks and labels for milemarkers on y-axis and timestamp on x-axis.
    start_time = datetime.datetime.strptime(datetime.datetime.fromtimestamp(starttime, tz=timezone).strftime("%H:%M"), "%H:%M")
    ticks = list(range(0, endtime-starttime + 1, minor_xtick))
    xlabels = [(start_time + datetime.timedelta(seconds=tick)).strftime("%H:%M") for tick in ticks]
    plt.xticks(ticks, labels=xlabels, rotation=45, fontsize=fs_2)
    plt.xlim(0, (endtime-starttime))
    plt.ylim(min_position, min_position+testbed_length)
    if yaxis_mile:
        yticks, _ = plt.yticks()
        ylabels = [tick*cs.FT2MILE for tick in yticks]
        plt.yticks(yticks, labels=ylabels, fontsize=fs_2)
        plt.ylabel('Position (Mile)', fontsize=fs_1)
    else:
        plt.yticks(fontsize=fs_2)
        plt.ylabel('Position (Feet)', fontsize=fs_1)
    if flip_yaxis:
        plt.gca().invert_yaxis()
    # Add a grid and colorbar.
    plt.grid(which='both', linewidth=1, linestyle='--')
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
    cbar.set_label(cbar_label[traffic_state], rotation=90, labelpad=20, fontsize=fs_1)
    cbar.ax.tick_params(labelsize=fs_2)
    plt.tight_layout()
