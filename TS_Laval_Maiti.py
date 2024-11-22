###########################################################################
# IMPORTS
###########################################################################
import os
import sys
import pytz

import numpy as np
import pandas as pd

from typing import List, Optional

import constants as cs

###########################################################################
# METHODS
###########################################################################
def _compute_traffic_states_single_lane(trajectory_df: pd.DataFrame, dx: float, dt: float, 
                                        starttime: float, endtime: float, lane_id: int, 
                                        min_position: float, testbed_length: float):
    c_cong_ftps = cs.BACKWARDS_WAVE_SPEED * cs.MILE2FT / cs.HR2SEC

    df = trajectory_df[trajectory_df["Lane_ID"] == lane_id].copy()
    df['Time'] = df['Global_Time'] - starttime
    df = df[df.Time >= 0]
    df['Transformed_Time'] = df['Time'] - df['Position'] / c_cong_ftps
    df['Transformed_Time_Index'] = (df['Transformed_Time']-df.Transformed_Time.min()) // dt
    df['Space_Index'] = (df['Position'] - min_position) // dx
    df['Transformed_Time_Index'] = df['Transformed_Time_Index'].astype(int)
    df['Space_Index'] = df['Space_Index'].astype(int)

    df[['Next_Position', 'Next_Transformed_Time']] = df.groupby('Vehicle_ID')[['Position', 'Transformed_Time']].shift(-1)
    df = df.dropna()
    df['Travel_Distance'] = df['Next_Position'] - df['Position']
    df['Travel_Transformed_Time'] = df['Next_Transformed_Time'] - df['Transformed_Time']
    
    grouped = df.groupby(by=['Space_Index', 'Transformed_Time_Index']).agg(
        Num_Vehicles=pd.NamedAgg(column="Vehicle_ID", aggfunc=pd.Series.nunique),
        TTD=pd.NamedAgg(column="Travel_Distance", aggfunc="sum"),
        TTT=pd.NamedAgg(column="Travel_Transformed_Time", aggfunc="sum"),
    )
    grouped = grouped.reset_index()
    grouped['Density_Transformed'] = grouped['TTT'] / (dx*cs.FT2MILE*dt) # veh / mile
    grouped['Flow'] = grouped['TTD'] / (dx*dt*cs.SEC2HR) # veh / hour
    grouped['Density'] = grouped['Density_Transformed'] + grouped['Flow'] / cs.BACKWARDS_WAVE_SPEED
    grouped['Speed'] = grouped['Flow'] / grouped['Density']

    grouped["Position"] = grouped["Space_Index"] * dx + min_position
    grouped["Time_Transformed"] = grouped["Transformed_Time_Index"] * dt + df.Transformed_Time.min()
    grouped["Time"] = grouped["Time_Transformed"] + grouped["Position"] / c_cong_ftps
    grouped["Global_Time"] = grouped["Time"] + starttime

    # from lower left corner of parallelogram and then clockwise
    grouped["A_x00"] = grouped["Position"]
    grouped["A_x01"] = grouped["Position"] + dx
    grouped["A_x11"] = grouped["Position"] + dx
    grouped["A_x10"] = grouped["Position"]
    grouped["A_t00"] = grouped["Time_Transformed"] + grouped["Position"] / c_cong_ftps
    grouped["A_t01"] = grouped["Time_Transformed"] + (grouped["Position"]+dx) / c_cong_ftps
    grouped["A_t11"] = (grouped["Time_Transformed"]+dt) + (grouped["Position"]+dx) / c_cong_ftps
    grouped["A_t10"] = (grouped["Time_Transformed"]+dt) + grouped["Position"] / c_cong_ftps

    grouped["center_T_Transformed"] = grouped["Time_Transformed"] + 0.5*dt
    grouped["center_X"] = grouped["Position"] + 0.5*dx
    grouped["center_T"] = grouped["center_T_Transformed"] + grouped["center_X"] / c_cong_ftps

    grouped = grouped.drop(columns=["Space_Index", "Transformed_Time_Index", "TTT", "TTD"])
    grouped["Lane_ID"] = lane_id
    grouped = grouped.dropna()
    return grouped
    
    """
    n_t_rows, n_x_cols = int((endtime-starttime)/dt), int((testbed_length-min_position)/dx)
    TTD_Matrix = np.zeros(shape=(n_t_rows, n_x_cols))
    TTT_Matrix = np.zeros(shape=(n_t_rows, n_x_cols))
    Num_Vehicles_Matrix = np.zeros(shape=(n_t_rows, n_x_cols))
    grouped = df.groupby(['Transformed_Time_Index', 'Space_Index', 'Vehicle_ID'])
    for (time, space, _), group_df in grouped:
        if time>=0 and time<TTD_Matrix.shape[0] and space>=0 and space<TTD_Matrix.shape[1]:
            TTD_Matrix[int(time), int(space)] += (group_df.Position.max() - group_df.Position.min())
            TTT_Matrix[int(time), int(space)] += (group_df.Transformed_Time.max() - group_df.Transformed_Time.min())
            Num_Vehicles_Matrix[int(time), int(space)] += 1
    
    list_traffic_states = []
    for t_idx in range(n_t_rows):
        for x_idx in range(n_x_cols):
            x = dx * x_idx + min_position
            t_transformed = dt * t_idx + df.Transformed_Time.min()
            t = t_transformed + x / c_cong_ftps

            center_T_Transformed = t_transformed + 0.5*dt
            center_X = x + 0.5*dx
            center_T = center_T_Transformed + center_X / c_cong_ftps

            x0, x1 = x, x + dx
            t_00 = t_transformed + x0/c_cong_ftps
            t_01 = t_transformed + x1/c_cong_ftps
            t_10 = t_transformed+dt + x0/c_cong_ftps
            t_11 = t_transformed+dt + x1/c_cong_ftps

            timestamp = t + starttime
            q = TTD_Matrix[t_idx, x_idx] / (dx*dt*cs.SEC2HR) # veh / hour
            khat = TTT_Matrix[t_idx, x_idx] / (dx*cs.FT2MILE*dt) # veh / mile
            k = khat + q / cs.BACKWARDS_WAVE_SPEED
            v = q / k # mile / hour

            list_traffic_states.append(
                (timestamp, t, t_transformed, x, 
                 k, q, v, Num_Vehicles_Matrix[t_idx, x_idx], 
                 center_T, center_T_Transformed, center_X,
                 t_00, t_01, t_11, t_10, x0, x1, x1, x0) 
            )
    df_traffic_states = pd.DataFrame(list_traffic_states)
    df_traffic_states.columns = [
        'Global_Time', 'Time', 'Time_Transformed', 'Position', 
        'Density', 'Flow', 'Speed', 'Num_Vehicles',
        'center_T', 'center_T_Transformed', 'center_X',
        'A_t00', 'A_t01', 'A_t11', 'A_t10', 'A_x00', 'A_x01', 'A_x11', 'A_x10', # from lower left corner of parallelogram and then clockwise
    ]
    df_traffic_states["Lane_ID"] = lane_id
    df_traffic_states = df_traffic_states.dropna()
    return df_traffic_states
    """
    


def compute_traffic_states(trajectory_df: pd.DataFrame, dx: float, dt: float, 
                           starttime: float, endtime: float, 
                           min_position: float, testbed_length: float,
                           lane_id: Optional[List[int] | int] = None):
    if lane_id is not None and isinstance(lane_id, int):
        return _compute_traffic_states_single_lane(trajectory_df, dx, dt, starttime, endtime, lane_id, min_position, testbed_length)
    
    if lane_id is None:
        unique_lane_ids = trajectory_df["Lane_ID"].unique()
    elif isinstance(lane_id, list):
        unique_lane_ids = lane_id.copy()
    traffic_df = None
    for id in unique_lane_ids:
        df_lane = _compute_traffic_states_single_lane(trajectory_df, dx, dt, starttime, endtime, id, min_position, testbed_length)
        if traffic_df is None:
            traffic_df = df_lane.copy()
        else:
            traffic_df = pd.concat((traffic_df, df_lane))
    del df_lane
    traffic_df = traffic_df.reset_index().drop(columns="index")
    return traffic_df
