###########################################################################
# IMPORTS
###########################################################################
import os
import sys
import datetime
import pytz
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy as sp

from tqdm import tqdm
from typing import List, Optional

import constants as cs

###########################################################################
# METHODS
###########################################################################
def _compute_traffic_states_GT_single_lane(trajectory_df: pd.DataFrame, dx: float, dt: float, starttime: float, endtime: float, lane_id: int, 
                                           min_position: float, testbed_length: float):
    df = trajectory_df[trajectory_df["Lane_ID"] == lane_id].copy()
    df['Time'] = df['Global_Time'] - starttime
    df[['Next_Position', 'Next_Time']] = df.groupby('Vehicle_ID')[['Position', 'Time']].shift(-1)
    df = df.dropna()
    grouped = df.groupby(['Vehicle_ID'])
    df['Num_Lanes'] = grouped['Lane_ID'].transform('nunique')
    df = df[(df["Lane_ID"] == lane_id) & (df['Time'] >= 0) & (df['Num_Lanes'] == 1)]

    df['Travel_Distance'] = df['Next_Position'] - df['Position']
    df['Travel_Time'] = df['Next_Time'] - df['Time']

    df['Time_Bin'] = pd.cut(x=df['Time'], bins=np.arange(0, endtime-starttime+1, dt))
    df['Position_Bin'] = pd.cut(x=df['Position'], bins=np.arange(min_position, testbed_length+1, dx))
    df = df.dropna()
    
    grouped = df.groupby(by=['Time_Bin', 'Position_Bin']).agg(
        Num_Vehicles=pd.NamedAgg(column="Vehicle_ID", aggfunc=pd.Series.nunique),
        TTD=pd.NamedAgg(column="Travel_Distance", aggfunc="sum"),
        TTT=pd.NamedAgg(column="Travel_Time", aggfunc="sum"),
    )
    #grouped = df.groupby(by=['Time_Bin', 'Position_Bin'])[['Travel_Distance', 'Travel_Time']].sum()
    grouped = grouped.reset_index()
    grouped['Density'] = grouped['TTT'] / (dx*cs.FT2MILE*dt) # veh / mile
    grouped['Flow'] = grouped['TTD'] / (dx*dt*cs.SEC2HR) # veh / hour
    grouped['Speed'] = grouped['Flow'] / grouped['Density']
    grouped['t'] = grouped['Time_Bin'].apply(lambda x: x.left)
    grouped['x'] = grouped['Position_Bin'].apply(lambda x: x.left)
    grouped = grouped.astype({'t': 'float64', 'x': 'float64'})
    grouped = ASM(grouped)

    n_t_rows, n_x_cols = int((endtime-starttime)/dt), int((testbed_length-min_position)/dx)
    Density_Mat = np.empty(shape=(n_t_rows, n_x_cols))
    Flow_Mat = np.empty(shape=(n_t_rows, n_x_cols))
    Speed_Mat = np.empty(shape=(n_t_rows, n_x_cols))
    Num_Vehicles_Mat = np.empty(shape=(n_t_rows, n_x_cols))
    Density_Mat.fill(np.nan)
    Flow_Mat.fill(np.nan)
    Speed_Mat.fill(np.nan)
    Num_Vehicles_Mat.fill(np.nan)
    for _, row in grouped.iterrows():
        it = int(row['Time_Bin'].left/dt)
        jx = int(row['Position_Bin'].left/dx)
        if it >= Density_Mat.shape[0] or jx >= Density_Mat.shape[1]:
            continue
        #if abs(row['Travel_Time']) <= 1e-03 and abs(row['Travel_Distance']) <= 1e-03:
        #    continue
        Density_Mat[it, jx] = row['Density_ASM']
        Flow_Mat[it, jx] = row['Flow_ASM']
        Speed_Mat[it, jx] = row['Speed_ASM']
        Num_Vehicles_Mat[it, jx] = row['Num_Vehicles']
    Density_Mat = np.flip(Density_Mat.T, axis=0)
    Flow_Mat = np.flip(Flow_Mat.T, axis=0)
    Speed_Mat = np.flip(Speed_Mat.T, axis=0)
    Num_Vehicles_Mat = np.flip(Num_Vehicles_Mat.T, axis=0)

    return Density_Mat, Flow_Mat, Speed_Mat, Num_Vehicles_Mat


def compute_traffic_states_GT(trajectory_df: pd.DataFrame, dx: float, dt: float, starttime: float, endtime: float, 
                              min_position: float, testbed_length: float, lane_id: Optional[List[int] | int] = None):
    if lane_id is not None and isinstance(lane_id, int):
        return _compute_traffic_states_GT_single_lane(trajectory_df, dx, dt, starttime, endtime, lane_id, min_position, testbed_length)
    
    if lane_id is None:
        unique_lane_ids = trajectory_df["Lane_ID"].unique()
    elif isinstance(lane_id, list):
        unique_lane_ids = lane_id.copy()
    results_dict = {}
    for id in unique_lane_ids:
        Density_Mat, Flow_Mat, Speed_Mat, Num_Vehicles_Mat = _compute_traffic_states_GT_single_lane(
            trajectory_df, dx, dt, starttime, endtime, id, min_position, testbed_length
        )
        results_dict[id] = {
            "Density": Density_Mat,
            "Flow": Flow_Mat,
            "Speed": Speed_Mat,
            "Num_Vehicles": Num_Vehicles_Mat
        }
    return results_dict

###########################################################################
# ASM
###########################################################################
def beta_free_flow(x, t, x_s, t_s, x_win, t_win, c_free=cs.FREE_FLOW_SPEED):
    c_free_fps = c_free*cs.MILE2FT/cs.HR2SEC
    dt = t-t_s- 3600*(x-x_s)/c_free_fps
    dx = x-x_s
    return np.exp(-(2*np.abs(dx)/x_win + 2*np.abs(dt)/t_win))


def beta_cong_flow(x, t, x_s, t_s, x_win, t_win, c_cong=cs.BACKWARDS_WAVE_SPEED):
    c_cong_fps = c_cong*cs.MILE2FT/cs.HR2SEC
    dt = t-t_s- 3600*(x-x_s)/c_cong_fps
    dx = x-x_s
    return np.exp(-(2*np.abs(dx)/x_win + 2*np.abs(dt)/t_win))


def EGTF(x, t, smooth_x_window, smooth_t_window, raw_df):
    local_raw_df = raw_df[(np.abs(raw_df.t - t)<=(smooth_t_window/2)) & (np.abs(raw_df.x - x)<=(smooth_x_window/2))]
    local_raw_df = local_raw_df.copy()
    EGTF_v_free, EGTF_v_cong = 80, 80
    EGTF_k_free, EGTF_k_cong = 0, 0
    EGTF_q_free, EGTF_q_cong = 2000, 2000
    # Now apply your functions
    local_raw_df['beta_free'] = local_raw_df.apply(lambda v: beta_free_flow(x, t, v.x, v.t, smooth_x_window, smooth_t_window), axis=1)
    local_raw_df['beta_cong'] = local_raw_df.apply(lambda v: beta_cong_flow(x, t, v.x, v.t, smooth_x_window, smooth_t_window), axis=1)
    if((sum(local_raw_df.beta_free)!=0) & (sum(local_raw_df.beta_cong)!=0)):
        EGTF_v_free = sum(local_raw_df.beta_free * local_raw_df.Speed) / sum(local_raw_df.beta_free)
        EGTF_v_cong = sum(local_raw_df.beta_cong * local_raw_df.Speed) / sum(local_raw_df.beta_cong)
        EGTF_k_free = sum(local_raw_df.beta_free * local_raw_df.Density) / sum(local_raw_df.beta_free)
        EGTF_k_cong = sum(local_raw_df.beta_cong * local_raw_df.Density) / sum(local_raw_df.beta_cong)
        EGTF_q_free = sum(local_raw_df.beta_free * local_raw_df.Flow) / sum(local_raw_df.beta_free)
        EGTF_q_cong = sum(local_raw_df.beta_cong * local_raw_df.Flow) / sum(local_raw_df.beta_cong)
    v = min(EGTF_v_free, EGTF_v_cong)
    tanh_term = np.tanh((36-v) / 12.43)
    w = 0.5*(1+tanh_term)
    EGTF_v = w*EGTF_v_cong + (1-w)*EGTF_v_free
    EGTF_k = w*EGTF_k_cong + (1-w)*EGTF_k_free
    EGTF_q = w*EGTF_q_cong + (1-w)*EGTF_q_free
    return [EGTF_k, EGTF_q, EGTF_v]


def ASM(df, smooth_x_window=0.60*cs.MILE2FT, smooth_t_window=70.0):
    raw_df = df.dropna().copy()
    tqdm.pandas(desc="Processing EGTF")
    df['EGTF_Out'] = df.progress_apply(lambda v: EGTF(v.x, v.t, smooth_x_window, smooth_t_window, raw_df), axis=1)
    df['Density_ASM'] = df['EGTF_Out'].apply(lambda x: x[0])
    df['Flow_ASM'] = df['EGTF_Out'].apply(lambda x: x[1])
    df['Speed_ASM'] = df['EGTF_Out'].apply(lambda x: x[2])
    df = df.drop(columns=['EGTF_Out'])
    return df