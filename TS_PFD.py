###########################################################################
# IMPORTS
###########################################################################
import os
import sys
import pytz
import gpflow
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tensorflow as tf

from typing import List, Optional
from gpflow.utilities import print_summary, positive
from tensorflow_probability import bijectors as tfb

import datetime
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import constants as cs

###########################################################################
# ANISOTROPIC GP KERNEL CLASS AND METHODS
###########################################################################
def fill_nan(data):
    """Fill nan values with the nearest non-nan value"""
    ind = np.arange(data.shape[0])
    for i in range(data.shape[1]):
        data[:, i] = np.interp(ind, ind[~np.isnan(data[:, i])], data[~np.isnan(data[:, i]), i])
    return data


def SE(r, variance):
    return variance * tf.exp(-0.5 * r ** 2)


def Matern52(r, variance):
    sqrt5 = np.sqrt(5.0)
    return variance * (1 + sqrt5 * r + 5.0 / 3.0 * r ** 2) * tf.exp(-sqrt5 * r)


def Matern32(r, variance):
    sqrt3 = np.sqrt(3.0)
    return variance * (1 + sqrt3 * r) * tf.exp(-sqrt3 * r)


def Rational_Quadratic(r, variance, alpha):
    return variance * (1 + r ** 2 / (2 * alpha)) ** (-alpha)


# Define the directional kernel
class Directional_Kernel(gpflow.kernels.Kernel):
    def __init__(self, kernel, variance=1.0, lengthscales=(1.0, 1.0), theta=0.01, alpha=None):
        super().__init__(name=f"Directional_{kernel.__name__}")
        sigmoid = tfb.Sigmoid(tf.cast(0.0, tf.float64), tf.cast(2*np.pi, tf.float64))
        self.variance = gpflow.Parameter(variance, transform=positive(), dtype=tf.float64)
        self.theta = gpflow.Parameter(theta, transform=sigmoid, dtype=tf.float64)
        self.lengthscale = gpflow.Parameter(lengthscales, transform=positive(), dtype=tf.float64)
        self.kernel = kernel
        if alpha is not None:
            self.alpha = gpflow.Parameter(alpha, transform=positive(), dtype=tf.float64)

    def square_distance(self, X, X2=None):
        if X2 is None:
            X2 = X
        rotation_matrix = tf.stack([tf.cos(self.theta), -tf.sin(self.theta), tf.sin(self.theta), tf.cos(self.theta)])
        rotation_matrix = tf.reshape(rotation_matrix, [2, 2])
        X = tf.matmul(X, rotation_matrix)
        X2 = tf.matmul(X2, rotation_matrix)
        X_scaled = X / self.lengthscale
        X2_scaled = X2 / self.lengthscale
        X2_scaled = tf.transpose(X2_scaled)
        return tf.reduce_sum(X_scaled ** 2, 1, keepdims=True) - 2 * tf.matmul(X_scaled, X2_scaled) + tf.reduce_sum(X2_scaled ** 2, 0, keepdims=True)

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        # Only use the first two dimensions
        r2 = self.square_distance(X[:, 0:2], X2[:, 0:2])
        r = tf.sqrt(tf.maximum(r2, 1e-36))
        if self.kernel.__name__ == "Rational_Quadratic":
            return self.kernel(r, self.variance, self.alpha)
        else:
            return self.kernel(r, self.variance)

    def K_diag(self, X):
        return tf.fill(tf.shape(X[:, 0:2])[:-1], tf.squeeze(self.variance))


def _estimate_traffic_states_GP_single_Lane(pfd_traffic_df: pd.DataFrame, dx: float, dt: float, lane_id: int, 
                                            starttime: float, endtime: float, min_position: float, testbed_length: float):
    df = pfd_traffic_df[pfd_traffic_df["Lane_ID"] == lane_id].copy()

    n_t_rows, n_x_cols = int((endtime-starttime)/dt), int((testbed_length-min_position)/dx)
    Density_Mat = np.empty(shape=(n_t_rows, n_x_cols))
    Flow_Mat = np.empty(shape=(n_t_rows, n_x_cols))
    Speed_Mat = np.empty(shape=(n_t_rows, n_x_cols))
    Num_Observations_Mat = np.zeros(shape=(n_t_rows, n_x_cols))
    Density_Mat.fill(np.nan)
    Flow_Mat.fill(np.nan)
    Speed_Mat.fill(np.nan)

    # 1. average traffic states inside grid based on coincidence of centroid within grid cell
    df['Time_Bin'] = pd.cut(x=df['center_T'], bins=np.arange(0, endtime-starttime+1, dt))
    df['Position_Bin'] = pd.cut(x=df['center_X'], bins=np.arange(min_position, testbed_length+1, dx))
    df = df.dropna()
    grouped = df.groupby(by=['Time_Bin', 'Position_Bin']).agg(
        Num_Observations=pd.NamedAgg(column="Density", aggfunc="count"),
        Density=pd.NamedAgg(column="Density", aggfunc="mean"),
        Flow=pd.NamedAgg(column="Flow", aggfunc="mean"),
        Speed=pd.NamedAgg(column="Speed", aggfunc="mean"),
    )
    grouped = grouped.reset_index().dropna()
    for _, row in grouped.iterrows():
        it = int(row['Time_Bin'].left/dt)
        jx = int(row['Position_Bin'].left/dx)
        if row['Num_Observations'] < cs.PFD_MIN_OBSERVATIONS:
            continue
        Density_Mat[it, jx] = row['Density']
        Flow_Mat[it, jx] = row['Flow']
        Speed_Mat[it, jx] = row['Speed']
        Num_Observations_Mat[it, jx] = row['Num_Observations']
    Density_Mat = np.flip(Density_Mat.T, axis=0)
    Flow_Mat = np.flip(Flow_Mat.T, axis=0)
    Speed_Mat = np.flip(Speed_Mat.T, axis=0)
    Num_Observations_Mat = np.flip(Num_Observations_Mat.T, axis=0)

    # 2. Train Rotated Anisotropic GP model
    mask = ~np.isnan(Speed_Mat)

    train_X = np.where(mask == 1)
    #train_Y = [Density_Mat[train_X].reshape(-1, 1), Flow_Mat[train_X].reshape(-1, 1), Speed_Mat[train_X].reshape(-1, 1)]
    train_Y = [Density_Mat[train_X].reshape(-1, 1), Speed_Mat[train_X].reshape(-1, 1)]
    train_Y = np.concatenate(train_Y, axis=-1)
    train_X = np.concatenate([x.reshape([-1, 1]) for x in train_X], axis = 1)
    train_X = train_X.astype(np.float64)

    mean_Y = np.mean(train_Y, axis=0)[np.newaxis, :]
    std_Y = np.std(train_Y, axis=0)[np.newaxis, :]
    train_Y  = (train_Y - mean_Y) / std_Y  # standardize

    n = min(max(int((~np.isnan(Speed_Mat)).sum() * 0.02), 100), 500)  # number of inducing points
    Z = np.random.permutation(train_X)[:n, :]  # initial location of inducing inputs

    kernel = Directional_Kernel(Matern52, lengthscales=[150.0, 13.0], theta=0.18, variance=0.2)
    model = gpflow.models.SGPR(data=(train_X, train_Y), kernel=kernel, mean_function=None,
                            inducing_variable=Z, noise_variance=0.3)
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=700))

    # 3. Use the trained GP model to make predictions
    test_X = np.where(mask >= 0)
    test_X = np.concatenate([x.reshape([-1, 1]) for x in test_X], axis=1)
    test_X = test_X.astype(np.float64)
    test_Y = model.predict_y(test_X)[0]
    predicted_Y = test_Y.numpy() * std_Y + mean_Y

    #Predicted_Density = predicted_Y[:, 0].reshape(Density_Mat.shape)
    #Predicted_Flow = predicted_Y[:, 1].reshape(Flow_Mat.shape)
    #Predicted_Speed = predicted_Y[:, 2].reshape(Speed_Mat.shape)
    Predicted_Density = predicted_Y[:, 0].reshape(Density_Mat.shape)
    Predicted_Speed = predicted_Y[:, 1].reshape(Speed_Mat.shape)
    Predicted_Flow = Predicted_Density * Predicted_Speed
    
    #Predicted_Density[mask] = Density_Mat[mask]
    #Predicted_Flow[mask] = Flow_Mat[mask]
    #Predicted_Speed[mask] = Speed_Mat[mask]

    return Predicted_Density, Predicted_Flow, Predicted_Speed, Num_Observations_Mat


def estimate_traffic_states_GP(pfd_traffic_df: pd.DataFrame, dx: float, dt: float, starttime: float, endtime: float, 
                               min_position: float, testbed_length: float, lane_id: Optional[List[int] | int] = None):
    if lane_id is not None and isinstance(lane_id, int):
        return _estimate_traffic_states_GP_single_Lane(pfd_traffic_df, dx, dt, lane_id, starttime, endtime, min_position, testbed_length)
    
    if lane_id is None:
        unique_lane_ids = pfd_traffic_df["Lane_ID"].unique()
    elif isinstance(lane_id, list):
        unique_lane_ids = lane_id.copy()
    results_dict = {}
    for id in unique_lane_ids:
        Predicted_Density, Predicted_Flow, Predicted_Speed, Num_Observations_Mat = _estimate_traffic_states_GP_single_Lane(
            pfd_traffic_df, dx, dt, id, starttime, endtime, min_position, testbed_length
        )
        results_dict[id] = {
            "Density": Predicted_Density,
            "Flow": Predicted_Flow,
            "Speed": Predicted_Speed,
            "Num_Observations": Num_Observations_Mat
        }
    return results_dict


###########################################################################
# METHODS: PFD
###########################################################################
def _get_trapezoid_center(x1_0, x1_t, xN_0, xN_t, dt):
    a = 0.5*dt*(x1_t-xN_t + x1_0-xN_0)
    cH = dt*(1 + min(x1_t-xN_t, x1_0-xN_0)/(x1_t-xN_t + x1_0-xN_0))/3.0
    cV = dt*(x1_t**2 + x1_0**2 + x1_0*x1_t - xN_t**2 - xN_0**2 - xN_0*xN_t)/3.0
    cV = cV/(2.0*a)
    return cH, cV

def _compute_PFD_traffic_states_single_lane(trajectory_df: pd.DataFrame, pen_rate: float, sampling_interval: float, starttime: float, endtime: float, lane_id: int, 
                                            min_position: float, testbed_length: float, random_seed: int):
    n_platoon = 2

    df = trajectory_df[trajectory_df["Lane_ID"] == lane_id].copy()
    df['Time'] = df['Global_Time'] - starttime
    df[['Next_Position', 'Next_Time', 'Next_Space_Hdwy']] = df.groupby('Vehicle_ID')[['Position', 'Time', 'Space_Hdwy']].shift(-1)
    df = df.dropna()
    grouped = df.groupby(['Vehicle_ID'])
    df['Num_Lanes'] = grouped['Lane_ID'].transform('nunique')
    df = df[(df["Lane_ID"] == lane_id) & (df['Time'] >= 0) & (df['Num_Lanes'] == 1)]

    num_vehicles = df.Vehicle_ID.nunique()
    if pen_rate < 1.0:
        sampled_vehicles = df.groupby(['Vehicle_ID'])['Num_Lanes'].mean().sample(n=int(pen_rate*num_vehicles), replace=False, random_state=random_seed)
        df = df[df['Vehicle_ID'].isin(sampled_vehicles.index)]

    df = df[(df["Next_Time"] - df["Time"]) <= sampling_interval]
    grouped = df[["Global_Time", "Vehicle_ID", "Preceeding", "Position", "Next_Position", "Time", "Next_Time", "Space_Hdwy", "Next_Space_Hdwy"]].copy()

    avg_veh_length = 5.0*cs.METER2FT
    grouped["TTT"] =  sampling_interval * (n_platoon-1) * cs.SEC2HR # hour
    grouped["x0"] = grouped["Position"]
    grouped["xt"] = grouped["Next_Position"]
    grouped["xL0"] = grouped["Position"] + grouped["Space_Hdwy"]
    grouped["xLt"] = grouped["Next_Position"] + grouped["Next_Space_Hdwy"]
    grouped["TTD"] = (grouped["xt"]-grouped["x0"]) * cs.FT2MILE # mile
    grouped["Area"] = 0.5*(sampling_interval*cs.SEC2HR)*(grouped["xL0"]-grouped["x0"] + grouped["xLt"]-grouped["xt"] + 2*avg_veh_length)*cs.FT2MILE
    grouped["Density"] = grouped["TTT"] / grouped["Area"]
    grouped["Flow"] = grouped["TTD"] / grouped["Area"]
    grouped["Speed"] = grouped["Flow"] / grouped["Density"]

    # from lower left corner of trapezoid and then clockwise
    grouped["A_x00"] = grouped["x0"] - avg_veh_length
    grouped["A_x01"] = grouped["xL0"]
    grouped["A_x11"] = grouped["xLt"]
    grouped["A_x10"] = grouped["xt"] - avg_veh_length
    grouped["A_t00"] = grouped["Time"]
    grouped["A_t01"] = grouped["Time"]
    grouped["A_t11"] = grouped["Time"] + sampling_interval
    grouped["A_t10"] = grouped["Time"] + sampling_interval

    grouped["Platoon_Vehicle_IDs"] = grouped[["Preceeding", "Vehicle_ID"]].values.tolist()

    grouped["centers"] = grouped.apply(lambda v: _get_trapezoid_center(x1_0=v.A_x01, x1_t=v.A_x11, xN_0=v.A_x00, xN_t=v.A_x10, dt=sampling_interval), axis=1)
    grouped['center_T'] = grouped['centers'].apply(lambda x: x[0]) + grouped["Time"]
    grouped['center_X'] = grouped['centers'].apply(lambda x: x[1])

    grouped = grouped.drop(columns=["Next_Position", "Position", "Space_Hdwy", "Next_Space_Hdwy", "Next_Time", "Preceeding", "Vehicle_ID", 
                                    "TTT", "TTD", "Area", "centers", "x0", "xt", "xL0", "xLt"])
    grouped["Lane_ID"] = lane_id
    grouped = grouped.dropna()
    return grouped

    """
    grouped = df.groupby(['Time', 'Vehicle_ID'])
    avg_veh_length = 5.0*cs.METER2FT

    list_platoon_states = []
    for (t, veh_id), group_df in grouped:
        if group_df.isnull().any().any():
            print(group_df)
            sys.exit(1)
        if group_df.Next_Time.item() - group_df.Time.item() > sampling_interval:
            continue
        total_travel_time = sampling_interval * (n_platoon-1) * cs.SEC2HR
        # ego vehicle positions
        x0, xt = group_df.Position.item(), group_df.Next_Position.item()
        xL0, xLt = x0+group_df.Space_Hdwy.item(), xt+group_df.Next_Space_Hdwy.item()
        total_travel_distance = (xt-x0) * cs.FT2MILE
        area = 0.5*(sampling_interval*cs.SEC2HR)*(xL0-x0 + xLt-xt + 2*avg_veh_length)*cs.FT2MILE
        k = total_travel_time / area # veh/mile
        q = total_travel_distance / area # veh/hour
        v = q / k # mph
        t_00, t_01 = t, t
        t_10, t_11 = t+sampling_interval, t+sampling_interval
        timestamp = t + starttime
        x_00, x_01 = x0-avg_veh_length, xL0
        x_10, x_11 = xt-avg_veh_length, xLt
        cH, cV = _get_trapezoid_center(x1_0=x_01, x1_t=x_11, xN_0=x_00, xN_t=x_10, dt=sampling_interval)
        center_T, center_X = np.round(t+cH, 2), np.round(cV, 2)
        list_platoon_states.append((timestamp, t, [group_df.Preceeding.item(), veh_id],
                                    k, q, v, center_T, center_X,
                                    t_00, t_01, t_11, t_10, x_00, x_01, x_11, x_10)) # from lower left corner of parallelogram and then clockwise
    df_platoon_states = pd.DataFrame(list_platoon_states)
    df_platoon_states.columns = [
        'Global_Time', 'Time', 'Platoon_Vehicle_IDs', 
        'Density', 'Flow', 'Speed', 'center_T', 'center_X',
        'A_t00', 'A_t01', 'A_t11', 'A_t10', 'A_x00', 'A_x01', 'A_x11', 'A_x10', # from lower left corner of parallelogram and then clockwise
    ]
    df_platoon_states["Lane_ID"] = lane_id
    df_platoon_states = df_platoon_states.dropna()
    return df_platoon_states
    """


def compute_PFD_traffic_states(trajectory_df: pd.DataFrame, pen_rate: float, sampling_interval: float, starttime: float, endtime: float, 
                               min_position: float, testbed_length: float, lane_id: Optional[List[int] | int] = None, random_seed: Optional[int] = None):
    if lane_id is not None and isinstance(lane_id, int):
        return _compute_PFD_traffic_states_single_lane(trajectory_df, pen_rate, sampling_interval, starttime, endtime, lane_id, min_position, testbed_length, random_seed)
    
    if lane_id is None:
        unique_lane_ids = trajectory_df["Lane_ID"].unique()
    elif isinstance(lane_id, list):
        unique_lane_ids = lane_id.copy()
    traffic_df = None
    for id in unique_lane_ids:
        df_lane = _compute_PFD_traffic_states_single_lane(trajectory_df, pen_rate, sampling_interval, starttime, endtime, id, min_position, testbed_length, random_seed)
        if traffic_df is None:
            traffic_df = df_lane.copy()
        else:
            traffic_df = pd.concat((traffic_df, df_lane))
    del df_lane
    traffic_df = traffic_df.reset_index().drop(columns="index")
    return traffic_df
