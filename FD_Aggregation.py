###########################################################################
# IMPORTS
###########################################################################
import os
import sys
import pytz

import numpy as np
import pandas as pd
import lmfit as lm
import matplotlib.pyplot as plt

from typing import List, Optional

import constants as cs

###########################################################################
# METHODS: Calibration of TFD
###########################################################################
def _nrmse_TFD(params, Ks, Qs, Vs):
    vf, w = params['vf'], params['w']
    k_jam = params['k_jam']
    Q_pred = np.minimum(vf*Ks, w*(k_jam - Ks))
    V_pred = Q_pred / Ks
    rmse_Q = np.sqrt(np.mean(np.square(Q_pred - Qs)))
    rmse_V = np.sqrt(np.mean(np.square(V_pred - Vs)))
    obj = rmse_Q/np.mean(Qs) + rmse_V/np.mean(Vs)
    return obj


def _calibrate_TFD(Ks, Qs, Vs):
    params = lm.create_params(
        vf = {'value': 10, 'min': 1e-02, 'max': 50},
        w = {'value': 5, 'min': 1e-02, 'max': 20},
        k_jam = {'value': 150, 'min': 1e-02, 'max': 350}
    )
    res = lm.minimize(_nrmse_TFD, params, args=(Ks, Qs, Vs), method='differential_evolution')
    #print(lm.fit_report(res.params))
    vf, w = res.params['vf'].value, res.params['w'].value
    k_jam = res.params['k_jam'].value
    k_crit = w * k_jam / (w + vf)
    K_test = np.linspace(0, k_jam, 200)
    Q_pred = np.minimum(vf*K_test, w*(k_jam - K_test))
    V_pred = Q_pred / K_test
    return K_test, Q_pred, V_pred, vf, w, k_jam, k_crit

###########################################################################
# METHODS: Calibration of Exponential FD
###########################################################################
def _nrmse_ExpFD(params, Ks, Qs, Vs):
    vf, alpha = params['vf'], params['alpha']
    k_crit = params['k_crit']
    V_pred = vf * np.exp(-np.power(Ks/k_crit, alpha)/alpha)
    Q_pred = Ks * V_pred
    rmse_Q = np.sqrt(np.mean(np.square(Q_pred - Qs)))
    rmse_V = np.sqrt(np.mean(np.square(V_pred - Vs)))
    obj = rmse_Q/np.mean(Qs) + rmse_V/np.mean(Vs)
    return obj


def _calibrate_ExpFD(Ks, Qs, Vs):
    params = lm.create_params(
        vf = {'value': 10, 'min': 1e-02, 'max': 50},
        alpha = {'value': 5, 'min': 1e-03, 'max': 50},
        k_crit = {'value': 150, 'min': 1e-02, 'max': 350}
    )
    res = lm.minimize(_nrmse_ExpFD, params, args=(Ks, Qs, Vs), method='differential_evolution')
    #print(lm.fit_report(res.params))
    vf, alpha = res.params['vf'].value, res.params['alpha'].value
    k_crit = res.params['k_crit'].value
    K_test = np.linspace(0, 3*k_crit, 200)
    V_pred = vf * np.exp(-np.power(K_test/k_crit, alpha)/alpha)
    Q_pred = K_test * V_pred
    return K_test, Q_pred, V_pred, vf, alpha, k_crit

###########################################################################
# METHODS
###########################################################################
def _plot_FD(Ks, Qs, Vs, k_FD, q_FD, v_FD, vf, k_crit, w=None, k_jam=None, alpha=None):
    max_density = max(np.amax(Ks), np.amax(k_FD))
    max_flow = max(np.amax(Qs), np.amax(q_FD))
    max_speed = max(np.amax(Vs), np.amax(v_FD))

    """
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].scatter(Ks, Qs)
    axs[0].plot(k_FD, q_FD, color='black', linestyle='dashed')
    axs[0].set_xlim(0, max_density+10)
    axs[0].set_ylim(0, max_flow+100)
    axs[0].set_xlabel("Density (veh/mile)")
    axs[0].set_ylabel("Flow (veh/hour)")

    axs[1].scatter(Ks, Vs)
    axs[1].plot(k_FD, v_FD, color='black', linestyle='dashed')
    axs[1].set_xlim(0, max_density+10)
    axs[1].set_ylim(0, max_speed+5)
    axs[1].set_xlabel("Density (veh/mile)")
    axs[1].set_ylabel("Speed (mph)")
    if w is not None and k_jam is not None:
        txt = f"vf = {vf:.2f} mph \nw = {w:.2f} mph \nk_crit = {k_crit:.2f} veh/mile \nk_jam = {k_jam:.2f} veh/mile"
        axs[1].text(0.4*max_density, 0.8*max_speed, txt)
    elif alpha is not None:
        txt = f"vf = {vf:.2f} mph \nalpha = {alpha:.2f} \nk_crit = {k_crit:.2f} veh/mile"
        axs[1].text(0.4*max_density, 0.8*max_speed, txt)
    fig.tight_layout()
    return fig
    """

    fig = plt.figure(figsize=(5,4))

    color = "tab:blue"
    plt.scatter(Ks, Qs, color=color, alpha=0.5, label="$k-q$ FD")
    p1 = plt.plot(k_FD, q_FD, color=color, linestyle='dashed', label="$k-q$ FD")
    plt.xlim(0, max_density+10)
    plt.ylim(0, max_flow+100)
    plt.xlabel("Density $\it{[veh/mile]}$")
    plt.ylabel("Flow $\it{[veh/h]}$", color=color)
    plt.tick_params(axis='y', labelcolor=color)

    ax1 = plt.gca()

    color = "tab:red"
    ax2 = ax1.twinx()
    ax2.scatter(Ks, Vs, color=color, alpha=0.5, label="$k-v$ FD")
    p2 = ax2.plot(k_FD, v_FD, color=color, linestyle='dashed', label="$k-v$ FD")
    ax2.set_xlim(0, max_density+10)
    ax2.set_ylim(0, max_speed+5)
    ax2.set_ylabel("Speed $\it{[mph]}$", color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    lns = p1 + p2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)
    fig.tight_layout()
    return fig




def aggregate_FD(traffic_df: pd.DataFrame, lane_id: int, bin_width: float, FD_mode: str = "TFD"):
    df = traffic_df[traffic_df["Lane_ID"] == lane_id].copy()
    df['Density_Bin'] = pd.cut(x=df['Density'], bins=np.arange(0, cs.PFD_MAX_DENSITY, bin_width))
    agg_df = df.groupby(["Density_Bin"]).agg({
        "Density": "mean", 
        "Flow": "mean",
        "Speed": "mean",
        "Density_Bin": "count"
    })
    agg_df = agg_df.rename(columns={"Density_Bin": "Num_Observations"})
    agg_df = agg_df.dropna()
    agg_df = agg_df[agg_df["Num_Observations"] >= cs.PFD_MIN_OBSERVATIONS]

    alpha, w, k_jam = None, None, None
    if FD_mode == "TFD":
        k_FD, q_FD, v_FD, vf, w, k_jam, k_crit = _calibrate_TFD(Ks=agg_df["Density"].to_numpy(), Qs=agg_df["Flow"].to_numpy(), Vs=agg_df["Speed"].to_numpy())
    elif FD_mode == "Exponential":
        k_FD, q_FD, v_FD, vf, alpha, k_crit = _calibrate_ExpFD(Ks=agg_df["Density"].to_numpy(), Qs=agg_df["Flow"].to_numpy(), Vs=agg_df["Speed"].to_numpy())
    else:
        raise ValueError
    
    fig = _plot_FD(
        agg_df["Density"].to_numpy(), agg_df["Flow"].to_numpy(), agg_df["Speed"].to_numpy(),
        k_FD, q_FD, v_FD, vf, k_crit, w, k_jam, alpha
    )
    return fig

