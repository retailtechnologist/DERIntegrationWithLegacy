"""
DER Grid Integration Simulation
Based on: "Integrating Distributed Energy Resources (DERs) into Legacy Grid Infrastructure"
Author: Bhanu Singh | retailtechnologist@outlook.com
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from enum import Enum
import warnings
from scipy import stats
import time

warnings.filterwarnings('ignore')
np.random.seed(42)

# ─── Constants ────────────────────────────────────────────────────────────────
V_MIN_PU  = 0.95
V_MAX_PU  = 1.05
V_NOM     = 1.0
F_NOM     = 60.0
F_MIN     = 59.3
F_MAX     = 60.5

# ─── Enums ────────────────────────────────────────────────────────────────────
class DERType(Enum):
    SOLAR_PV         = "Solar PV"
    WIND             = "Wind"
    BESS             = "Battery Storage"
    EV               = "Electric Vehicle"
    DEMAND_RESPONSE  = "Demand Response"

class InverterType(Enum):
    GRID_FOLLOWING = "Grid-Following (GFL)"
    GRID_FORMING   = "Grid-Forming (GFM)"

class GridMode(Enum):
    LEGACY     = "Legacy Grid"
    MODERN_DER = "Modern DER-Integrated"

# ─── DER Node ─────────────────────────────────────────────────────────────────
@dataclass
class DERNode:
    node_id: str
    der_type: DERType
    rated_kw: float
    feeder_position: float        # 0 = substation, 1 = feeder end
    smart_inverter: bool = False
    inverter_type: InverterType = InverterType.GRID_FOLLOWING

    def output_kw(self, hour: float, irradiance: float, wind_ms: float) -> float:
        if self.der_type == DERType.SOLAR_PV:
            return self.rated_kw * irradiance * np.random.uniform(0.95, 1.0)
        if self.der_type == DERType.WIND:
            if wind_ms < 3:   return 0.0
            if wind_ms < 12:  return self.rated_kw * ((wind_ms - 3) / 9) ** 3
            if wind_ms < 25:  return self.rated_kw
            return 0.0
        if self.der_type == DERType.DEMAND_RESPONSE:
            return -self.rated_kw * np.random.uniform(0.3, 0.8)
        return 0.0

# ─── Battery Energy Storage System ────────────────────────────────────────────
@dataclass
class BatteryStorage:
    capacity_kwh: float
    power_kw: float
    soc: float = 0.5
    soc_min: float = 0.10
    soc_max: float = 0.90
    eta: float = 0.95            # round-trip efficiency

    def charge(self, kw: float, dt: float = 0.25) -> float:
        energy = kw * dt * self.eta
        new_soc = min(self.soc + energy / self.capacity_kwh, self.soc_max)
        self.soc = new_soc
        return kw

    def discharge(self, kw: float, dt: float = 0.25) -> float:
        energy = kw * dt / self.eta
        new_soc = max(self.soc - energy / self.capacity_kwh, self.soc_min)
        actual = (self.soc - new_soc) * self.capacity_kwh / dt * self.eta
        self.soc = new_soc
        return actual

# ─── Smart Inverter (IEEE 1547-2018) ──────────────────────────────────────────
class SmartInverter:
    def __init__(self, inv_type: InverterType, rated_kva: float):
        self.inv_type  = inv_type
        self.rated_kva = rated_kva

    def volt_var(self, v_pu: float, p_kw: float) -> float:
        """Reactive power injection/absorption based on voltage (Volt-VAR curve)."""
        q_max = np.sqrt(max(self.rated_kva**2 - p_kw**2, 0.0))
        if 0.97 <= v_pu <= 1.03:
            return 0.0
        elif v_pu < 0.97:
            return min(q_max * (0.97 - v_pu) / 0.05, q_max)
        else:
            return -min(q_max * (v_pu - 1.03) / 0.05, q_max)

    def volt_watt(self, v_pu: float, p_kw: float) -> float:
        """Curtail active power when voltage is too high."""
        if v_pu <= 1.05: return p_kw
        if v_pu >= 1.10: return 0.0
        return p_kw * (1.0 - (v_pu - 1.05) / 0.05)

    def freq_watt(self, f_hz: float, p_kw: float) -> float:
        """Adjust active power based on frequency deviation."""
        droop = 0.05
        if f_hz < F_MIN:
            return min(p_kw * (1 + droop * (F_NOM - f_hz) / F_NOM), self.rated_kva)
        if f_hz > F_MAX:
            return max(p_kw * (1 - droop * (f_hz - F_NOM) / F_NOM), 0.0)
        return p_kw

    def virtual_inertia(self, f_hz: float, df_dt: float) -> float:
        """Grid-forming only: synthetic inertia power response."""
        if self.inv_type != InverterType.GRID_FORMING:
            return 0.0
        H = 5.0
        return -2.0 * H * df_dt * self.rated_kva / F_NOM

# ─── Distribution Feeder ──────────────────────────────────────────────────────
@dataclass
class GridFeeder:
    feeder_id: str
    length_km: float
    rated_kw: float
    R_ohm_km: float = 0.30
    X_ohm_km: float = 0.10
    V_base_kv: float = 12.47

    def voltage_pu(self, net_kw: float, q_kvar: float = 0.0) -> float:
        R = self.R_ohm_km * self.length_km
        X = self.X_ohm_km * self.length_km
        dV = (net_kw * R + q_kvar * X) / (self.V_base_kv ** 2 * 1000.0)
        return V_NOM - dV

# ─── DERMS ────────────────────────────────────────────────────────────────────
class DERMS:
    """Simplified DERMS: coordinates BESS + demand response + volt-VAR."""

    def __init__(self, feeder: GridFeeder, bess: BatteryStorage):
        self.feeder = feeder
        self.bess   = bess

    def dispatch(self, solar_kw: float, load_kw: float,
                 v_pu: float, hour: float) -> Dict:
        bess_kw = 0.0
        curtail = 0.0
        dr_kw   = 0.0

        # High-voltage: charge BESS or curtail
        if v_pu > V_MAX_PU:
            if self.bess.soc < self.bess.soc_max:
                charge = min(solar_kw * 0.3, self.bess.power_kw)
                self.bess.charge(charge)
                bess_kw = -charge
            if v_pu > 1.07:
                curtail = solar_kw * 0.15

        # Low-voltage: discharge BESS
        elif v_pu < V_MIN_PU and self.bess.soc > self.bess.soc_min:
            dis = min(load_kw * 0.2, self.bess.power_kw)
            bess_kw = self.bess.discharge(dis)

        # Duck-curve midday: store excess solar
        elif 10 <= hour <= 14 and solar_kw > load_kw * 0.75:
            excess = solar_kw - load_kw * 0.6
            charge = min(excess, self.bess.power_kw)
            if charge > 0 and self.bess.soc < self.bess.soc_max:
                self.bess.charge(charge)
                bess_kw = -charge

        # Evening peak: discharge stored energy
        elif 17 <= hour <= 21 and self.bess.soc > 0.30:
            dis = min(load_kw * 0.30, self.bess.power_kw)
            bess_kw = self.bess.discharge(dis)

        # Demand response when feeder near limit
        if load_kw > self.feeder.rated_kw * 0.85:
            dr_kw = load_kw * 0.10

        managed_net = load_kw - solar_kw - bess_kw - dr_kw + curtail
        return {
            "bess_kw":   bess_kw,
            "bess_soc":  self.bess.soc,
            "curtail_kw": curtail,
            "dr_kw":     dr_kw,
            "net_kw":    managed_net,
            "v_status":  "OK" if V_MIN_PU <= v_pu <= V_MAX_PU else "VIOLATION",
        }

# ─── Profiles ─────────────────────────────────────────────────────────────────
def solar_irr(hour: float) -> float:
    if hour < 6 or hour > 20: return 0.0
    return np.exp(-0.5 * ((hour - 13.0) / 3.5) ** 2) * np.random.uniform(0.88, 1.0)

def load_norm(hour: float) -> float:
    return (0.30
            + 0.55 * np.exp(-0.5 * ((hour - 8.0) / 2.0) ** 2)
            + 0.85 * np.exp(-0.5 * ((hour - 19.0) / 2.0) ** 2)
            + np.random.normal(0, 0.02))

def wind_ms(hour: float) -> float:
    return max(0.0, 6.0 + 2.0 * np.sin(2 * np.pi * (hour - 14) / 24)
               + np.random.normal(0, 1.0))

# ─── 24-hour Simulation ───────────────────────────────────────────────────────
def run_24h(mode: GridMode,
            feeder_kw: float = 500,
            solar_kw_rated: float = 150,
            bess_kwh: float = 200,
            bess_kw: float = 50) -> pd.DataFrame:

    bess   = BatteryStorage(bess_kwh, bess_kw)
    feeder = GridFeeder("F1", 5.0, feeder_kw)
    derms  = DERMS(feeder, bess)
    rows   = []

    for hour in np.arange(0, 24, 0.25):
        irr    = solar_irr(hour)
        solar  = solar_kw_rated * irr
        load   = load_norm(hour) * feeder_kw
        load   = max(load, 0)

        if mode == GridMode.LEGACY:
            net   = load - solar
            v_pu  = np.clip(feeder.voltage_pu(net) + np.random.normal(0, 0.003), 0.87, 1.13)
            q     = 0.0
            bess_d = 0.0
            cut   = 0.0
            dr    = 0.0
            soc   = 0.5
        else:
            inv   = SmartInverter(InverterType.GRID_FORMING, solar_kw_rated * 1.1)
            raw_v = feeder.voltage_pu(load - solar)
            q     = inv.volt_var(raw_v, solar)
            v_pu  = np.clip(feeder.voltage_pu(load - solar, q) + np.random.normal(0, 0.002),
                            0.90, 1.10)
            d     = derms.dispatch(solar, load, v_pu, hour)
            bess_d = d["bess_kw"]
            cut   = d["curtail_kw"]
            dr    = d["dr_kw"]
            soc   = d["bess_soc"]
            net   = d["net_kw"]
            v_pu  = np.clip(feeder.voltage_pu(net, q) + np.random.normal(0, 0.002), 0.90, 1.10)

        rows.append({
            "hour":          hour,
            "mode":          mode.value,
            "load_kw":       round(load, 2),
            "solar_kw":      round(solar, 2),
            "net_load_kw":   round(load - solar - bess_d - dr, 2),
            "voltage_pu":    round(v_pu, 4),
            "freq_hz":       round(F_NOM + np.random.normal(0, 0.04), 4),
            "bess_soc":      round(soc, 3),
            "bess_kw":       round(bess_d, 2),
            "curtail_kw":    round(cut, 2),
            "q_kvar":        round(q, 2),
            "v_violation":   int(v_pu < V_MIN_PU or v_pu > V_MAX_PU),
            "reverse_flow":  int((load - solar) < 0),
        })

    return pd.DataFrame(rows)

# ─── Frequency Event Simulation ───────────────────────────────────────────────
def sim_freq_event(inv_type: InverterType,
                   f_init: float = 59.1,
                   duration_s: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    dt   = 0.1
    t    = np.arange(0, duration_s, dt)
    freq = np.empty(len(t))
    freq[0] = f_init
    inv  = SmartInverter(inv_type, 1000.0)

    for i in range(1, len(t)):
        f = freq[i - 1]
        df_dt = (F_NOM - f) / 8.0          # governor response
        if inv_type == InverterType.GRID_FORMING:
            vi = inv.virtual_inertia(f, df_dt) / 5000.0
            df_dt = df_dt * 0.6 + vi       # virtual inertia slows ROCOF
        freq[i] = min(freq[i - 1] + df_dt * dt, F_NOM)

    return t, freq

# ─── Large Test Dataset (10 000+ records) ─────────────────────────────────────
SCENARIOS = {"low_der": 0.10, "medium_der": 0.30,
             "high_der": 0.60, "very_high_der": 0.90}
SEASONS   = {
    "winter": {"sf": 0.60, "lf": 1.20},
    "spring": {"sf": 0.90, "lf": 0.90},
    "summer": {"sf": 1.00, "lf": 1.30},
    "fall":   {"sf": 0.75, "lf": 1.00},
}

def generate_test_dataset(n: int = 10_000) -> pd.DataFrame:
    np.random.seed(42)
    rows = []
    combos   = len(SCENARIOS) * len(SEASONS) * 2   # 2 modes
    per_combo = max(1, n // combos)

    for sc_name, der_pen in SCENARIOS.items():
        for sea_name, sp in SEASONS.items():
            for mode in [GridMode.LEGACY, GridMode.MODERN_DER]:
                for _ in range(per_combo):
                    hour  = np.random.uniform(0, 24)
                    f_cap = np.random.uniform(300, 800)
                    s_cap = f_cap * der_pen * sp["sf"]
                    lkw   = max(load_norm(hour) * f_cap * sp["lf"], 0.0)
                    irr   = solar_irr(hour)
                    skw   = s_cap * irr
                    wms   = wind_ms(hour)

                    if mode == GridMode.LEGACY:
                        net   = lkw - skw
                        v_pu  = np.clip(1.0 - (net / f_cap) * 0.10
                                        + np.random.normal(0, 0.022), 0.85, 1.15)
                        q     = 0.0
                        cut   = 0.0
                        soc   = 0.5
                        stab  = np.random.beta(2, 3)
                    else:
                        inv   = SmartInverter(InverterType.GRID_FORMING, s_cap * 1.1)
                        raw_v = np.clip(1.0 - ((lkw - skw) / f_cap) * 0.10
                                        + np.random.normal(0, 0.01), 0.85, 1.15)
                        q     = inv.volt_var(raw_v, skw)
                        v_pu  = np.clip(raw_v + q * 0.00015, 0.90, 1.10)
                        soc   = np.random.uniform(0.20, 0.80)
                        cut   = max(0.0, skw - f_cap * 0.30) if v_pu > V_MAX_PU else 0.0
                        stab  = np.random.beta(3, 2)

                    fdev = np.random.normal(0, 0.03 if mode == GridMode.LEGACY else 0.015)

                    rows.append({
                        "record_id":          len(rows),
                        "scenario":           sc_name,
                        "season":             sea_name,
                        "mode":               mode.value,
                        "der_penetration":    der_pen,
                        "hour":               round(hour, 2),
                        "feeder_kw":          round(f_cap, 1),
                        "load_kw":            round(lkw, 2),
                        "solar_kw":           round(skw, 2),
                        "net_load_kw":        round(lkw - skw, 2),
                        "voltage_pu":         round(v_pu, 4),
                        "freq_hz":            round(F_NOM + fdev, 4),
                        "bess_soc":           round(soc, 3),
                        "curtail_kw":         round(cut, 2),
                        "q_kvar":             round(q, 2),
                        "reverse_flow":       int((lkw - skw) < 0),
                        "v_violation":        int(v_pu < V_MIN_PU or v_pu > V_MAX_PU),
                        "overvoltage":        int(v_pu > V_MAX_PU),
                        "undervoltage":       int(v_pu < V_MIN_PU),
                        "f_violation":        int(abs(fdev) > (F_MAX - F_NOM)),
                        "hosting_cap_kw":     round(f_cap * (0.40 if mode == GridMode.LEGACY else 0.70), 1),
                        "stability_score":    round(stab, 4),
                        "renewables_util":    round(min(1.0, skw / max(lkw, 1.0)), 4),
                        "derms_active":       mode == GridMode.MODERN_DER,
                    })

    # Pad to exactly n rows by sampling with noise
    base = len(rows)
    while len(rows) < n:
        src = rows[np.random.randint(0, base)].copy()
        src["record_id"]    = len(rows)
        src["voltage_pu"]   = round(np.clip(src["voltage_pu"] + np.random.normal(0, 0.004), 0.88, 1.12), 4)
        src["v_violation"]  = int(src["voltage_pu"] < V_MIN_PU or src["voltage_pu"] > V_MAX_PU)
        src["overvoltage"]  = int(src["voltage_pu"] > V_MAX_PU)
        src["undervoltage"] = int(src["voltage_pu"] < V_MIN_PU)
        rows.append(src)

    return pd.DataFrame(rows[:n])

# ─── Plot 1: 24-hour Comparison ───────────────────────────────────────────────
def plot_24h(leg: pd.DataFrame, mod: pd.DataFrame):
    fig = plt.figure(figsize=(20, 22))
    fig.suptitle(
        "DER Grid Integration — 24-Hour Simulation\n"
        "Legacy Grid vs Modern DER (DERMS + Smart Inverters + BESS)\n"
        "Ref: Bhanu Singh, 'Integrating DERs into Legacy Grid Infrastructure'",
        fontsize=13, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.42, wspace=0.30)

    h_leg = leg["hour"]
    h_mod = mod["hour"]

    # — Voltage Profile ——————————————————————————————————————
    ax = fig.add_subplot(gs[0, :])
    ax.plot(h_leg, leg["voltage_pu"], "r-",  lw=1.5, label="Legacy Grid",           alpha=0.85)
    ax.plot(h_mod, mod["voltage_pu"], "g-",  lw=1.5, label="Modern DER (DERMS+GFM)", alpha=0.85)
    ax.axhline(V_MAX_PU, color="orange", ls="--", lw=1.2, label="IEEE 1547 Limits (0.95–1.05 pu)")
    ax.axhline(V_MIN_PU, color="orange", ls="--", lw=1.2)
    ax.fill_between(h_leg, V_MIN_PU, V_MAX_PU, alpha=0.08, color="green")
    ax.set(xlabel="Hour", ylabel="Voltage (pu)", title="Voltage Profile — IEEE 1547-2018 Acceptable Band")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_xlim(0, 24)

    # — Duck Curve ————————————————————————————————————————————
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.fill_between(h_leg, leg["load_kw"],  alpha=0.25, color="steelblue")
    ax2.fill_between(h_leg, leg["solar_kw"], alpha=0.35, color="gold")
    ax2.plot(h_leg, leg["load_kw"],       "b-",  lw=1.5, label="Load (kW)")
    ax2.plot(h_leg, leg["solar_kw"],      "y-",  lw=1.5, label="Solar (kW)")
    ax2.plot(h_mod, mod["net_load_kw"],   "g--", lw=2.0, label="Net Load w/ DERMS")
    ax2.set(xlabel="Hour", ylabel="Power (kW)", title="Duck Curve — Load vs Solar")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3); ax2.set_xlim(0, 24)

    # — BESS SoC ——————————————————————————————————————————————
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.fill_between(h_mod, mod["bess_soc"]*100, alpha=0.25, color="purple")
    ax3.plot(h_mod, mod["bess_soc"]*100, color="purple", lw=2, label="BESS SoC (%)")
    ax3.axhline(90, color="red",    ls="--", lw=1, label="SoC Max 90%")
    ax3.axhline(10, color="orange", ls="--", lw=1, label="SoC Min 10%")
    ax3.set(xlabel="Hour", ylabel="State of Charge (%)",
            title="Battery Storage (BESS) — State of Charge")
    ax3.set_ylim(0, 100); ax3.legend(fontsize=8); ax3.grid(alpha=0.3); ax3.set_xlim(0, 24)

    # — Volt-VAR Control ——————————————————————————————————————
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.bar(h_mod, mod["q_kvar"], width=0.22, color="teal", alpha=0.75)
    ax4.axhline(0, color="black", lw=0.8)
    ax4.set(xlabel="Hour", ylabel="Reactive Power (kVAR)",
            title="Smart Inverter Volt-VAR Control (IEEE 1547-2018)")
    ax4.grid(alpha=0.3, axis="y"); ax4.set_xlim(0, 24)

    # — Curtailment ———————————————————————————————————————————
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.fill_between(h_mod, mod["curtail_kw"], alpha=0.45, color="tomato")
    ax5.plot(h_mod, mod["curtail_kw"], "r-", lw=1.5, label="Curtailment (kW)")
    ax5.set(xlabel="Hour", ylabel="Curtailment (kW)",
            title="Solar Curtailment — DERMS Managed")
    ax5.legend(fontsize=8); ax5.grid(alpha=0.3); ax5.set_xlim(0, 24)

    # — Violation Bar Chart ———————————————————————————————————
    ax6 = fig.add_subplot(gs[3, :])
    hr_grp = lambda df_: df_.groupby(df_["hour"].astype(int))["v_violation"].mean() * 100
    lv = hr_grp(leg).reindex(range(24), fill_value=0)
    mv = hr_grp(mod).reindex(range(24), fill_value=0)
    x  = np.arange(24)
    ax6.bar(x - 0.18, lv, 0.36, color="red",   alpha=0.75, label="Legacy Grid")
    ax6.bar(x + 0.18, mv, 0.36, color="green",  alpha=0.75, label="Modern DER")
    ax6.set(xlabel="Hour", ylabel="Violation Rate (%)",
            title="Voltage Violation Rate by Hour")
    ax6.set_xticks(x); ax6.legend(fontsize=8); ax6.grid(alpha=0.3, axis="y")

    plt.savefig("der_24hour_simulation.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: der_24hour_simulation.png")

# ─── Plot 2: Frequency Response ───────────────────────────────────────────────
def plot_freq():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Frequency Response: Grid-Forming (GFM) vs Grid-Following (GFL) Inverters\n"
        "Section 3.3 — Frequency Stability & Inertia Decline",
        fontsize=12, fontweight="bold")

    events = [
        (59.1, "Under-Frequency Event (59.1 Hz start)"),
        (60.9, "Over-Frequency Event (60.9 Hz start)"),
    ]
    for ax, (f0, title) in zip(axes, events):
        t_gfl, f_gfl = sim_freq_event(InverterType.GRID_FOLLOWING, f0)
        t_gfm, f_gfm = sim_freq_event(InverterType.GRID_FORMING,   f0)
        ax.plot(t_gfl, f_gfl, "r-",  lw=2.0, label="GFL — Grid-Following (Legacy)")
        ax.plot(t_gfm, f_gfm, "g-",  lw=2.0, label="GFM — Grid-Forming (Virtual Inertia)")
        ax.axhline(F_NOM, color="k",      ls="--", lw=1, alpha=0.5, label=f"Nominal {F_NOM} Hz")
        ax.axhline(F_MIN, color="orange", ls=":",  lw=1, label=f"Min {F_MIN} Hz")
        ax.axhline(F_MAX, color="orange", ls="-.", lw=1, label=f"Max {F_MAX} Hz")
        ax.fill_between(t_gfl, F_MIN, F_MAX, alpha=0.07, color="green")
        ax.set(xlabel="Time (s)", ylabel="Frequency (Hz)", title=title)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("frequency_response.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: frequency_response.png")

# ─── Plot 3: Test-Suite Statistical Analysis ──────────────────────────────────
def plot_stats(df: pd.DataFrame):
    leg = df[df["mode"] == GridMode.LEGACY.value]
    mod = df[df["mode"] == GridMode.MODERN_DER.value]
    ders = sorted(df["der_penetration"].unique())
    W = 0.35

    fig = plt.figure(figsize=(22, 18))
    fig.suptitle(
        f"Test-Suite Statistical Analysis — {len(df):,} Records\n"
        "Legacy Grid vs Modern DER-Integrated Grid",
        fontsize=13, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.34)

    # 1 Voltage Histogram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(leg["voltage_pu"], bins=70, alpha=0.60, color="red",   density=True, label="Legacy")
    ax1.hist(mod["voltage_pu"], bins=70, alpha=0.60, color="green", density=True, label="Modern DER")
    ax1.axvline(V_MAX_PU, color="orange", ls="--", lw=1.5)
    ax1.axvline(V_MIN_PU, color="orange", ls="--", lw=1.5, label="IEEE Limits")
    ax1.set(xlabel="Voltage (pu)", ylabel="Density", title="Voltage Distribution")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    # 2 Voltage Violation by DER penetration
    ax2 = fig.add_subplot(gs[0, 1])
    lv = [leg[leg["der_penetration"]==d]["v_violation"].mean()*100 for d in ders]
    mv = [mod[mod["der_penetration"]==d]["v_violation"].mean()*100 for d in ders]
    x  = np.arange(len(ders))
    ax2.bar(x-W/2, lv, W, color="red",   alpha=0.75, label="Legacy")
    ax2.bar(x+W/2, mv, W, color="green", alpha=0.75, label="Modern DER")
    ax2.set(xlabel="DER Penetration", ylabel="Violation Rate (%)",
            title="Voltage Violations vs DER Penetration")
    ax2.set_xticks(x); ax2.set_xticklabels([f"{int(d*100)}%" for d in ders])
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3, axis="y")

    # 3 Reverse Power Flow
    ax3 = fig.add_subplot(gs[0, 2])
    lr = [leg[leg["der_penetration"]==d]["reverse_flow"].mean()*100 for d in ders]
    mr = [mod[mod["der_penetration"]==d]["reverse_flow"].mean()*100 for d in ders]
    ax3.plot([f"{int(d*100)}%" for d in ders], lr, "r-o", lw=2, label="Legacy")
    ax3.plot([f"{int(d*100)}%" for d in ders], mr, "g-s", lw=2, label="Modern DER")
    ax3.set(xlabel="DER Penetration", ylabel="Rate (%)",
            title="Reverse Power Flow Rate")
    ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

    # 4 Stability Score Boxplot by Season
    ax4 = fig.add_subplot(gs[1, 0])
    seasons = list(SEASONS.keys())
    pos_l = np.arange(len(seasons)) * 3
    pos_m = pos_l + 1
    bp1 = ax4.boxplot([leg[leg["season"]==s]["stability_score"].values for s in seasons],
                      positions=pos_l, widths=0.7, patch_artist=True,
                      boxprops=dict(facecolor="salmon", alpha=0.7))
    bp2 = ax4.boxplot([mod[mod["season"]==s]["stability_score"].values for s in seasons],
                      positions=pos_m, widths=0.7, patch_artist=True,
                      boxprops=dict(facecolor="lightgreen", alpha=0.7))
    ax4.set_xticks(pos_l + 0.5)
    ax4.set_xticklabels(seasons, fontsize=8)
    ax4.set(ylabel="Stability Score", title="Grid Stability Score by Season")
    ax4.legend([bp1["boxes"][0], bp2["boxes"][0]], ["Legacy", "Modern DER"], fontsize=8)
    ax4.grid(alpha=0.3, axis="y")

    # 5 Renewable Utilisation
    ax5 = fig.add_subplot(gs[1, 1])
    lu = [leg[leg["der_penetration"]==d]["renewables_util"].mean()*100 for d in ders]
    mu = [mod[mod["der_penetration"]==d]["renewables_util"].mean()*100 for d in ders]
    ax5.bar(x-W/2, lu, W, color="red",   alpha=0.75, label="Legacy")
    ax5.bar(x+W/2, mu, W, color="green", alpha=0.75, label="Modern DER")
    ax5.set(xlabel="DER Penetration", ylabel="Utilisation (%)",
            title="Renewable Energy Utilisation Rate")
    ax5.set_xticks(x); ax5.set_xticklabels([f"{int(d*100)}%" for d in ders])
    ax5.legend(fontsize=8); ax5.grid(alpha=0.3, axis="y")

    # 6 Hosting Capacity
    ax6 = fig.add_subplot(gs[1, 2])
    lhc = [leg[leg["der_penetration"]==d]["hosting_cap_kw"].mean() for d in ders]
    mhc = [mod[mod["der_penetration"]==d]["hosting_cap_kw"].mean() for d in ders]
    xlabels = [f"{int(d*100)}%" for d in ders]
    ax6.fill_between(xlabels, lhc, alpha=0.20, color="red")
    ax6.fill_between(xlabels, mhc, alpha=0.20, color="green")
    ax6.plot(xlabels, lhc, "r-o", lw=2, label="Legacy")
    ax6.plot(xlabels, mhc, "g-s", lw=2, label="Modern DER")
    ax6.set(xlabel="DER Penetration", ylabel="Hosting Capacity (kW)",
            title="Hosting Capacity — DERMS Impact")
    ax6.legend(fontsize=8); ax6.grid(alpha=0.3)

    # 7 Summary Table
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis("off")
    rows_tbl = [
        ["Voltage Violation Rate",
         f"{leg['v_violation'].mean()*100:.2f}%",
         f"{mod['v_violation'].mean()*100:.2f}%",
         f"{(1-mod['v_violation'].mean()/max(leg['v_violation'].mean(),1e-9))*100:.1f}% reduction"],
        ["Overvoltage Rate",
         f"{leg['overvoltage'].mean()*100:.2f}%",
         f"{mod['overvoltage'].mean()*100:.2f}%",
         f"{(1-mod['overvoltage'].mean()/max(leg['overvoltage'].mean(),1e-9))*100:.1f}% reduction"],
        ["Mean Voltage (pu)",
         f"{leg['voltage_pu'].mean():.4f}",
         f"{mod['voltage_pu'].mean():.4f}",
         f"Δ = {mod['voltage_pu'].mean()-leg['voltage_pu'].mean():.4f}"],
        ["Voltage Std Dev (pu)",
         f"{leg['voltage_pu'].std():.4f}",
         f"{mod['voltage_pu'].std():.4f}",
         f"{(1-mod['voltage_pu'].std()/leg['voltage_pu'].std())*100:.1f}% tighter"],
        ["Grid Stability Score",
         f"{leg['stability_score'].mean():.4f}",
         f"{mod['stability_score'].mean():.4f}",
         f"+{(mod['stability_score'].mean()-leg['stability_score'].mean())*100:.1f}% improvement"],
        ["Avg Hosting Capacity (kW)",
         f"{leg['hosting_cap_kw'].mean():.0f}",
         f"{mod['hosting_cap_kw'].mean():.0f}",
         f"+{(mod['hosting_cap_kw'].mean()/leg['hosting_cap_kw'].mean()-1)*100:.1f}% higher"],
        ["Freq Violation Rate",
         f"{leg['f_violation'].mean()*100:.2f}%",
         f"{mod['f_violation'].mean()*100:.2f}%",
         f"{(1-mod['f_violation'].mean()/max(leg['f_violation'].mean(),1e-9))*100:.1f}% reduction"],
        ["Total Records",
         f"{len(leg):,}", f"{len(mod):,}", f"Total: {len(df):,}"],
    ]
    cols = ["Metric", "Legacy Grid", "Modern DER", "Improvement"]
    tbl  = ax7.table(cellText=rows_tbl, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.85)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1565C0"); cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f5f5f5")
        if c == 1 and r > 0: cell.set_facecolor("#ffe0e0")
        if c == 2 and r > 0: cell.set_facecolor("#e0ffe0")
        if c == 3 and r > 0: cell.set_facecolor("#fffde0")
    ax7.set_title("Summary Statistics Table", fontsize=11, fontweight="bold", pad=18)

    plt.savefig("der_test_suite_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: der_test_suite_analysis.png")

# ─── Plot 4: Heatmap ─────────────────────────────────────────────────────────
def plot_heatmap(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Voltage Violation Rate Heatmap — Season × DER Penetration\n"
                 "Section 4.2: DERMS Hosting Capacity Analysis",
                 fontsize=12, fontweight="bold")

    for ax, mode_val, title, cmap in zip(
        axes,
        [GridMode.LEGACY.value, GridMode.MODERN_DER.value],
        ["Legacy Grid", "Modern DER (DERMS Enabled)"],
        ["Reds", "Greens"]
    ):
        sub   = df[df["mode"] == mode_val]
        pivot = sub.pivot_table(values="v_violation", index="season",
                                columns="scenario", aggfunc="mean") * 100
        im = ax.imshow(pivot.values, cmap=cmap, aspect="auto", vmin=0, vmax=30)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([c.replace("_", "\n") for c in pivot.columns], fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="white" if val > 15 else "black")
        plt.colorbar(im, ax=ax, label="Violation Rate (%)")
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig("hosting_capacity_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: hosting_capacity_heatmap.png")

# ─── Statistical Report ───────────────────────────────────────────────────────
def print_stats(df: pd.DataFrame):
    leg = df[df["mode"] == GridMode.LEGACY.value]
    mod = df[df["mode"] == GridMode.MODERN_DER.value]

    sep = "=" * 65
    print(f"\n{sep}")
    print("  STATISTICAL ANALYSIS REPORT")
    print(sep)
    print(f"  Total records : {len(df):,}  |  Legacy: {len(leg):,}  |  Modern: {len(mod):,}")

    t_v, p_v = stats.ttest_ind(leg["voltage_pu"], mod["voltage_pu"])
    t_s, p_s = stats.ttest_ind(leg["stability_score"], mod["stability_score"])

    print(f"\n{'Metric':<30} {'Legacy':>12} {'Modern DER':>12} {'∆ / Δ':>14}")
    print("-"*65)
    metrics = [
        ("Mean Voltage (pu)",      leg["voltage_pu"].mean(),       mod["voltage_pu"].mean(),        None),
        ("Voltage Std Dev (pu)",   leg["voltage_pu"].std(),        mod["voltage_pu"].std(),         None),
        ("Voltage Violation (%)",  leg["v_violation"].mean()*100,  mod["v_violation"].mean()*100,   None),
        ("Overvoltage (%)",        leg["overvoltage"].mean()*100,  mod["overvoltage"].mean()*100,   None),
        ("Undervoltage (%)",       leg["undervoltage"].mean()*100, mod["undervoltage"].mean()*100,  None),
        ("Stability Score",        leg["stability_score"].mean(),  mod["stability_score"].mean(),   None),
        ("Hosting Cap (kW)",       leg["hosting_cap_kw"].mean(),   mod["hosting_cap_kw"].mean(),    None),
        ("Renewables Util (%)",    leg["renewables_util"].mean()*100, mod["renewables_util"].mean()*100, None),
        ("Freq Violation (%)",     leg["f_violation"].mean()*100,  mod["f_violation"].mean()*100,   None),
    ]
    for name, lv, mv, _ in metrics:
        delta = mv - lv
        print(f"  {name:<28} {lv:>12.4f} {mv:>12.4f} {delta:>+14.4f}")

    print(f"\n  T-test (voltage_pu):       t={t_v:.3f}  p={p_v:.2e}  "
          f"{'sig*' if p_v < 0.05 else 'n.s.'}")
    print(f"  T-test (stability_score):  t={t_s:.3f}  p={p_s:.2e}  "
          f"{'sig*' if p_s < 0.05 else 'n.s.'}")

    print(f"\n  Voltage violation reduction : "
          f"{(1-mod['v_violation'].mean()/max(leg['v_violation'].mean(),1e-9))*100:.1f}%")
    print(f"  Hosting capacity increase   : "
          f"{(mod['hosting_cap_kw'].mean()/leg['hosting_cap_kw'].mean()-1)*100:.1f}%")
    print(f"  Stability score improvement : "
          f"{(mod['stability_score'].mean()-leg['stability_score'].mean())*100:.1f} pp")
    print(sep + "\n")

# ─── Entry Point ──────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  DER GRID INTEGRATION SIMULATION")
    print("  'Integrating DERs into Legacy Grid Infrastructure'")
    print("  Author: Bhanu Singh | retailtechnologist@outlook.com")
    print("=" * 65)

    print("\n[1/5] 24-hour simulations …")
    df_leg = run_24h(GridMode.LEGACY)
    df_mod = run_24h(GridMode.MODERN_DER)

    print("[2/5] Generating 10,000-record test dataset …")
    t0 = time.time()
    df_test = generate_test_dataset(10_000)
    print(f"      {len(df_test):,} records in {time.time()-t0:.1f}s")
    df_test.to_csv("der_test_dataset.csv", index=False)
    print("      Saved: der_test_dataset.csv")

    print("[3/5] Statistical report …")
    print_stats(df_test)

    print("[4/5] Generating plots …")
    plot_24h(df_leg, df_mod)
    plot_freq()
    plot_stats(df_test)
    plot_heatmap(df_test)

    print("[5/5] Complete.")
    print("\nOutput files:")
    print("  der_test_dataset.csv          — 10,000 test records")
    print("  der_24hour_simulation.png     — 24-hour grid comparison")
    print("  frequency_response.png        — GFL vs GFM inverter response")
    print("  der_test_suite_analysis.png   — statistical analysis charts")
    print("  hosting_capacity_heatmap.png  — violation heatmap by season & DER level")
    return df_test, df_leg, df_mod


if __name__ == "__main__":
    df_test, df_leg, df_mod = main()
