# EV_planner_clean.py
# Electric Vehicle Power & Energy Planner
# Clean build: ASCII-only, safe triple-quoted strings, no stray parentheses.

import math
from dataclasses import dataclass
from typing import Dict, List, Any
import json
import io

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import altair as alt

st.set_page_config(page_title="Electric Vehicle Power & Energy Planner", page_icon="⚡", layout="wide")

# ---------- Helpers ----------
def kmh_to_ms(v_kmh: float) -> float:
    return v_kmh / 3.6

def wheel_rpm(v_ms: float, r_m: float) -> float:
    if r_m <= 0:
        return 0.0
    return (v_ms / (2 * math.pi * r_m)) * 60.0

@dataclass
class ScenarioInput:
    name: str
    speed_kmh: float
    time_s: float
    grade_pct: float

@dataclass
class ScenarioResult:
    name: str
    v_ms: float
    a_ms2: float
    rpm: float
    F_accel_N: float
    R_roll_N: float
    R_drag_N: float
    R_grade_N: float
    F_resist_N: float
    F_total_N: float
    T_accel_Nm: float
    T_resist_Nm: float
    T_total_Nm: float
    P_accel_kW: float
    P_resist_kW: float
    P_total_kW: float

def scenario_calc(sc: ScenarioInput, m_kg: float, r_m: float, A_m2: float, rho_air: float, c_d: float, c_r: float, g: float) -> ScenarioResult:
    v = kmh_to_ms(sc.speed_kmh)
    a = v / sc.time_s if sc.time_s > 0 else 0.0

    F_accel = m_kg * a
    R_roll = c_r * m_kg * g
    R_drag = 0.5 * rho_air * A_m2 * c_d * (v ** 2)
    R_grade = m_kg * g * (sc.grade_pct / 100.0)

    F_resist = R_roll + R_drag + R_grade
    F_total = F_accel + F_resist

    T_accel = F_accel * r_m
    T_resist = F_resist * r_m
    T_total = F_total * r_m

    P_accel = (F_accel * v) / 1000.0
    P_resist = (F_resist * v) / 1000.0
    P_total = P_accel + P_resist

    rpm = wheel_rpm(v, r_m)

    return ScenarioResult(
        name=sc.name, v_ms=v, a_ms2=a, rpm=rpm,
        F_accel_N=F_accel, R_roll_N=R_roll, R_drag_N=R_drag, R_grade_N=R_grade,
        F_resist_N=F_resist, F_total_N=F_total,
        T_accel_Nm=T_accel, T_resist_Nm=T_resist, T_total_Nm=T_total,
        P_accel_kW=P_accel, P_resist_kW=P_resist, P_total_kW=P_total
    )

def battery_capacity_kwh(wh_per_km: float, range_km: float, dod_pct: float) -> float:
    dod = max(min(dod_pct / 100.0, 0.9999), 1e-6)
    return (wh_per_km * range_km) / (1000.0 * dod)

def per_motor(value: float, motors: int, mismatch: float, cv_eff: float) -> float:
    motors = max(1, int(motors))
    chain_eff = max(1e-6, mismatch * cv_eff)
    return (value / motors) / chain_eff

def safe_positive(x: float) -> float:
    return max(0.0, float(x))

# ---------- Presets ----------
PRESETS: Dict[str, Dict[str, Any]] = {
    "4-Seater Example": {
        "kerb_kg": 700, "payload_kg": 250, "tyre_radius_mm": 253, "frontal_area_m2": 1.914,
        "rho_air": 1.2, "g": 9.81, "c_r": 0.015, "c_d": 0.30,
        "motor_mismatch": 0.95, "cv_eff": 1.0, "overload_capacity": 2.0,
        "max_speed_kmh": 105, "min_grade_pct": 0.5, "time_to_max_s": 20,
        "avg_speed_kmh": 60, "avg_grade_pct": 8, "time_to_avg_s": 30,
        "max_grade_pct": 30, "grade_speed_kmh": 0.1, "time_to_grade_s": 10,
        "wh_per_km": 150, "range_km": 160, "dod_pct": 80, "num_motors": 2
    },
    "Urban": {
        "kerb_kg": 750, "payload_kg": 200, "tyre_radius_mm": 250, "frontal_area_m2": 2.0,
        "rho_air": 1.225, "g": 9.81, "c_r": 0.014, "c_d": 0.32,
        "motor_mismatch": 0.95, "cv_eff": 0.98, "overload_capacity": 2.0,
        "max_speed_kmh": 80, "min_grade_pct": 1.0, "time_to_max_s": 18,
        "avg_speed_kmh": 45, "avg_grade_pct": 4, "time_to_avg_s": 12,
        "max_grade_pct": 15, "grade_speed_kmh": 5, "time_to_grade_s": 8,
        "wh_per_km": 130, "range_km": 180, "dod_pct": 85, "num_motors": 2
    },
    "Highway": {
        "kerb_kg": 820, "payload_kg": 200, "tyre_radius_mm": 255, "frontal_area_m2": 2.1,
        "rho_air": 1.2, "g": 9.81, "c_r": 0.015, "c_d": 0.28,
        "motor_mismatch": 0.96, "cv_eff": 0.99, "overload_capacity": 2.0,
        "max_speed_kmh": 120, "min_grade_pct": 0.5, "time_to_max_s": 25,
        "avg_speed_kmh": 100, "avg_grade_pct": 2, "time_to_avg_s": 20,
        "max_grade_pct": 10, "grade_speed_kmh": 40, "time_to_grade_s": 15,
        "wh_per_km": 170, "range_km": 220, "dod_pct": 85, "num_motors": 2
    },
    "Hill-Climb": {
        "kerb_kg": 850, "payload_kg": 150, "tyre_radius_mm": 260, "frontal_area_m2": 2.0,
        "rho_air": 1.2, "g": 9.81, "c_r": 0.016, "c_d": 0.35,
        "motor_mismatch": 0.94, "cv_eff": 0.98, "overload_capacity": 2.5,
        "max_speed_kmh": 90, "min_grade_pct": 3, "time_to_max_s": 30,
        "avg_speed_kmh": 50, "avg_grade_pct": 10, "time_to_avg_s": 20,
        "max_grade_pct": 30, "grade_speed_kmh": 10, "time_to_grade_s": 12,
        "wh_per_km": 180, "range_km": 150, "dod_pct": 80, "num_motors": 2
    },
}

# ---------- Sidebar ----------
st.sidebar.title("Inputs")
with st.sidebar.expander("Presets & Save/Load", expanded=True):
    cA, cB, cC, cD = st.columns(4)
    for i, name in enumerate(PRESETS.keys()):
        if [cA, cB, cC, cD][i].button(name):
            st.session_state.update(PRESETS[name])
    d1, d2 = st.columns(2)
    if d1.button("Reset to Defaults"):
        st.session_state.clear()
    uploaded_json = d2.file_uploader("Load Inputs (JSON)", type="json", label_visibility="collapsed")
    if uploaded_json:
        try:
            st.session_state.update(json.load(io.StringIO(uploaded_json.getvalue().decode())))
            st.success("Inputs loaded from JSON.")
        except Exception as e:
            st.error(f"Failed to load JSON: {e}")

with st.sidebar.form("inputs"):
    st.markdown("**Vehicle & Environmental Parameters**")
    kerb_kg = st.number_input("Unladen (Kerb) Weight [kg]", min_value=0.0, value=float(st.session_state.get("kerb_kg", 700.0)), step=10.0)
    payload_kg = st.number_input("Payload [kg]", min_value=0.0, value=float(st.session_state.get("payload_kg", 250.0)), step=10.0)
    tyre_radius_mm = st.number_input("Tyre Rolling Radius [mm]", min_value=1.0, value=float(st.session_state.get("tyre_radius_mm", 253.0)), step=1.0)
    frontal_area_m2 = st.number_input("Frontal Area A [m^2]", min_value=0.1, value=float(st.session_state.get("frontal_area_m2", 1.914)), step=0.01)
    rho_air = st.number_input("Air Density rho [kg/m^3]", min_value=0.5, value=float(st.session_state.get("rho_air", 1.2)), step=0.05)
    g = st.number_input("Gravity g [m/s^2]", min_value=9.0, value=float(st.session_state.get("g", 9.81)), step=0.01)
    st.markdown("---")

    st.markdown("**Vehicle Resistance Coefficients**")
    c_r = st.number_input("Rolling Resistance c_r [-]", min_value=0.0, value=float(st.session_state.get("c_r", 0.015)), step=0.001)
    c_d = st.number_input("Aerodynamic Drag c_d [-]", min_value=0.0, value=float(st.session_state.get("c_d", 0.30)), step=0.01)
    st.markdown("---")

    st.markdown("**Powertrain Efficiency Factors**")
    motor_mismatch = st.number_input("Motor Mismatch Factor [-]", min_value=0.01, value=float(st.session_state.get("motor_mismatch", 0.95)), step=0.01)
    cv_eff = st.number_input("CV Joint Efficiency [-]", min_value=0.01, value=float(st.session_state.get("cv_eff", 1.00)), step=0.01)
    overload_capacity = st.number_input("Overload Capacity (Peak/Rated) [-]", min_value=1.0, value=float(st.session_state.get("overload_capacity", 2.0)), step=0.1)
    st.markdown("---")

    st.markdown("**Driving Scenarios**")
    max_speed_kmh = st.number_input("Max Speed [km/h]", min_value=0.0, value=float(st.session_state.get("max_speed_kmh", 105.0)), step=1.0)
    min_grade_pct = st.number_input("Min Grade @ Max Speed [%]", min_value=0.0, value=float(st.session_state.get("min_grade_pct", 0.5)), step=0.1)
    time_to_max_s = st.number_input("Time to Max Speed [s]", min_value=0.01, value=float(st.session_state.get("time_to_max_s", 20.0)), step=1.0)

    avg_speed_kmh = st.number_input("Average Speed [km/h]", min_value=0.0, value=float(st.session_state.get("avg_speed_kmh", 60.0)), step=1.0)
    avg_grade_pct = st.number_input("Average Grade [%]", min_value=0.0, value=float(st.session_state.get("avg_grade_pct", 8.0)), step=0.5)
    time_to_avg_s = st.number_input("Time to Avg Speed [s]", min_value=0.01, value=float(st.session_state.get("time_to_avg_s", 30.0)), step=1.0)

    max_grade_pct = st.number_input("Max Gradeability [%]", min_value=0.0, value=float(st.session_state.get("max_grade_pct", 30.0)), step=0.5)
    grade_speed_kmh = st.number_input("Speed @ Max Grade [km/h]", min_value=0.0, value=float(st.session_state.get("grade_speed_kmh", 0.1)), step=0.1)
    time_to_grade_s = st.number_input("Time to Grade Speed [s]", min_value=0.01, value=float(st.session_state.get("time_to_grade_s", 10.0)), step=1.0)
    st.markdown("---")

    st.markdown("**Battery Calculation Parameters**")
    wh_per_km = st.number_input("Energy Consumption [Wh/km]", min_value=1.0, value=float(st.session_state.get("wh_per_km", 150.0)), step=5.0)
    range_km = st.number_input("Range Target [km]", min_value=1.0, value=float(st.session_state.get("range_km", 160.0)), step=5.0)
    dod_pct = st.number_input("Depth of Discharge DoD [%]", min_value=1.0, max_value=99.0, value=float(st.session_state.get("dod_pct", 80.0)), step=1.0)
    num_motors = st.number_input("Number of Motors", min_value=1, value=int(st.session_state.get("num_motors", 2)), step=1)

    apply = st.form_submit_button("Apply Settings")

# Save JSON
cur_inputs = {
    "kerb_kg": kerb_kg, "payload_kg": payload_kg, "tyre_radius_mm": tyre_radius_mm,
    "frontal_area_m2": frontal_area_m2, "rho_air": rho_air, "g": g,
    "c_r": c_r, "c_d": c_d, "motor_mismatch": motor_mismatch, "cv_eff": cv_eff,
    "overload_capacity": overload_capacity, "max_speed_kmh": max_speed_kmh, "min_grade_pct": min_grade_pct,
    "time_to_max_s": time_to_max_s, "avg_speed_kmh": avg_speed_kmh, "avg_grade_pct": avg_grade_pct,
    "time_to_avg_s": time_to_avg_s, "max_grade_pct": max_grade_pct, "grade_speed_kmh": grade_speed_kmh,
    "time_to_grade_s": time_to_grade_s, "wh_per_km": wh_per_km, "range_km": range_km,
    "dod_pct": dod_pct, "num_motors": num_motors
}

with st.sidebar.expander("Export current inputs", expanded=False):
    st.download_button(
        "Download Inputs (JSON)",
        data=json.dumps(cur_inputs, default=float, indent=2).encode(),
        file_name="ev_inputs.json",
        mime="application/json"
    )

# ---------- Validation ----------
errors = []
if tyre_radius_mm <= 0:
    errors.append("Tyre rolling radius must be > 0 mm.")
if time_to_max_s <= 0 or time_to_avg_s <= 0 or time_to_grade_s <= 0:
    errors.append("Times to reach speed must be > 0 s for all scenarios.")
if not (0 < dod_pct <= 99):
    errors.append("Depth of Discharge (DoD) must be between 1 and 99%.")
for val, name in [(c_r, "c_r"), (c_d, "c_d"), (motor_mismatch, "motor mismatch"), (cv_eff, "CV efficiency")]:
    if val <= 0 or val > 1.5:
        errors.append(f"Check {name}: expected a positive value (typically <= 1.2).")

if errors:
    for e in errors:
        st.warning(e)
    st.stop()

# ---------- Derived ----------
m_kg = safe_positive(kerb_kg + payload_kg)
r_m = safe_positive(tyre_radius_mm / 1000.0)

# ---------- Title ----------
st.title("Electric Vehicle Power & Energy Planner")
st.markdown("""
**What this tool does**  
This tool calculates the motor and battery specifications your EV needs using a scenario-based approach.  
It simulates acceleration, cruising, and steep hill-climb cases to determine the worst-case torque and worst-case power requirements.  
It then adjusts results for the number of motors, drivetrain efficiency, and battery energy needed to meet your target range.  
""")

# ---------- Compute scenarios ----------
scenarios = [
    ScenarioInput("0 to Max Speed @ Min Grade", max_speed_kmh, time_to_max_s, min_grade_pct),
    ScenarioInput("0 to Avg Speed @ Avg Grade", avg_speed_kmh, time_to_avg_s, avg_grade_pct),
    ScenarioInput("Grade Speed @ Max Grade",  grade_speed_kmh, time_to_grade_s, max_grade_pct),
]
results: List[ScenarioResult] = [scenario_calc(s, m_kg, r_m, frontal_area_m2, rho_air, c_d, c_r, g) for s in scenarios]

def to_row(sr: ScenarioResult) -> Dict[str, float]:
    return {
        "Scenario": sr.name,
        "Speed (km/h)": round(sr.v_ms * 3.6, 3),
        "Accel (m/s^2)": round(sr.a_ms2, 4),
        "Wheel RPM": round(sr.rpm, 2),
        "Torque Accel (Nm)": round(sr.T_accel_Nm, 2),
        "Torque Resistive (Nm)": round(sr.T_resist_Nm, 2),
        "Torque Total (Nm)": round(sr.T_total_Nm, 2),
        "Power Accel (kW)": round(sr.P_accel_kW, 3),
        "Power Resist (kW)": round(sr.P_resist_kW, 3),
        "Power Total (kW)": round(sr.P_total_kW, 3),
    }
summary_df = pd.DataFrame([to_row(r) for r in results])

# ---------- KPIs ----------
peak_total_torque = max(r.T_total_Nm for r in results)
peak_total_power  = max(r.P_total_kW for r in results)
torque_per_motor_peak = per_motor(peak_total_torque, num_motors, motor_mismatch, cv_eff)
power_per_motor_peak  = per_motor(peak_total_power,  num_motors, motor_mismatch, cv_eff)
power_per_motor_rated = power_per_motor_peak / max(1e-6, overload_capacity)
driver_torque = max(results, key=lambda r: r.T_total_Nm).name
driver_power  = max(results, key=lambda r: r.P_total_kW).name

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Peak Total Torque", f"{peak_total_torque:,.0f} Nm")
k2.metric("Peak Total Power", f"{peak_total_power:,.1f} kW")
k3.metric("Per-Motor Peak Power", f"{power_per_motor_peak:,.1f} kW")
k4.metric("Per-Motor Rated Power", f"{power_per_motor_rated:,.1f} kW")
k5.metric("Battery Capacity", f"{battery_capacity_kwh(wh_per_km, range_km, dod_pct):,.1f} kWh")

with st.expander("What drives these sizing results?", expanded=False):
    st.markdown(f"""
- **Torque-driven constraint** -> **{driver_torque}**: Scenario producing the maximum **tractive torque** at the wheel, typically low-speed, high-grade operation where gravitational and rolling resistances dominate.
- **Power-driven constraint** -> **{driver_power}**: Scenario producing the maximum **mechanical power** at the wheel, typically high-speed operation where aerodynamic drag dominates.
The selected motor(s) must be capable of simultaneously satisfying both torque and power constraints. Battery capacity is independently determined from the specified energy consumption rate (Wh/km), target range, and allowable depth of discharge (DoD).
""")

# ---------- Tabs ----------
tabs = st.tabs(["Results", "Scenarios", "Charts", "Downloads", "How this was computed"])

# Results Tab
with tabs[0]:
    st.subheader("Scenario Summary")
    st.dataframe(summary_df, use_container_width=True)

# Scenarios Tab
with tabs[1]:
    st.subheader("Scenario details")
    for sr in results:
        with st.expander(f"{sr.name} — details", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.write(f"Speed: {sr.v_ms*3.6:.3f} km/h")
                st.write(f"Acceleration: {sr.a_ms2:.4f} m/s^2")
                st.write(f"Wheel RPM: {sr.rpm:.2f} rpm")
            with c2:
                st.write(f"Rolling Resistance: {sr.R_roll_N:.2f} N")
                st.write(f"Aerodynamic Drag: {sr.R_drag_N:.2f} N")
                st.write(f"Grade Resistance: {sr.R_grade_N:.2f} N")
            with c3:
                st.write(f"Total Resistive Force: {sr.F_resist_N:.2f} N")
                st.write(f"Total Force: {sr.F_total_N:.2f} N")

            st.markdown("---")
            c4, c5, c6 = st.columns(3)
            with c4:
                st.write(f"Torque (Accel): {sr.T_accel_Nm:.2f} Nm")
            with c5:
                st.write(f"Torque (Resistive): {sr.T_resist_Nm:.2f} Nm")
            with c6:
                st.write(f"Torque (Total): {sr.T_total_Nm:.2f} Nm")

            c7, c8, c9 = st.columns(3)
            with c7:
                st.write(f"Power (Accel): {sr.P_accel_kW:.3f} kW")
            with c8:
                st.write(f"Power (Resistive): {sr.P_resist_kW:.3f} kW")
            with c9:
                st.write(f"Power (Total): {sr.P_total_kW:.3f} kW")

            # Resistance split pie
            st.markdown("Resistance split at this speed")
            vals = np.array([sr.R_roll_N, sr.R_drag_N, sr.R_grade_N])
            labels = ["Rolling", "Drag", "Grade"]
            if float(vals.sum()) <= 0.0:
                vals = np.array([1.0, 0.0, 0.0])
            fig = plt.figure()
            plt.pie(vals, labels=labels, autopct="%1.1f%%", startangle=90)
            plt.axis("equal")
            st.pyplot(fig)

# Charts Tab (Altair)
with tabs[2]:
    st.subheader("Interactive Charts")
    names = [r.name for r in results]
    df_torque = pd.DataFrame({
        "Scenario": names,
        "Accel Torque (Nm)": [r.T_accel_Nm for r in results],
        "Resist Torque (Nm)": [r.T_resist_Nm for r in results],
        "Total Torque (Nm)": [r.T_total_Nm for r in results],
    })
    df_power = pd.DataFrame({
        "Scenario": names,
        "Accel Power (kW)": [r.P_accel_kW for r in results],
        "Resist Power (kW)": [r.P_resist_kW for r in results],
        "Total Power (kW)": [r.P_total_kW for r in results],
    })

    torque_long = df_torque.melt(id_vars="Scenario", var_name="Type", value_name="Nm")
    power_long  = df_power.melt(id_vars="Scenario", var_name="Type", value_name="kW")

    chart_torque = alt.Chart(torque_long).mark_bar().encode(
        x=alt.X("Scenario:N", sort=names),
        y=alt.Y("Nm:Q"),
        color="Type:N",
        tooltip=["Scenario", "Type", alt.Tooltip("Nm:Q", format=".1f")]
    ).properties(title="Torque Breakdown by Scenario")

    chart_power = alt.Chart(power_long).mark_bar().encode(
        x=alt.X("Scenario:N", sort=names),
        y=alt.Y("kW:Q"),
        color="Type:N",
        tooltip=["Scenario", "Type", alt.Tooltip("kW:Q", format=".2f")]
    ).properties(title="Power Breakdown by Scenario")

    st.altair_chart(chart_torque, use_container_width=True)
    st.altair_chart(chart_power, use_container_width=True)

# Downloads Tab
with tabs[3]:
    st.subheader("Export")
    csv_bytes = summary_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Scenario Summary (CSV)", data=csv_bytes, file_name="ev_scenarios_summary.csv", mime="text/csv")

    st.markdown("Mini Report (HTML)")
    html = f"""
<html><head><meta charset='utf-8'><title>EV Report</title></head><body>
<h2>EV Sizing Summary</h2>
<p><b>Torque-driven sizing:</b> {driver_torque} | <b>Power-driven sizing:</b> {driver_power}</p>
<ul>
  <li>Peak Total Torque: {peak_total_torque:,.0f} Nm</li>
  <li>Peak Total Power: {peak_total_power:,.1f} kW</li>
  <li>Per-Motor Peak Power: {power_per_motor_peak:,.1f} kW</li>
  <li>Per-Motor Rated Power: {power_per_motor_rated:,.1f} kW</li>
  <li>Battery Capacity: {battery_capacity_kwh(wh_per_km, range_km, dod_pct):,.1f} kWh</li>
</ul>
<h3>Inputs</h3>
<pre>{json.dumps(cur_inputs, indent=2)}</pre>
</body></html>
"""
    st.download_button("Download Mini Report (HTML)", data=html.encode("utf-8"), file_name="ev_report.html", mime="text/html")


# --- How this was computed (notation-rich) ---
with tabs[4]:
    st.subheader("How this was computed")

    st.markdown("### 1) Vehicle mass")
    st.latex(r"m = m_{\mathrm{kerb}} + m_{\mathrm{payload}}")

    st.markdown("### 2) Unit conversions & kinematics")
    st.latex(r"v\,[\mathrm{m/s}] = \frac{v_{\mathrm{km/h}}}{3.6}")
    st.latex(r"a = \frac{v}{t}")
    st.latex(r"\mathrm{RPM} = \frac{v}{2\pi r}\times 60")
    st.markdown("Here, $r$ is the tyre rolling radius in meters (the app converts mm to m).")

    st.markdown("### 3) Resistive forces")
    st.latex(r"R_r = c_r \, m \, g")
    st.latex(r"R_d = \tfrac{1}{2}\,\rho\,A\,c_d\,v^2")
    st.latex(r"R_g = m\,g \cdot \frac{\mathrm{grade}[\%]}{100}")
    st.latex(r"F_{\mathrm{resist}} = R_r + R_d + R_g")

    st.markdown("### 4) Tractive force & wheel torque")
    st.latex(r"F_{\mathrm{accel}} = m \cdot a")
    st.latex(r"F_{\mathrm{total}} = F_{\mathrm{accel}} + F_{\mathrm{resist}}")
    st.latex(r"T_{\mathrm{accel}} = F_{\mathrm{accel}} \cdot r")
    st.latex(r"T_{\mathrm{resist}} = F_{\mathrm{resist}} \cdot r")
    st.latex(r"T_{\mathrm{total}} = F_{\mathrm{total}} \cdot r")

    st.markdown("### 5) Power at the wheels")
    st.latex(r"P_{\mathrm{accel}} = \frac{F_{\mathrm{accel}} \cdot v}{1000} \quad [\mathrm{kW}]")
    st.latex(r"P_{\mathrm{resist}} = \frac{F_{\mathrm{resist}} \cdot v}{1000} \quad [\mathrm{kW}]")
    st.latex(r"P_{\mathrm{total}} = P_{\mathrm{accel}} + P_{\mathrm{resist}}")

    st.markdown("### 6) Per-motor allocation & ratings")
    st.latex(r"\text{Per-motor peak} \approx \frac{\text{Peak total}}{N \cdot \eta}")
    st.markdown(r"where $N$ is the number of motors and $\eta=\text{mismatch} \times \text{CV efficiency} \le 1$.")
    st.latex(r"P_{\mathrm{rated}} = \frac{P_{\mathrm{peak}}}{\gamma}")
    st.markdown(r"where $\gamma$ is the overload ratio (Peak/Rated).")

    st.markdown("### 7) Battery sizing (energy basis)")
    st.latex(r"\text{Battery [kWh]} = \frac{E_{\mathrm{km}} \times R}{1000 \times f}")
    st.markdown(r"$E_{\mathrm{km}}$ is energy consumption (Wh/km), $R$ is target range (km), and $f$ is the usable fraction (DoD).")

    st.markdown("### 8) Notation & terms")
    st.markdown(r"""
**Kinematics & geometry**

| Symbol | Full form | Units | Meaning |
|---|---|---|---|
| $m$ | Vehicle mass | kg | $m = m_{\mathrm{kerb}} + m_{\mathrm{payload}}$ |
| $v$ | Vehicle speed | m/s | $v = v_{\mathrm{km/h}}/3.6$ |
| $a$ | Acceleration | m/s^2 | $a=v/t$ for 0 to $v$ in time $t$ |
| $t$ | Time to target speed | s | Scenario input |
| $r$ | Tyre rolling radius | m | Input mm converted to m |
| $\mathrm{RPM}$ | Wheel speed | rev/min | $\mathrm{RPM}=\frac{v}{2\pi r}\,60$ |

**Resistance coefficients & constants**

| Symbol | Full form | Units | Meaning |
|---|---|---|---|
| $c_r$ | Rolling resistance coef. | - | Tyre/road rolling losses |
| $c_d$ | Aerodynamic drag coef. | - | Drag coefficient |
| $\rho$ | Air density | kg/m^3 | Depends on altitude/temperature |
| $A$ | Frontal area | m^2 | Projected frontal area |
| $g$ | Gravity | m/s^2 | ~9.81 m/s^2 |
| $\mathrm{grade}[\%]$ | Road grade | % | Slope (rise/run x 100) |

**Forces, torques, power**

| Symbol | Full form | Units | Meaning |
|---|---|---|---|
| $R_r$ | Rolling resistance force | N | $R_r=c_r m g$ |
| $R_d$ | Aerodynamic drag force | N | $R_d=\frac{1}{2}\rho A c_d v^2$ |
| $R_g$ | Grade resistance | N | $R_g=m g\,\frac{\mathrm{grade}[\%]}{100}$ |
| $F_{\mathrm{resist}}$ | Total resistive force | N | $R_r+R_d+R_g$ |
| $F_{\mathrm{accel}}$ | Acceleration force | N | $m a$ |
| $F_{\mathrm{total}}$ | Total tractive force | N | $F_{\mathrm{accel}}+F_{\mathrm{resist}}$ |
| $T_{\mathrm{accel}}$ | Accel torque | Nm | $F_{\mathrm{accel}} r$ |
| $T_{\mathrm{resist}}$ | Resistive torque | Nm | $F_{\mathrm{resist}} r$ |
| $T_{\mathrm{total}}$ | Total torque | Nm | $F_{\mathrm{total}} r$ |
| $P_{\mathrm{accel}}$ | Accel power | kW | $F_{\mathrm{accel}} v/1000$ |
| $P_{\mathrm{resist}}$ | Resistive power | kW | $F_{\mathrm{resist}} v/1000$ |
| $P_{\mathrm{total}}$ | Total power | kW | $P_{\mathrm{accel}}+P_{\mathrm{resist}}$ |

**Battery & motor sizing**

| Symbol | Full form | Units | Meaning |
|---|---|---|---|
| $E_{\mathrm{km}}$ | Energy per km | Wh/km | Used for range sizing |
| $R$ | Range | km | Target driving range |
| $f$ | DoD fraction | - | Usable fraction (e.g., 0.8 for 80%) |
| $N$ | Number of motors | - | Load sharing |
| $\eta$ | Driveline efficiency | - | mismatch x CV efficiency |
| $\gamma$ | Overload ratio | - | Peak/Rated (Rated = Peak/\gamma) |
""")
