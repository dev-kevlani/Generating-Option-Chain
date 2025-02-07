"""
Greeks Impact Analysis using QuantLib in Python

This script analyzes the sensitivity of option Greeks (Delta, Gamma, Theta,
Vega, and Rho) for a European call option as key inputs change, such as the
underlying price, time to expiry, and the risk-free rate.

Results:
  - Plots showing how Delta, Gamma, Theta, Vega, and Rho vary with changes in underlying price.
  - Plots showing the impact of time to expiry on the Greeks.
  - An example analysis of the effects of changes in the risk-free rate.

Requirements:
  - QuantLib (QuantLib-Python)
  - numpy
  - matplotlib

Usage:
  python greeks_impact_analysis.py
"""

import QuantLib as ql
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Global parameters
strike = 100.0                # Fixed option strike price
vol = 0.20                    # Constant volatility (20%)
dividend_yield = 0.0          # Dividend yield
risk_free_rate = 0.01         # Base risk free rate (1%)
day_count = ql.Actual365Fixed()
calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
evaluation_date = ql.Date.todaysDate()
ql.Settings.instance().evaluationDate = evaluation_date

def create_option(spot, strike, vol, r, q, expiry_date):
    """
    Create a European call option in QuantLib and return the option and the Black Scholes Merton process.
    """
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)
    exercise = ql.EuropeanExercise(expiry_date)
    option = ql.VanillaOption(payoff, exercise)

    # Market data handles
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
    risk_free_curve = ql.YieldTermStructureHandle(
        ql.FlatForward(evaluation_date, r, day_count)
    )
    dividend_curve = ql.YieldTermStructureHandle(
        ql.FlatForward(evaluation_date, q, day_count)
    )
    vol_curve = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(evaluation_date, calendar, vol, day_count)
    )
    bsm_process = ql.BlackScholesMertonProcess(spot_handle,
                                               dividend_curve,
                                               risk_free_curve,
                                               vol_curve)
    # Set analytic pricing engine for option
    engine = ql.AnalyticEuropeanEngine(bsm_process)
    option.setPricingEngine(engine)
    return option

def compute_greeks(spot, expiry_date, r=risk_free_rate):
    """
    Compute the Greeks for a European call option given the spot price and expiry date.
    Returns a dictionary with Delta, Gamma, Theta, Vega, and Rho.
    """
    option = create_option(spot, strike, vol, r, dividend_yield, expiry_date)
    # Ensure NPV is calculated (this will trigger the engine)
    npv = option.NPV()
    try:
        delta = option.delta()
    except Exception:
        delta = np.nan
    try:
        gamma = option.gamma()
    except Exception:
        gamma = np.nan
    try:
        theta = option.theta()
    except Exception:
        theta = np.nan
    try:
        vega = option.vega()
    except Exception:
        vega = np.nan
    try:
        rho = option.rho()
    except Exception:
        rho = np.nan

    return {"NPV": npv, "Delta": delta, "Gamma": gamma, "Theta": theta, "Vega": vega, "Rho": rho}

def plot_greeks_vs_spot():
    """
    Plot the option Greeks as a function of the underlying spot price.
    The expiry is fixed (e.g. 30 days away) and risk-free rate is constant.
    """
    days_to_expiry = 30
    expiry_date = evaluation_date + days_to_expiry

    spot_range = np.linspace(strike * 0.7, strike * 1.3, 50)
    deltas, gammas, thetas, vegas, rhos = [], [], [], [], []

    for spot in spot_range:
        greeks = compute_greeks(spot, expiry_date)
        deltas.append(greeks["Delta"])
        gammas.append(greeks["Gamma"])
        thetas.append(greeks["Theta"])
        vegas.append(greeks["Vega"])
        rhos.append(greeks["Rho"])

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.plot(spot_range, deltas, label="Delta", color='blue')
    plt.xlabel("Spot Price")
    plt.ylabel("Delta")
    plt.title("Delta vs Spot")

    plt.subplot(2, 3, 2)
    plt.plot(spot_range, gammas, label="Gamma", color='red')
    plt.xlabel("Spot Price")
    plt.ylabel("Gamma")
    plt.title("Gamma vs Spot")

    plt.subplot(2, 3, 3)
    plt.plot(spot_range, thetas, label="Theta", color='green')
    plt.xlabel("Spot Price")
    plt.ylabel("Theta")
    plt.title("Theta vs Spot")

    plt.subplot(2, 3, 4)
    plt.plot(spot_range, vegas, label="Vega", color='purple')
    plt.xlabel("Spot Price")
    plt.ylabel("Vega")
    plt.title("Vega vs Spot")

    plt.subplot(2, 3, 5)
    plt.plot(spot_range, rhos, label="Rho", color='brown')
    plt.xlabel("Spot Price")
    plt.ylabel("Rho")
    plt.title("Rho vs Spot")

    plt.tight_layout()
    plt.suptitle("Option Greeks vs Underlying Price (Expiry = 30 days)", y=1.02)
    plt.show()

def plot_greeks_vs_time():
    """
    Plot the option Greeks as a function of time to expiry.
    The spot price is set at strike (ATM) and risk-free rate is constant.
    """
    spot = strike  # ATM position
    days_range = np.linspace(1, 90, 50)  # from 1 day to 90 days
    deltas, gammas, thetas, vegas, rhos = [], [], [], [], []

    for days in days_range:
        expiry_date = evaluation_date + int(days)
        greeks = compute_greeks(spot, expiry_date)
        deltas.append(greeks["Delta"])
        gammas.append(greeks["Gamma"])
        thetas.append(greeks["Theta"])
        vegas.append(greeks["Vega"])
        rhos.append(greeks["Rho"])

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.plot(days_range, deltas, label="Delta", color='blue')
    plt.xlabel("Days to Expiry")
    plt.ylabel("Delta")
    plt.title("Delta vs Time to Expiry")

    plt.subplot(2, 3, 2)
    plt.plot(days_range, gammas, label="Gamma", color='red')
    plt.xlabel("Days to Expiry")
    plt.ylabel("Gamma")
    plt.title("Gamma vs Time to Expiry")

    plt.subplot(2, 3, 3)
    plt.plot(days_range, thetas, label="Theta", color='green')
    plt.xlabel("Days to Expiry")
    plt.ylabel("Theta")
    plt.title("Theta vs Time to Expiry")

    plt.subplot(2, 3, 4)
    plt.plot(days_range, vegas, label="Vega", color='purple')
    plt.xlabel("Days to Expiry")
    plt.ylabel("Vega")
    plt.title("Vega vs Time to Expiry")

    plt.subplot(2, 3, 5)
    plt.plot(days_range, rhos, label="Rho", color='brown')
    plt.xlabel("Days to Expiry")
    plt.ylabel("Rho")
    plt.title("Rho vs Time to Expiry")

    plt.tight_layout()
    plt.suptitle("Option Greeks vs Time to Expiry (ATM Option)", y=1.02)
    plt.show()

def plot_greeks_vs_riskfree():
    """
    Plot the option Greeks as a function of the risk-free rate.
    The spot price is ATM and the expiry is fixed (30 days away).
    """
    expiry_date = evaluation_date + 30
    spot = strike
    r_range = np.linspace(0.0, 0.05, 50)  # risk-free rate from 0% to 5%
    deltas, gammas, thetas, vegas, rhos = [], [], [], [], []

    for r in r_range:
        greeks = compute_greeks(spot, expiry_date, r)
        deltas.append(greeks["Delta"])
        gammas.append(greeks["Gamma"])
        thetas.append(greeks["Theta"])
        vegas.append(greeks["Vega"])
        rhos.append(greeks["Rho"])

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.plot(r_range, deltas, label="Delta", color='blue')
    plt.xlabel("Risk-Free Rate")
    plt.ylabel("Delta")
    plt.title("Delta vs Risk-Free Rate")

    plt.subplot(2, 3, 2)
    plt.plot(r_range, gammas, label="Gamma", color='red')
    plt.xlabel("Risk-Free Rate")
    plt.ylabel("Gamma")
    plt.title("Gamma vs Risk-Free Rate")

    plt.subplot(2, 3, 3)
    plt.plot(r_range, thetas, label="Theta", color='green')
    plt.xlabel("Risk-Free Rate")
    plt.ylabel("Theta")
    plt.title("Theta vs Risk-Free Rate")

    plt.subplot(2, 3, 4)
    plt.plot(r_range, vegas, label="Vega", color='purple')
    plt.xlabel("Risk-Free Rate")
    plt.ylabel("Vega")
    plt.title("Vega vs Risk-Free Rate")

    plt.subplot(2, 3, 5)
    plt.plot(r_range, rhos, label="Rho", color='brown')
    plt.xlabel("Risk-Free Rate")
    plt.ylabel("Rho")
    plt.title("Rho vs Risk-Free Rate")

    plt.tight_layout()
    plt.suptitle("Option Greeks vs Risk-Free Rate (ATM, Expiry = 30 days)", y=1.02)
    plt.show()

def main():
    print("Plotting Greeks versus underlying spot price...")
    plot_greeks_vs_spot()

    print("Plotting Greeks versus time to expiry...")
    plot_greeks_vs_time()

    print("Plotting Greeks versus risk-free rate...")
    plot_greeks_vs_riskfree()

if __name__ == "__main__":
    main()
