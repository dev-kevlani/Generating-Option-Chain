"""
Greeks Effect Analysis for Multiple Strikes and Expiry Dates with Spot Price Sensitivity
Analyzes calls and puts with up/down 1% spot moves
"""

import QuantLib as ql
import numpy as np
import matplotlib.pyplot as plt

# Global parameters
spot = 100.0
vol = 0.20
dividend_yield = 0.0
risk_free_rate = 0.01
day_count = ql.Actual365Fixed()
calendar = ql.UnitedStates(ql.UnitedStates.NYSE)

# Define evaluation date
evaluation_date = ql.Date.todaysDate()
ql.Settings.instance().evaluationDate = evaluation_date

def create_option(spot, strike, vol, r, q, expiry_date, option_type):
    """
    Create a European option in QuantLib and return configured option with pricing engine.
    """
    payoff = ql.PlainVanillaPayoff(
        ql.Option.Call if option_type == 'call' else ql.Option.Put, 
        strike
    )
    exercise = ql.EuropeanExercise(expiry_date)
    option = ql.VanillaOption(payoff, exercise)

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
    
    bsm_process = ql.BlackScholesMertonProcess(
        spot_handle, dividend_curve, risk_free_curve, vol_curve
    )
    option.setPricingEngine(ql.AnalyticEuropeanEngine(bsm_process))
    
    return option

def compute_greeks(spot, strike, expiry_date, option_type='call'):
    """
    Compute Greeks for an option given parameters.
    Returns dictionary of Greeks.
    """
    option = create_option(spot, strike, vol, risk_free_rate, 
                         dividend_yield, expiry_date, option_type)
    
    _ = option.NPV()
    
    greeks = {}
    for greek in ['delta', 'gamma', 'theta', 'vega', 'rho']:
        try:
            greeks[greek.capitalize()] = getattr(option, greek)()
        except Exception:
            greeks[greek.capitalize()] = np.nan
            
    return greeks

def compute_greeks_with_shifts(spot, strike, expiry_date, option_type='call'):
    """
    Compute Greeks at base spot and with ±1% spot shifts
    """
    shift_up = spot * 1.01
    shift_down = spot * 0.99
    
    base_greeks = compute_greeks(spot, strike, expiry_date, option_type)
    up_greeks = compute_greeks(shift_up, strike, expiry_date, option_type)
    down_greeks = compute_greeks(shift_down, strike, expiry_date, option_type)
    
    return {
        'base': base_greeks,
        'up': up_greeks,
        'down': down_greeks
    }

def generate_greeks_data():
    """
    Generate Greeks data for both calls and puts with spot price sensitivity
    """
    strikes = np.linspace(90.0, 110.0, 21)
    expiries = np.array([1, 10, 30, 90])
    
    greek_names = ["Delta", "Gamma", "Theta", "Vega", "Rho"]
    
    greeks_dict = {
        'call': {greek: {'base': [], 'up': [], 'down': []} for greek in greek_names},
        'put': {greek: {'base': [], 'up': [], 'down': []} for greek in greek_names}
    }
    
    for days in expiries:
        expiry_date = evaluation_date + int(days)
        for strike in strikes:

            call_results = compute_greeks_with_shifts(spot, strike, expiry_date, 'call')
            put_results = compute_greeks_with_shifts(spot, strike, expiry_date, 'put')
            
            for greek in greek_names:
                for move in ['base', 'up', 'down']:
                    greeks_dict['call'][greek][move].append(call_results[move][greek])
                    greeks_dict['put'][greek][move].append(put_results[move][greek])
    
    for option_type in ['call', 'put']:
        for greek in greek_names:
            for move in ['base', 'up', 'down']:
                greeks_dict[option_type][greek][move] = np.array(
                    greeks_dict[option_type][greek][move]
                )
    
    return strikes, expiries, greeks_dict

def plot_greeks_analysis(strikes, expiries, greeks_dict):
    """
    Plot 6 subplots for each Greek showing calls and puts with up/down moves
    """
    greek_names = ["Delta", "Gamma", "Theta", "Vega", "Rho"]
    selected_expiries = [1, 10, 30, 90]
    
    for greek in greek_names:
        fig, axes = plt.subplots(1, 6, figsize=(24, 6))
        
        titles = [
            'Call Base', 'Call +1%', 'Call -1%',
            'Put Base', 'Put +1%', 'Put -1%'
        ]
        
        data_configs = [
            ('call', 'base'), ('call', 'up'), ('call', 'down'),
            ('put', 'base'), ('put', 'up'), ('put', 'down')
        ]
        
        for ax, title, (option_type, move) in zip(axes, titles, data_configs):
            data = greeks_dict[option_type][greek][move]
            for i, expiry in enumerate(selected_expiries):
                expiry_idx = np.where(expiries == expiry)[0][0]
                start_idx = expiry_idx * len(strikes)
                end_idx = start_idx + len(strikes)
                ax.plot(strikes, data[start_idx:end_idx],
                       label=f'{expiry} days', marker='o', markersize=3)
            
            ax.set_xlabel('Strike')
            ax.set_ylabel(greek)
            ax.set_title(f'{title}\n{greek}')
            ax.grid(True)
            ax.legend()
        
        plt.suptitle(f'{greek} Analysis - Spot: {spot}, Vol: {vol*100}%, Rate: {risk_free_rate*100}%')
        plt.tight_layout()
        plt.show()

def main():
    print("Generating Greeks data...")
    strikes, expiries, greeks_dict = generate_greeks_data()
    print("Plotting Greeks analysis...")
    plot_greeks_analysis(strikes, expiries, greeks_dict)
    return greeks_dict

if __name__ == "__main__":
    greeks_dict = main()
