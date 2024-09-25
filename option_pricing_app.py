import streamlit as st
import numpy as np
from scipy.stats import norm
from math import exp, sqrt, log
import matplotlib.pyplot as plt
import yfinance as yahooFinance
# Define the models

class BlackScholesModel:
    @staticmethod
    def calculate_option_price(S, K, T, r, sigma, option_type='call'):
        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        if option_type == 'call':
            option_price = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
        else:
            option_price = K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return option_price

class MonteCarloPricing:
    @staticmethod
    def calculate_option_price(S, K, T, r, sigma, simulations=10000, option_type='call'):
        simulations = int(simulations)
        dt = T / simulations
        option_payoffs = []
        for _ in range(simulations):
            ST = S * exp((r - 0.5 * sigma ** 2) * T + sigma * sqrt(T) * np.random.normal(0, 1))
            if option_type == 'call':
                payoff = max(0, ST - K)
            else:
                payoff = max(0, K - ST)
            option_payoffs.append(payoff)
        return exp(-r * T) * np.mean(option_payoffs)

class BinomialTreeModel:
    @staticmethod
    def calculate_option_price(S, K, T, r, sigma, steps=100, option_type='call'):
        steps = int(steps) 
        dt = T / steps
        u = exp(sigma * sqrt(dt))
        d = 1 / u
        p = (exp(r * dt) - d) / (u - d)
        prices = np.zeros((steps + 1, steps + 1))
        option_values = np.zeros((steps + 1, steps + 1))

        for i in range(steps + 1):
            prices[i, steps] = S * (u ** (steps - i)) * (d ** i)

        for i in range(steps + 1):
            if option_type == 'call':
                option_values[i, steps] = max(0, prices[i, steps] - K)
            else:
                option_values[i, steps] = max(0, K - prices[i, steps])

        for j in range(steps - 1, -1, -1):
            for i in range(j + 1):
                option_values[i, j] = exp(-r * dt) * (
                    p * option_values[i, j + 1] + (1 - p) * option_values[i + 1, j])

        return option_values[0, 0]

# Streamlit UI
st.title('Option Pricing for S&P 500')
st.sidebar.header('Input Parameters')

# Input parameters
S = float(st.sidebar.number_input('Underlying Asset Price (S)', value=4500.0))
K = float(st.sidebar.number_input('Strike Price (K)', value=4500.0))
T = float(st.sidebar.number_input('Time to Maturity (T) in years', value=1.0))
r = float(st.sidebar.number_input('Risk-Free Rate (r)', value=0.01))
sigma = float(st.sidebar.number_input('Volatility (σ)', value=0.2))
option_type = st.sidebar.selectbox('Option Type', ['call', 'put'])
simulations = int(st.sidebar.number_input('Number of Simulations', value=10000))  # Ensure this is an integer

# Calculate prices
bs_price = BlackScholesModel.calculate_option_price(S, K, T, r, sigma, option_type)
mc_price = MonteCarloPricing.calculate_option_price(S, K, T, r, sigma, simulations, option_type)
bt_price = BinomialTreeModel.calculate_option_price(S, K, T, r, sigma, simulations // 100)

# Display results
st.write(f'### Black-Scholes Model Price: ${bs_price:.2f}')
st.write(f'### Monte Carlo Simulation Price: ${mc_price:.2f}')
st.write(f'### Binomial Tree Model Price: ${bt_price:.2f}')

# Option type visual representation
# if st.checkbox('Show price distribution for Monte Carlo'):
#     st_vals = []
#     for _ in range(simulations):
#         ST = S * exp((r - 0.5 * sigma ** 2) * T + sigma *
#                      sqrt(T) * np.random.normal(0, 1))
#         st_vals.append(ST)
#     plt.hist(st_vals, bins=50)
#     plt.title('Monte Carlo Simulation: Asset Price Distribution at Maturity')
#     plt.xlabel('Asset Price')
#     plt.ylabel('Frequency')
#     st.pyplot(plt.gcf())

if st.checkbox('Show price distribution for Monte Carlo'):
    st_vals = []
    for _ in range(simulations):
        ST = S * exp((r - 0.5 * sigma ** 2) * T + sigma * sqrt(T) * np.random.normal(0, 1))
        st_vals.append(ST)

    # Convert st_vals to a NumPy array if necessary
    st_vals = np.array(st_vals)

    # Now plot the histogram
    plt.hist(st_vals, bins=50)
    plt.title('Monte Carlo Simulation: Asset Price Distribution at Maturity')
    plt.xlabel('Asset Price')
    plt.ylabel('Frequency')
    st.pyplot(plt.gcf())
