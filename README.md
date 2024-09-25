# **Overview**

This Streamlit app calculates option prices for S&P 500 companies using three models: Black-Scholes, Monte Carlo Simulation, and Binomial Tree. Users can input parameters such as asset price, strike price, volatility, and more to explore various option pricing scenarios.

## Features

	•	Option Pricing Models:
	•	Black-Scholes Model: Calculates the theoretical price of European options.
	•	Monte Carlo Simulation: Uses random sampling to estimate option prices.
	•	Binomial Tree Model: Implements a discrete-time model for option pricing.
	•	Visualizations:
	•	Histogram of asset price distributions from Monte Carlo simulations.
	•	Interactive UI for user inputs and selections.


### Future Updates

Real-Time S&P 500 Company Pricing

	•	The app will be integrated with the Yahoo Finance API to fetch real-time stock prices of S&P 500 companies.
	•	Users can select a specific company from a dropdown list, and the app will update the option pricing calculations based on the latest market data.
	•	Visualizations will display current asset price distributions and option pricing results dynamically.


### Clone Repository 
       •        git clone https://github.com/Bojack15/option-pricing-app.git
       •        cd option-pricing-app

##### Install Required packages 
      •         pip install -r requirements.txt

##### Run the Streamlit app:
      •         streamlit run app.py


