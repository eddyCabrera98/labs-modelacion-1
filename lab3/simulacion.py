import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import norm
from scipy.stats import uniform
from sklearn.linear_model import LinearRegression

def get_random_uniform_values(low, high, num_samples):
    return np.random.uniform(low, high, num_samples)
def get_random_triangular_values(left, mode, right, num_samples):
    return np.random.triangular(left, mode, right, num_samples)



def simular_ejercicio1(rates, annual_contribution):

    # Initialize the value of the investment
    final_value = 0
    accum_by_year = []
    saved_by_year = []
    interest_by_year = []
    # Calculate the investment value year by year
    for rate in rates:
        interest = (final_value + annual_contribution)* rate
        interest_by_year.append(interest)
        next_value = (final_value + annual_contribution) + interest
        saved_by_year.append(next_value - final_value)
        final_value = next_value
        accum_by_year.append(final_value)

    return final_value, accum_by_year, saved_by_year, interest_by_year

def ejercicio1(annual_contribution, mean, std_dev, years, n_sims):
    final_values = []
    accumulated_by_year_by_sim = []
    saved_by_year_by_sim = []
    interest_by_year_by_sim = []
    for i in range(n_sims):
        random_rates = np.random.normal(loc=mean, scale=std_dev, size=years)
        final_value, accum_by_year, saved_by_year, interest_by_year = simular_ejercicio1(random_rates, annual_contribution)
        final_values.append(final_value)
        accumulated_by_year_by_sim.append(accum_by_year)
        saved_by_year_by_sim.append(saved_by_year)
        interest_by_year_by_sim.append(interest_by_year)
    return final_values, accumulated_by_year_by_sim, saved_by_year_by_sim, interest_by_year_by_sim

def ejercicio2(n_sims, num_items, df, model):
    df_results = pd.DataFrame({'TV': [],'Radio': [],  'Newspaper': []})
    data = df["Radio"]
    max_radio = data.max()
    min_radio = data.min()
    data = df["TV"]
    max_tv = data.max()
    min_tv = data.min()
    data = df["Newspaper"]
    min_news = data.min()
    mode_news = data.mode()[0]
    max_news = data.max()
    for i in range(n_sims):
        highest_sale = 0
        X_best = None
        random_radio = get_random_uniform_values( min_radio, max_radio, num_items)
        random_tv = get_random_uniform_values(min_tv, max_tv, num_items)
        random_newspaper = get_random_triangular_values( min_news, mode_news, max_news, num_items)
        X = pd.DataFrame({'TV': random_tv, 'Radio': random_radio, 'Newspaper': random_newspaper})
        result = model.predict(X)
        X_best = X.iloc[np.argmax(result)]
        df_results = pd.concat([df_results, X_best.to_frame().T], ignore_index=True)
    return df_results