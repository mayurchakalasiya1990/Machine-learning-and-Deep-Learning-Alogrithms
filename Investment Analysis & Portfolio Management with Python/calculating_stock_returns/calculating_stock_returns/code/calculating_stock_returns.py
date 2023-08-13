# © Copyright, Fervent | All Rights Reserved
"""
# =====================================================
# Calculating Stock Returns - Applied
# =====================================================

Beginner?
We STRONGLY recommend using the .ipynb version instead of this .py version
The .ipynb has more explanatory notes to help and guide you through.

The .py version is largely designed for more intermediate / advanced users of
Python.
"""

# Import package dependencies
import pandas as pd  # for dealing with data
import matplotlib.pyplot as plt  # for plotting
import seaborn as sns  # for making plots look nicer
sns.set()  # implementing Seaborn's style and themes

# Load the Facebook price data (Source: Yahoo! Finance)
# Note that "../" changes the directory to the preceding folder.
df = pd.read_csv("../data/fb_price.csv")

# Extract the 'Date' and Adjusted Close ('Adj Close') columns only
df = df[['Date', 'Adj Close']]

# Rename to better match PEP8 standards
df.rename(columns={'Date' : 'date', 'Adj Close' : 'price_t'}, inplace=True)

# Shift the price_t column by 1 index to obtain price_t-1
df['price_t-1'] = df['price_t'].shift(1)

# Calculate returns using our standard formula ((P_t / P_t-1) - 1)
df['returns_manual'] = (df['price_t'] / df['price_t-1']) - 1

# Calculate returns using the pct_change() method
df['returns_pct_change_method'] = df['price_t'].pct_change(1)

# Calculate the returns "manually", without creating a separate column for price_t-1
df['returns'] = (df['price_t'] / df['price_t'].shift(1)) - 1

# Set the 'date' column as the index to help ensure dates show up in our plot.
df.set_index('date', inplace=True)

# Plot Facebook's Price data
df['price_t'].plot(figsize=(12, 8))

# Plot Facebook's Returns data
df['returns'].plot(figsize=(12, 8))
