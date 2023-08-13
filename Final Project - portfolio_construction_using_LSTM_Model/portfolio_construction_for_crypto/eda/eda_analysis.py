# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 14:00:15 2022

@author: ChakalasiyaMayurVash
"""

#importing libraries
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import  matplotlib.dates  as mdates
from matplotlib import rc
import seaborn as sns
import pandas_datareader.data as web
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import plotly
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import warnings
import time
import yfinance as yf
warnings.filterwarnings("ignore")

#for bold
start='\033[1m'
end='\033[0;0m'
ts = str(time.time())


def plot_top_four_crypto_as_per_marketcap(coin_list):
    mcaps = {}
    for t in coin_list:
        stock = yf.Ticker(t)
        mcaps[t] = stock.info["marketCap"]
        
    coins_marktcaps=pd.DataFrame(mcaps.items(), columns=["Symbol","Market Cap"]) 
    top_5_currency_names =  coins_marktcaps.groupby(['Symbol'])['Market Cap'].last().sort_values(ascending=False).head(5).sort_values()
    fig_name ="../output/marketcap_bar_chart"+ts+".png" 
    plt.figure(figsize=(18,5))
    top_5_currency_names_plt=top_5_currency_names.plot(kind='barh', color=(0.1, 0.1, 0.1, 0.1),  edgecolor='blue')
    top_5_currency_names_plt.ticklabel_format( style='plain', axis='x')
    top_5_currency_names_plt.set_xlabel("Market Cap (in billion $)")
    top_5_currency_names_plt.ticklabel_format( style='plain', axis='x')
    plt.title("Top 5 Cryptocurrencies by Market Cap in Currency", fontsize=15)
    plt.show()
    plt.savefig(fig_name)
    top_5_currency_names = pd.DataFrame(top_5_currency_names)
    top_5_currency_names.reset_index(inplace=True)
    top_5_currency_names.sort_values("Market Cap",ascending=False,inplace=True)
    return top_5_currency_names['Symbol'].tolist()


def plot_close_price_top_four_crypto(data,top_five_coins):
    
    
    data_top_5_currencies = data[data['Symbol'].isin(top_five_coins)]
    top_five_coins.remove('BTC-USD')
    top_5_currencies_after_BTC = data[data['Symbol'].isin(top_five_coins)]
    top_five_coins.remove('ETH-USD')
    top_5_currencies_after_BTC_ETH = data[data['Symbol'].isin(top_five_coins)]
    top_five_coins.remove('BNB-USD')
    top_5_currencies_after_BTC_ETH_BNB = data[data['Symbol'].isin(top_five_coins)]

    
    
    plt.figure(figsize=(20,25))    
    
    
    plt.subplot(4,1,1)    
    sns.lineplot(data=data_top_5_currencies, x="Date", y="Close", hue='Symbol')
    plt.title("Closing Prices of Top 5 Cryptocurrencies", fontsize=20)
    plt.legend(loc='upper left')
    
    plt.subplot(4,1,2)
    sns.lineplot(data=top_5_currencies_after_BTC, x="Date", y="Close", hue='Symbol')
    plt.title("Closing Prices of Top 5 Cryptocurrencies except BTC", fontsize=20)
    plt.legend(loc='upper left')
    
    plt.subplot(4,1,3)
    sns.lineplot(data=top_5_currencies_after_BTC_ETH,x="Date", y="Close", hue='Symbol')
    plt.title("Closing Prices of Top 5 Cryptocurrencies except BTC & ETH", fontsize=20)
    plt.legend(loc='upper left')
    
    plt.subplot(4,1,4)
    sns.lineplot(data=top_5_currencies_after_BTC_ETH_BNB,x="Date", y="Close", hue='Symbol')
    plt.title("Closing Prices of Top 5 Cryptocurrencies except BTC, ETH & BNB", fontsize=20)
    plt.legend(loc='upper left')
    
    fig, ax = plt.subplots(figsize = (20,25))    
    locator = mdates.MonthLocator(bymonth=[4,7,10])
    ax.xaxis.set_minor_locator(locator)
    ax.xaxis.set_minor_formatter(mdates.ConciseDateFormatter(locator))
    
    plt.show()
    


def plot_candlestick_for_top_four_crypto(data,top_5_coins_by_market_cap):    
    for i in top_5_coins_by_market_cap:        
        j=1
        options=[i]
        rslt_df = data[data['Symbol'].isin(options)]        
        layout = dict(
                title=f"{i} Candlestick Chart",
                xaxis=go.layout.XAxis(title=go.layout.xaxis.Title( text="Time")),
                yaxis=go.layout.YAxis(title=go.layout.yaxis.Title( text="Price US Dollars"))
        )
        data=[go.Candlestick(x=rslt_df['Date'],
                        open=rslt_df['Open'],
                        high=rslt_df['High'],
                        low=rslt_df['Low'],
                        close=rslt_df['Close'])]
        figSignal = go.Figure(data=data,layout=layout)
        j=j+1
        figSignal.show()
        
def plot_moving_avg_50days_200days(data,top_5_coins_by_market_cap):
    fig_name2 ="../output/moving_avg" 
    for i in top_5_coins_by_market_cap:
        j=1
        crypto_data=data[data['Symbol']==i]        
        top_currency = crypto_data[crypto_data['Symbol'].isin(top_5_coins_by_market_cap)]
        top_currency['Moving Average 50d']=top_currency['Close'].rolling(window=50).mean()
        top_currency['Moving Average 200d']=top_currency['Close'].rolling(window=200).mean()
        plt.subplot(5,1,j)
        top_currency['Close'].plot(figsize=(15,18))
        ax=top_currency['Moving Average 50d'].rolling(window=50).mean().plot()
        ax=top_currency['Moving Average 200d'].rolling(window=200).mean().plot()
        ax.set_ylabel("Price per 1 USD");
        plt.title(f"Moving Average vs Closing Price {i}", fontsize=25);
        plt.legend()
        j=j+1
        fig_name2 = fig_name2 +"_"+str(j)+"_"+ts + ".png"
        plt.savefig(fig_name2)
        plt.show()
        
        
def area_plot_for_top_crypto(data):
    area = px.area(data_frame= data , x = "Date" ,y= "High", line_group="Name" , color = "Name" , color_discrete_sequence=px.colors.qualitative.Alphabet_r,title = 'Area Plot for TOP Cryptocurrencies')

    area.update_xaxes(
        title_text = 'Date',rangeslider_visible = True,rangeselector = dict(buttons = list([dict(count = 1, label = '1M', step = 'month', stepmode = 'backward'),
                dict(count = 6, label = '6M', step = 'month', stepmode = 'backward'),
                dict(count = 1, label = 'YTD', step = 'year', stepmode = 'todate'),
                dict(count = 1, label = '1Y', step = 'year', stepmode = 'backward'),
                dict(step = 'all')])))

    area.update_yaxes(title_text = 'Price in USD', ticksuffix = '$')
    area.update_layout(showlegend = True,title = {'text': 'Area Plot for TOP Cryptocurrencies','y':0.9,'x':0.5,'xanchor': 'center',
                                                  'yanchor': 'top'})        
    plotly.offline.plot(area, filename='../area_plot_for_TOP_cryptocurrencies.html')
    area.show()

def area_plot_for_marketcap_change(data):
    area = px.area(data_frame = data,y  = data.High , x = data.Date , line_group=data.Symbol, color = data.Symbol, color_discrete_sequence=px.colors.qualitative.Alphabet, title = 'Market Cap Change of all Cryptocurrencies')

    area.update_xaxes(title_text = 'Date',rangeslider_visible = True,rangeselector = dict(buttons = list([
            dict(count = 1, label = '1M', step = 'month', stepmode = 'backward'),
            dict(count = 6, label = '6M', step = 'month', stepmode = 'backward'),
            dict(count = 1, label = 'YTD', step = 'year', stepmode = 'todate'),
            dict(count = 1, label = '1Y', step = 'year', stepmode = 'backward'),
            dict(step = 'all')])))

    area.update_yaxes(title_text = 'Percentage Change ', ticksuffix = '%')
    area.update_layout(showlegend = True,title = {'text': 'Volume Change of all Cryptocurrencies','y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'})
    plotly.offline.plot(area, filename='../area_plot_for_volume_change_all_cryptocurrencies.html')

    area.show()

def profit_correlation_for_top_crypto(data, top_5_coins_by_market_cap):    
    
    top_5_coins_data = data[data['Symbol'].isin(top_5_coins_by_market_cap)]
    top_5_coins_data=top_5_coins_data.pivot(index='Date', columns='Symbol', values='Adj Close').fillna(0).reset_index().rename_axis(None, axis=1)
    top_5_coins_data.plot(grid=True, figsize=(15, 10))
    
    top_5_coins_data.set_index("Date", inplace=True)
    
    coins_profit=top_5_coins_data.iloc[:,0:].pct_change().dropna(axis=0)
    coins_profit.replace([np.inf, -np.inf], 0, inplace=True)
    fig_name1 ="../output/profite" + ts + ".png"
    fig_name2 ="../output/profite_correlation" + ts + ".png"
    #ploting the returns
    fig, axs = plt.subplots(3,2,figsize=(20,12),gridspec_kw ={'hspace': 0.2, 'wspace': 0.1})
    axs[0,0].plot(coins_profit['BTC-USD'])
    axs[0,0].set_title('BTC')
    axs[0,0].set_ylim([-0.5,0.5])
    axs[0,1].plot(coins_profit['ETH-USD'])
    axs[0,1].set_title('ETH')
    axs[0,1].set_ylim([-0.5,0.5])
    axs[1,0].plot(coins_profit['USDT-USD'])
    axs[1,0].set_title('USDT')
    axs[1,0].set_ylim([-0.5,0.5])
    axs[1,1].plot(coins_profit['BNB-USD'])
    axs[1,1].set_title('BNB')
    axs[1,1].set_ylim([-0.5,0.5])
    axs[2,0].plot(coins_profit['USDC-USD'])
    axs[2,0].set_title('USDC')
    axs[2,0].set_ylim([-0.5,0.5])
    plt.show()
    plt.savefig(fig_name1)
    
    profit_corr = coins_profit.corr()
    
    sns.heatmap(profit_corr, annot=True, cmap='coolwarm')    
    plt.savefig(fig_name2)
    plt.show()
    return coins_profit

def valatility_for_top_crytp(coins_profit):
    fig_name="../output/valatility" + ts + ".png"
    fig, axs = plt.subplots(3,2,figsize=(20,12),gridspec_kw ={'hspace': 0.2, 'wspace': 0.1})
    axs[0,0].hist(coins_profit['BTC-USD'], bins=50, range=(-0.2, 0.2))
    axs[0,0].set_title('BTC')
    axs[0,1].hist(coins_profit['ETH-USD'], bins=50, range=(-0.2, 0.2))
    axs[0,1].set_title('ETH')
    axs[1,0].hist(coins_profit['USDT-USD'], bins=50, range=(-0.2, 0.2))
    axs[1,0].set_title('USDT')
    axs[1,1].hist(coins_profit['BNB-USD'], bins=50, range=(-0.2, 0.2))
    axs[1,1].set_title('BNB')
    axs[2,0].hist(coins_profit['USDC-USD'], bins=50, range=(-0.2, 0.2))
    axs[2,0].set_title('USDC')    
    plt.savefig(fig_name)
    plt.show()

def cumulative_return_for_top_crytp(coins_profit):
    # Cumulative return series
    fig_name="../output/cumulative_returns" + ts + ".png"
    cum_profit = ((1 + coins_profit).cumprod() - 1) *100
    cum_profit.head()    
    cum_profit.plot(figsize=(20,6))
    plt.title('Cumulative Returns')
    plt.Text(0.5, 1.0, 'Cumulative Returns')
    plt.savefig(fig_name)
    plt.show()

def close_price_correlation_for_all_crypto(data):    
    data.drop_duplicates(subset=None, keep='first', inplace=True, ignore_index=False)

    coins_data=data.pivot(index='Date', columns='Symbol', values='Adj Close').fillna(0).reset_index().rename_axis(None, axis=1)    
    coins_close_price=coins_data.iloc[:,1:]
    coins_close_price.replace([np.inf, -np.inf], 0, inplace=True)    
    close_price_corr = coins_close_price.corr()
    fig_name1 ="../output/profite_correlation" + ts + ".png"
    sns.heatmap(close_price_corr, annot=True, cmap='coolwarm')    
    plt.savefig(fig_name1)
    plt.show()
    return close_price_corr

coin_list = ['BUSD-USD','BTC-USD', 'ETH-USD', 'XRP-USD', 'ADA-USD', 'USDT-USD', 'DOGE-USD', 'XLM-USD', 'DOT-USD', 'UNI7083-USD', 'LINK-USD', 'USDC-USD', 'BCH-USD', 'LTC-USD', 'GRT6719-USD', 'ETC-USD', 'FIL-USD', 'AAVE-USD', 'ALGO-USD', 'EOS-USD','BNB-USD','TRX-USD','UNI7083-USD','SHIB-USD','SOL-USD','DAI-USD','MATIC-USD']
crypto_data = pd.read_csv("../data/final_df.csv")
crypto_data['Date']=pd.to_datetime(crypto_data['Date']).dt.date
print("Missing Values in Dataset")
print(crypto_data.isnull().sum())

#close_price_correlation_for_all_crypto(crypto_data)
top_five_coins=['BTC-USD', 'ETH-USD', 'USDT-USD', 'BNB-USD', 'USDC-USD']
coins_returns = profit_correlation_for_top_crypto(crypto_data,top_five_coins)
cumulative_return_for_top_crytp(coins_returns)
"""
#top_five_coins=plot_top_four_crypto_as_per_marketcap(coin_list)
print(top_five_coins)
plot_close_price_top_four_crypto(crypto_data,top_five_coins)
#plot_candlestick_for_top_four_crypto(crypto_data,top_five_coins)
#plot_moving_avg_50days_200days(crypto_data,top_five_coins)
#area_plot_for_top_crypto(crypto_data)
#coins_returns = profit_correlation_for_top_crypto(crypto_data,top_five_coins)
#valatility_for_top_crytp(coins_returns)
#cumulative_return_for_top_crytp(coins_returns)
"""