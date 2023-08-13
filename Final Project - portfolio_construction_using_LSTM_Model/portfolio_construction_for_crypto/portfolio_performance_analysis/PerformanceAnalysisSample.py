# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 20:37:39 2022

@author: ChakalasiyaMayurVash
"""

from datetime import date


import profolio_perf_analysis as pa

print(pa.perfAnalysis(['AAPL', 'AMZN', 'GOOG', 'TSLA'], 
                portf_weights=[0.25,0.25,0.25,0.25],
                start=date(2011,1,4), end=date.today(),
                riskfree_rate=0,  init_cap=1,
                chart_size=(22,12)))