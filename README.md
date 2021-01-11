# momentum-value-portfolio
Momentum vs Value Portfolio of 50-best S&amp;P500 stocks
# sentiment-amazon-markers-reviews
Sentiment A
# Summary
I am using "yahoofinancials" and "yfinance" libraries to access prices and value indicators of stocks in SP500 index, constructing portfolio of 50 best momentum/value stocks and back test reallocation approaches
## Dependencies
```python
import pandas_datareader.data as web
import pandas as pd
from datetime import date
import yahoo_fin.stock_info as si
from yahoo_fin.stock_info import get_data
import numpy as np
import math
import time
from yahoofinancials import YahooFinancials
from scipy import stats
from statistics import mean
from scipy.stats import percentileofscore
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
from decimal import Decimal
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline
import plotly.graph_objects as go
from plotly.offline import iplot
import plotly.express as px
from plotly.subplots import make_subplots
import pickle
import yfinance as yf
```
## List of stocks in SP500
Uploading stocks in SP500 using ```python yahoo_fin.stock_info```
```python
sp_list = si.tickers_sp500()
len(sp_list)
```
