import requests

API_KEY = 'YOUR_API_KEY'  # Replace with your Alpha Vantage API key

def get_company_overview(ticker):
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'OVERVIEW',
        'symbol': ticker,
        'apikey': API_KEY
    }
    response = requests.get(url, params=params)
    return response.json()



def get_fx_weekly(from_symbol, to_symbol):
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'FX_WEEKLY',
        'from_symbol': from_symbol,
        'to_symbol': to_symbol,
        'apikey': API_KEY
    }
    response = requests.get(url, params=params)
    return response.json()

def get_symbol_search(keywords):
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'SYMBOL_SEARCH',
        'keywords': keywords,
        'apikey': API_KEY
    }
    response = requests.get(url, params=params)
    return response.json()

def get_time_series_intraday(symbol, interval='5min'):
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_INTRADAY',
        'symbol': symbol,
        'interval': interval,
        'apikey': API_KEY
    }
    response = requests.get(url, params=params)
    return response.json()

def get_time_series_monthly(symbol):
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_MONTHLY',
        'symbol': symbol,
        'apikey': API_KEY
    }
    response = requests.get(url, params=params)
    return response.json()

def get_time_series_monthly_adjusted(symbol):
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_MONTHLY_ADJUSTED',
        'symbol': symbol,
        'apikey': API_KEY
    }
    response = requests.get(url, params=params)
    return response.json()

def get_time_series_weekly(symbol):
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_WEEKLY',
        'symbol': symbol,
        'apikey': API_KEY
    }
    response = requests.get(url, params=params)
    return response.json()



def get_balance_sheet(symbol):
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'BALANCE_SHEET',
        'symbol': symbol,
        'apikey': API_KEY
    }
    response = requests.get(url, params=params)
    return response.json()

def get_cash_flow(symbol):
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'CASH_FLOW',
        'symbol': symbol,
        'apikey': API_KEY
    }
    response = requests.get(url, params=params)
    return response.json()

def get_earnings(symbol):
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'EARNINGS',
        'symbol': symbol,
        'apikey': API_KEY
    }
    response = requests.get(url, params=params)
    return response.json()

def get_earnings_calendar(horizon='3month', symbol=None):
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'EARNINGS_CALENDAR',
        'horizon': horizon,
        'symbol': symbol,
        'apikey': API_KEY
    }
    response = requests.get(url, params=params)
    return response.json()

def get_ipo_calendar():
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'IPO_CALENDAR',
        'apikey': API_KEY
    }
    response = requests.get(url, params=params)
    return response.json()

def get_news_sentiment(tickers=None, limit=50, sort='LATEST'):
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'NEWS_SENTIMENT',
        'tickers': tickers,
        'limit': limit,
        'sort': sort,
        'apikey': API_KEY
    }
    response = requests.get(url, params=params)
    return response.json()

def get_real_gdp(interval='annual'):
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'REAL_GDP',
        'interval': interval,
        'apikey': API_KEY
    }
    response = requests.get(url, params=params)
    return response.json()

def get_sector():
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'SECTOR',
        'apikey': API_KEY
    }
    response = requests.get(url, params=params)
    return response.json()

def get_splits(symbol):
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'SPLITS',
        'symbol': symbol,
        'apikey': API_KEY
    }
    response = requests.get(url, params=params)
    return response.json()

def get_technical_indicator(symbol, function, interval='daily', series_type='close', time_period=14):
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': function,
        'symbol': symbol,
        'interval': interval,
        'series_type': series_type,
        'time_period': time_period,
        'apikey': API_KEY
    }
    response = requests.get(url, params=params)
    return response.json()