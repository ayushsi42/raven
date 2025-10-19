import os
import requests
from dotenv import load_dotenv
from google import genai
from typing import TypedDict
import json

# Load environment variables from .env file
load_dotenv()

# Set API keys
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

print("Checking API keys...")
if not ALPHA_VANTAGE_API_KEY:
    raise ValueError("ALPHA_VANTAGE_API_KEY must be set in the .env file.")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY must be set in the .env file.")

print("Initializing Gemini 2.5 Flash...")
try:
    # Initialize the Gemini client with API key
    client = genai.Client(api_key=GEMINI_API_KEY)
    print("Gemini 2.5 Flash initialized successfully.")
except Exception as e:
    raise ValueError(f"Error initializing Gemini: {str(e)}")

# --- Helper function to call Gemini ---
def call_gemini(prompt: str, model: str = "gemini-2.5-flash") -> str:
    """
    Uses Google Gemini to generate responses
    Available models: gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-lite
    """
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "temperature": 0.1,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
            }
        )
        return response.text
    except Exception as e:
        print(f"Error calling Gemini: {str(e)}")
        return json.dumps({
            "error": f"Could not complete LLM analysis: {str(e)}",
            "fallback": "Analysis incomplete due to LLM unavailability"
        })

# --- Tool Definitions from alpha_tools.py ---

def get_company_overview(ticker):
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'OVERVIEW',
        'symbol': ticker,
        'apikey': ALPHA_VANTAGE_API_KEY
    }
    response = requests.get(url, params=params)
    return response.json()

def get_symbol_search(keywords):
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'SYMBOL_SEARCH',
        'keywords': keywords,
        'apikey': ALPHA_VANTAGE_API_KEY
    }
    response = requests.get(url, params=params)
    return response.json()

def get_time_series_daily_adjusted(symbol):
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_DAILY_ADJUSTED',
        'symbol': symbol,
        'outputsize': 'compact',
        'apikey': ALPHA_VANTAGE_API_KEY
    }
    response = requests.get(url, params=params)
    return response.json()

def get_income_statement(symbol):
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'INCOME_STATEMENT',
        'symbol': symbol,
        'apikey': ALPHA_VANTAGE_API_KEY
    }
    response = requests.get(url, params=params)
    return response.json()

def get_balance_sheet(symbol):
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'BALANCE_SHEET',
        'symbol': symbol,
        'apikey': ALPHA_VANTAGE_API_KEY
    }
    response = requests.get(url, params=params)
    return response.json()

def get_cash_flow(symbol):
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'CASH_FLOW',
        'symbol': symbol,
        'apikey': ALPHA_VANTAGE_API_KEY
    }
    response = requests.get(url, params=params)
    return response.json()

def get_news_sentiment(tickers):
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'NEWS_SENTIMENT',
        'tickers': tickers,
        'limit': 50,
        'sort': 'LATEST',
        'apikey': ALPHA_VANTAGE_API_KEY
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
        'time_period': str(time_period),
        'apikey': ALPHA_VANTAGE_API_KEY
    }
    response = requests.get(url, params=params)
    return response.json()


# --- LangGraph State Definition ---

class AgentState(TypedDict, total=False):
    hypothesis: str
    parsed_hypothesis: dict
    market_analysis: dict
    fundamental_analysis: dict
    sentiment_analysis: dict
    risk_analysis: dict
    final_report: dict

# --- Agent Definitions ---

def parsing_agent(state: AgentState):
    print("--- Running Parsing Agent ---")
    ticker = state['parsed_hypothesis']['ticker']
    
    try:
        overview = get_company_overview(ticker)
        company_name = overview.get("Name", ticker)
    except Exception:
        company_name = ticker

    updated_parsed_hypothesis = state['parsed_hypothesis'].copy()
    updated_parsed_hypothesis['company'] = company_name
    
    return {"parsed_hypothesis": updated_parsed_hypothesis}

def market_momentum_agent(state: AgentState):
    print("--- Running Market/Momentum Agent ---")
    ticker = state['parsed_hypothesis']['ticker']
    
    try:
        # Fetch all data
        daily_data = get_time_series_daily_adjusted(ticker)
        sma = get_technical_indicator(ticker, 'SMA', time_period=20)
        ema = get_technical_indicator(ticker, 'EMA', time_period=20)
        macd = get_technical_indicator(ticker, 'MACD')
        rsi = get_technical_indicator(ticker, 'RSI', time_period=14)
        
        # Extract recent price data
        time_series = daily_data.get('Time Series (Daily)', {})
        if not time_series:
            raise Exception("No time series data available")
        
        recent_dates = sorted(time_series.keys(), reverse=True)[:10]
        recent_prices = []
        for date in recent_dates:
            day_data = time_series[date]
            recent_prices.append({
                "date": date,
                "open": day_data.get("1. open"),
                "high": day_data.get("2. high"),
                "low": day_data.get("3. low"),
                "close": day_data.get("4. close"),
                "volume": day_data.get("6. volume")
            })
        
        # Extract SMA data
        sma_data = sma.get('Technical Analysis: SMA', {})
        sma_recent = []
        if sma_data:
            sma_dates = sorted(sma_data.keys(), reverse=True)[:5]
            for date in sma_dates:
                sma_recent.append({
                    "date": date,
                    "SMA": sma_data[date].get("SMA")
                })
        
        # Extract EMA data
        ema_data = ema.get('Technical Analysis: EMA', {})
        ema_recent = []
        if ema_data:
            ema_dates = sorted(ema_data.keys(), reverse=True)[:5]
            for date in ema_dates:
                ema_recent.append({
                    "date": date,
                    "EMA": ema_data[date].get("EMA")
                })
        
        # Extract MACD data
        macd_data = macd.get('Technical Analysis: MACD', {})
        macd_recent = []
        if macd_data:
            macd_dates = sorted(macd_data.keys(), reverse=True)[:5]
            for date in macd_dates:
                macd_recent.append({
                    "date": date,
                    "MACD": macd_data[date].get("MACD"),
                    "MACD_Signal": macd_data[date].get("MACD_Signal"),
                    "MACD_Hist": macd_data[date].get("MACD_Hist")
                })
        
        # Extract RSI data
        rsi_data = rsi.get('Technical Analysis: RSI', {})
        rsi_recent = []
        if rsi_data:
            rsi_dates = sorted(rsi_data.keys(), reverse=True)[:5]
            for date in rsi_dates:
                rsi_recent.append({
                    "date": date,
                    "RSI": rsi_data[date].get("RSI")
                })
        
        # Prepare comprehensive prompt with actual data
        prompt = f"""You are a market analysis expert. Analyze the following ACTUAL market data for {ticker}.

HYPOTHESIS: {state['hypothesis']}

RECENT PRICE DATA (Last 10 Days):
{json.dumps(recent_prices, indent=2)}

SIMPLE MOVING AVERAGE (SMA-20, Last 5 Days):
{json.dumps(sma_recent, indent=2)}

EXPONENTIAL MOVING AVERAGE (EMA-20, Last 5 Days):
{json.dumps(ema_recent, indent=2)}

MACD INDICATOR (Last 5 Days):
{json.dumps(macd_recent, indent=2)}

RSI INDICATOR (Last 5 Days):
{json.dumps(rsi_recent, indent=2)}

Based on this ACTUAL data, provide a comprehensive market analysis. Analyze:
1. Price trend (look at the closing prices over time)
2. Volume trends
3. SMA/EMA crossovers and price position relative to moving averages
4. MACD signals (bullish/bearish crossovers, histogram)
5. RSI levels (overbought >70, oversold <30, neutral 30-70)

You MUST respond with ONLY a valid JSON object (no markdown, no code blocks) with this structure:

{{
    "trend_direction": "upward/downward/sideways",
    "momentum_strength": 0.75,
    "short_term_signal": "bullish/bearish/neutral",
    "indicators": {{
        "sma": "detailed analysis with actual values",
        "ema": "detailed analysis with actual values",
        "macd": "detailed analysis with actual values and signals",
        "rsi": "detailed analysis with actual RSI value and interpretation"
    }},
    "confidence": 0.8,
    "hypothesis_support": "supports/opposes/neutral",
    "reasoning": "detailed reasoning based on the actual data provided"
}}"""

        # Get Gemini analysis
        response = call_gemini(prompt, model="gemini-2.5-flash")
        
        # Parse the response
        try:
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith('```'):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            analysis = json.loads(cleaned_response)
            # Store raw data for reference
            analysis['raw_data_summary'] = {
                'latest_close': recent_prices[0]['close'] if recent_prices else 'N/A',
                'latest_rsi': rsi_recent[0]['RSI'] if rsi_recent else 'N/A',
                'data_points': len(recent_prices)
            }
        except Exception as parse_error:
            print(f"Error parsing Gemini response: {parse_error}")
            print(f"Raw response: {response[:500]}")
            analysis = {
                "trend_direction": "unknown",
                "momentum_strength": 0.5,
                "short_term_signal": "neutral",
                "indicators": {
                    "sma": "Parsing error",
                    "ema": "Parsing error",
                    "macd": "Parsing error",
                    "rsi": "Parsing error"
                },
                "confidence": 0.3,
                "hypothesis_support": "inconclusive",
                "reasoning": f"Error parsing response: {str(parse_error)}"
            }
            
    except Exception as e:
        print(f"Warning: Error in market analysis - {str(e)}")
        analysis = {
            "trend_direction": "unknown",
            "momentum_strength": 0.5,
            "short_term_signal": "neutral",
            "error": f"Could not fetch market data: {str(e)}"
        }
    
    return {"market_analysis": analysis}

def fundamental_analysis_agent(state: AgentState):
    print("--- Running Fundamental Analysis Agent ---")
    ticker = state['parsed_hypothesis']['ticker']
    
    try:
        # Fetch all financial data
        income = get_income_statement(ticker)
        balance = get_balance_sheet(ticker)
        cashflow = get_cash_flow(ticker)
        overview = get_company_overview(ticker)
        
        # Extract comprehensive overview metrics
        overview_metrics = {
            "Name": overview.get("Name"),
            "Description": overview.get("Description"),
            "Sector": overview.get("Sector"),
            "Industry": overview.get("Industry"),
            "MarketCapitalization": overview.get("MarketCapitalization"),
            "EBITDA": overview.get("EBITDA"),
            "PERatio": overview.get("PERatio"),
            "PEGRatio": overview.get("PEGRatio"),
            "BookValue": overview.get("BookValue"),
            "DividendPerShare": overview.get("DividendPerShare"),
            "DividendYield": overview.get("DividendYield"),
            "EPS": overview.get("EPS"),
            "RevenuePerShareTTM": overview.get("RevenuePerShareTTM"),
            "ProfitMargin": overview.get("ProfitMargin"),
            "OperatingMarginTTM": overview.get("OperatingMarginTTM"),
            "ReturnOnAssetsTTM": overview.get("ReturnOnAssetsTTM"),
            "ReturnOnEquityTTM": overview.get("ReturnOnEquityTTM"),
            "RevenueTTM": overview.get("RevenueTTM"),
            "GrossProfitTTM": overview.get("GrossProfitTTM"),
            "QuarterlyEarningsGrowthYOY": overview.get("QuarterlyEarningsGrowthYOY"),
            "QuarterlyRevenueGrowthYOY": overview.get("QuarterlyRevenueGrowthYOY"),
            "AnalystTargetPrice": overview.get("AnalystTargetPrice"),
            "TrailingPE": overview.get("TrailingPE"),
            "ForwardPE": overview.get("ForwardPE"),
            "PriceToSalesRatioTTM": overview.get("PriceToSalesRatioTTM"),
            "PriceToBookRatio": overview.get("PriceToBookRatio"),
            "EVToRevenue": overview.get("EVToRevenue"),
            "EVToEBITDA": overview.get("EVToEBITDA"),
            "Beta": overview.get("Beta"),
            "52WeekHigh": overview.get("52WeekHigh"),
            "52WeekLow": overview.get("52WeekLow"),
            "50DayMovingAverage": overview.get("50DayMovingAverage"),
            "200DayMovingAverage": overview.get("200DayMovingAverage")
        }
        
        # Extract recent income statement data
        income_reports = []
        if income.get('annualReports'):
            for report in income['annualReports'][:3]:  # Last 3 years
                income_reports.append({
                    "fiscalDateEnding": report.get("fiscalDateEnding"),
                    "totalRevenue": report.get("totalRevenue"),
                    "grossProfit": report.get("grossProfit"),
                    "operatingIncome": report.get("operatingIncome"),
                    "netIncome": report.get("netIncome"),
                    "ebitda": report.get("ebitda"),
                    "eps": report.get("eps")
                })
        
        # Extract recent balance sheet data
        balance_reports = []
        if balance.get('annualReports'):
            for report in balance['annualReports'][:3]:
                balance_reports.append({
                    "fiscalDateEnding": report.get("fiscalDateEnding"),
                    "totalAssets": report.get("totalAssets"),
                    "totalLiabilities": report.get("totalLiabilities"),
                    "totalShareholderEquity": report.get("totalShareholderEquity"),
                    "cashAndCashEquivalents": report.get("cashAndCashEquivalentsAtCarryingValue"),
                    "currentDebt": report.get("currentDebt"),
                    "longTermDebt": report.get("longTermDebt")
                })
        
        # Extract recent cash flow data
        cashflow_reports = []
        if cashflow.get('annualReports'):
            for report in cashflow['annualReports'][:3]:
                cashflow_reports.append({
                    "fiscalDateEnding": report.get("fiscalDateEnding"),
                    "operatingCashflow": report.get("operatingCashflow"),
                    "capitalExpenditures": report.get("capitalExpenditures"),
                    "freeCashFlow": str(int(report.get("operatingCashflow", 0)) - int(report.get("capitalExpenditures", 0))) if report.get("operatingCashflow") and report.get("capitalExpenditures") else "N/A"
                })
        
        # Create comprehensive prompt
        prompt = f"""You are a financial analyst expert. Analyze the following ACTUAL fundamental data for {ticker}.

HYPOTHESIS: {state['hypothesis']}

COMPANY OVERVIEW & KEY METRICS:
{json.dumps(overview_metrics, indent=2)}

INCOME STATEMENT (Last 3 Years):
{json.dumps(income_reports, indent=2)}

BALANCE SHEET (Last 3 Years):
{json.dumps(balance_reports, indent=2)}

CASH FLOW STATEMENT (Last 3 Years):
{json.dumps(cashflow_reports, indent=2)}

Based on this ACTUAL financial data, analyze:
1. Profitability trends (revenue growth, profit margins, EPS growth)
2. Financial health (debt levels, liquidity, equity position)
3. Valuation metrics (P/E, PEG, P/B ratios compared to industry)
4. Growth trajectory (YoY revenue and earnings growth)
5. Cash generation ability (operating cash flow, free cash flow)

You MUST respond with ONLY a valid JSON object (no markdown, no code blocks) with this structure:

{{
    "profitability_support": 0.75,
    "financial_health": "strong/moderate/weak",
    "growth_trajectory": "accelerating/stable/declining",
    "key_metrics": {{
        "revenue_growth": "specific analysis with actual numbers",
        "profit_margins": "specific analysis with actual percentages",
        "debt_levels": "specific analysis with debt-to-equity ratio",
        "cash_position": "specific analysis with actual cash amounts"
    }},
    "risk_factors": ["specific risk 1", "specific risk 2"],
    "confidence": 0.8,
    "hypothesis_support": "supports/opposes/neutral",
    "reasoning": "detailed reasoning using actual financial metrics"
}}"""

        # Get Gemini analysis
        response = call_gemini(prompt, model="gemini-2.5-flash")
        
        # Parse the response
        try:
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith('```'):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            analysis = json.loads(cleaned_response)
            # Store key metrics
            analysis['key_financial_data'] = {
                'market_cap': overview_metrics.get('MarketCapitalization'),
                'pe_ratio': overview_metrics.get('PERatio'),
                'eps': overview_metrics.get('EPS'),
                'revenue_growth': overview_metrics.get('QuarterlyRevenueGrowthYOY')
            }
        except Exception as parse_error:
            print(f"Error parsing Gemini response: {parse_error}")
            analysis = {
                "profitability_support": 0.5,
                "financial_health": "unknown",
                "growth_trajectory": "unclear",
                "key_metrics": {
                    "revenue_growth": "Parsing error",
                    "profit_margins": "Parsing error",
                    "debt_levels": "Parsing error",
                    "cash_position": "Parsing error"
                },
                "risk_factors": ["Data parsing error"],
                "confidence": 0.3,
                "hypothesis_support": "inconclusive",
                "reasoning": f"Error parsing response: {str(parse_error)}"
            }
            
    except Exception as e:
        print(f"Warning: Error in fundamental analysis - {str(e)}")
        analysis = {
            "profitability_support": 0.5,
            "financial_health": "unknown",
            "error": str(e)
        }
    
    return {"fundamental_analysis": analysis}

def sentiment_agent(state: AgentState):
    print("--- Running Sentiment Agent ---")
    ticker = state['parsed_hypothesis']['ticker']
    
    try:
        sentiment_data = get_news_sentiment(ticker)
        
        article_count = 0
        relevant_articles = []
        sentiment_scores = []
        
        if 'feed' in sentiment_data and sentiment_data['feed']:
            for item in sentiment_data['feed']:
                ticker_sentiments = item.get('ticker_sentiment', [])
                for ticker_sentiment in ticker_sentiments:
                    if ticker_sentiment.get('ticker') == ticker:
                        article_count += 1
                        sentiment_score = float(ticker_sentiment.get('ticker_sentiment_score', '0'))
                        sentiment_label = ticker_sentiment.get('ticker_sentiment_label', 'Neutral')
                        relevance_score = float(ticker_sentiment.get('relevance_score', '0'))
                        sentiment_scores.append(sentiment_score)
                        
                        if len(relevant_articles) < 15:
                            relevant_articles.append({
                                'title': item.get('title', 'Untitled'),
                                'source': item.get('source', 'Unknown'),
                                'published': item.get('time_published', 'Unknown'),
                                'overall_sentiment': item.get('overall_sentiment_label', 'Neutral'),
                                'ticker_sentiment_score': sentiment_score,
                                'ticker_sentiment_label': sentiment_label,
                                'relevance_score': relevance_score,
                                'summary': item.get('summary', '')[:200]  # First 200 chars
                            })

        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        
        # Categorize sentiment distribution
        bullish = sum(1 for s in sentiment_scores if s > 0.15)
        bearish = sum(1 for s in sentiment_scores if s < -0.15)
        neutral = len(sentiment_scores) - bullish - bearish

        prompt = f"""You are a sentiment analysis expert. Analyze the following ACTUAL news sentiment data for {ticker}.

HYPOTHESIS: {state['hypothesis']}

SENTIMENT STATISTICS:
- Total relevant articles analyzed: {article_count}
- Average sentiment score: {avg_sentiment:.3f} (range: -1 to 1, where -1 is very bearish, 0 is neutral, +1 is very bullish)
- Bullish articles (>0.15): {bullish}
- Bearish articles (<-0.15): {bearish}
- Neutral articles: {neutral}

DETAILED ARTICLE ANALYSIS (Top {len(relevant_articles)} most relevant):
{json.dumps(relevant_articles, indent=2)}

Based on this ACTUAL sentiment data, analyze:
1. Overall market sentiment trend
2. Key themes in recent news
3. Sentiment momentum (recent vs older articles)
4. Quality and relevance of sources
5. Potential catalysts or concerns mentioned

You MUST respond with ONLY a valid JSON object (no markdown, no code blocks) with this structure:

{{
    "sentiment_score": {avg_sentiment:.3f},
    "confidence": "high/medium/low",
    "article_count": {article_count},
    "key_themes": ["specific theme 1 from articles", "specific theme 2", "specific theme 3"],
    "market_sentiment": "bullish/bearish/neutral",
    "risk_factors": ["specific risk from news 1", "specific risk from news 2"],
    "sample_headlines": ["{relevant_articles[0]['title'] if relevant_articles else 'No articles'}", "{relevant_articles[1]['title'] if len(relevant_articles) > 1 else 'N/A'}"],
    "hypothesis_support": "supports/opposes/neutral",
    "reasoning": "detailed analysis based on actual article content and sentiment scores"
}}"""

        # Get Gemini analysis
        response = call_gemini(prompt, model="gemini-2.5-flash")
        
        # Parse the response
        try:
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith('```'):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            analysis = json.loads(cleaned_response)
            # Store summary stats
            analysis['sentiment_distribution'] = {
                'bullish': bullish,
                'bearish': bearish,
                'neutral': neutral,
                'average_score': avg_sentiment
            }
        except Exception as parse_error:
            print(f"Error parsing Gemini response: {parse_error}")
            analysis = {
                "sentiment_score": avg_sentiment,
                "confidence": "medium",
                "article_count": article_count,
                "key_themes": ["Error parsing themes"],
                "market_sentiment": "neutral" if abs(avg_sentiment) < 0.15 else ("bullish" if avg_sentiment > 0 else "bearish"),
                "risk_factors": ["Parsing error occurred"],
                "sample_headlines": [a['title'] for a in relevant_articles[:2]],
                "hypothesis_support": "inconclusive",
                "reasoning": f"Error parsing response: {str(parse_error)}"
            }
            
    except Exception as e:
        print(f"Warning: Error in sentiment analysis - {str(e)}")
        analysis = {
            "sentiment_score": 0,
            "confidence": "low",
            "article_count": 0,
            "error": str(e)
        }
    
    return {"sentiment_analysis": analysis}

def risk_volatility_agent(state: AgentState):
    print("--- Running Risk/Volatility Agent ---")
    ticker = state['parsed_hypothesis']['ticker']
    
    try:
        market_state = state.get('market_analysis', {})
        fundamental_state = state.get('fundamental_analysis', {})
        sentiment_state = state.get('sentiment_analysis', {})
        
        # Create prompt for Gemini
        prompt = f"""You are a risk analysis expert. Synthesize the following analyses for {ticker} to assess overall risk.

HYPOTHESIS: {state['hypothesis']}

MARKET ANALYSIS:
{json.dumps(market_state, indent=2)}

FUNDAMENTAL ANALYSIS:
{json.dumps(fundamental_state, indent=2)}

SENTIMENT ANALYSIS:
{json.dumps(sentiment_state, indent=2)}

Based on ALL the actual data from these analyses, evaluate:
1. Price volatility and technical risk
2. Financial stability and fundamental risk
3. Market sentiment risk
4. Overall risk-reward profile
5. Probability of achieving the hypothesis target

You MUST respond with ONLY a valid JSON object (no markdown, no code blocks) with this structure:

{{
    "volatility_index": 0.6,
    "risk_level": "high/medium/low",
    "confidence_interval": "±X%",
    "risk_factors": ["specific risk from analysis 1", "specific risk 2", "specific risk 3"],
    "market_conditions": "favorable/neutral/unfavorable",
    "probability_metrics": {{
        "upside_probability": 0.55,
        "downside_probability": 0.45,
        "volatility_trend": "increasing/stable/decreasing"
    }},
    "hypothesis_support": "supports/opposes/neutral",
    "reasoning": "synthesized risk assessment based on all three analyses"
}}"""

        # Get Gemini analysis
        response = call_gemini(prompt, model="gemini-2.5-flash")
        
        # Parse the response
        try:
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith('```'):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            analysis = json.loads(cleaned_response)
        except Exception as parse_error:
            print(f"Error parsing Gemini response: {parse_error}")
            analysis = {
                "volatility_index": 0.5,
                "risk_level": "medium",
                "confidence_interval": "±5%",
                "risk_factors": ["Parsing error"],
                "market_conditions": "uncertain",
                "probability_metrics": {
                    "upside_probability": 0.5,
                    "downside_probability": 0.5,
                    "volatility_trend": "stable"
                },
                "hypothesis_support": "inconclusive",
                "reasoning": f"Error parsing response: {str(parse_error)}"
            }
            
    except Exception as e:
        print(f"Warning: Error in risk analysis - {str(e)}")
        analysis = {
            "volatility_index": 0.5,
            "risk_level": "unknown",
            "error": str(e)
        }
    
    return {"risk_analysis": analysis}

def aggregator_agent(state: AgentState):
    print("--- Running Aggregator Agent ---")
    
    try:
        market = state.get('market_analysis', {})
        fundamental = state.get('fundamental_analysis', {})
        sentiment = state.get('sentiment_analysis', {})
        risk = state.get('risk_analysis', {})
        
        # Create prompt for Gemini
        prompt = f"""You are a senior investment analyst. Analyze and synthesize the following multi-factor analysis for {state['parsed_hypothesis']['ticker']} to evaluate the given hypothesis.

HYPOTHESIS: {state['hypothesis']}

MARKET ANALYSIS:
{json.dumps(market, indent=2)}

FUNDAMENTAL ANALYSIS:
{json.dumps(fundamental, indent=2)}

SENTIMENT ANALYSIS:
{json.dumps(sentiment, indent=2)}

RISK ANALYSIS:
{json.dumps(risk, indent=2)}

Based on all the analyses, provide a comprehensive evaluation. You MUST respond with ONLY a valid JSON object (no markdown, no code blocks) with the following structure:

{{
    "final_verdict": "Strong Support/Moderate Support/Neutral/Moderate Opposition/Strong Opposition",
    "overall_confidence": 0.75,
    "factor_contributions": {{
        "market": 0.7,
        "fundamental": 0.8,
        "sentiment": 0.6,
        "risk": 0.65
    }},
    "supporting_factors": ["specific factor 1 from analyses", "specific factor 2", "specific factor 3"],
    "opposing_factors": ["specific factor 1 from analyses", "specific factor 2"],
    "risk_assessment": "detailed risk assessment synthesizing all risk factors",
    "time_horizon_analysis": "analysis of whether time horizon is realistic given data",
    "recommendations": ["actionable recommendation 1", "actionable recommendation 2", "actionable recommendation 3"],
    "detailed_reasoning": "comprehensive reasoning that synthesizes all factors with specific data points"
}}"""

        # Get Gemini analysis
        response = call_gemini(prompt, model="gemini-2.5-flash")
        
        # Parse the response
        try:
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith('```'):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            analysis = json.loads(cleaned_response)
            
            # Add raw data to the analysis
            analysis["raw_data"] = {
                "market": market,
                "fundamental": fundamental,
                "sentiment": sentiment,
                "risk": risk
            }
            
            # Add data quality metrics
            analysis["data_quality"] = {
                "market": "error" not in market,
                "fundamental": "error" not in fundamental,
                "sentiment": "error" not in sentiment,
                "risk": "error" not in risk
            }
            
        except Exception as parse_error:
            print(f"Error parsing Gemini response: {parse_error}")
            # Fallback if JSON parsing fails
            analysis = {
                "final_verdict": "Analysis inconclusive due to processing error",
                "overall_confidence": 0.5,
                "factor_contributions": {
                    "market": market.get('momentum_strength', 0.5) if 'error' not in market else 0.3,
                    "fundamental": fundamental.get('profitability_support', 0.5) if 'error' not in fundamental else 0.3,
                    "sentiment": (sentiment.get('sentiment_score', 0) + 1) / 2 if 'error' not in sentiment else 0.5,
                    "risk": 1 - risk.get('volatility_index', 0.5) if 'error' not in risk else 0.5
                },
                "supporting_factors": ["Data collected from multiple sources"],
                "opposing_factors": ["Analysis processing encountered errors"],
                "risk_assessment": "Manual review recommended due to processing errors",
                "time_horizon_analysis": "Unable to process time horizon analysis",
                "recommendations": [
                    "Review collected data manually",
                    "Consult with financial advisor",
                    "Monitor market conditions closely"
                ],
                "detailed_reasoning": f"Error in processing LLM response: {str(parse_error)}",
                "raw_data": {
                    "market": market,
                    "fundamental": fundamental,
                    "sentiment": sentiment,
                    "risk": risk
                },
                "error": "Failed to parse LLM response"
            }
            
    except Exception as e:
        print(f"Warning: Error in aggregator analysis - {str(e)}")
        market = state.get('market_analysis', {})
        fundamental = state.get('fundamental_analysis', {})
        sentiment = state.get('sentiment_analysis', {})
        risk = state.get('risk_analysis', {})
        
        analysis = {
            "final_verdict": "Error in analysis aggregation",
            "error": str(e),
            "raw_data": {
                "market": market,
                "fundamental": fundamental,
                "sentiment": sentiment,
                "risk": risk
            }
        }
    
    return {"final_report": analysis}

# --- Graph Definition ---

def run_analysis_workflow(state: AgentState) -> AgentState:
    # Run each agent in sequence
    parsed_state = parsing_agent(state)
    state.update(parsed_state)
    
    market_state = market_momentum_agent(state)
    state.update(market_state)
    
    fundamental_state = fundamental_analysis_agent(state)
    state.update(fundamental_state)
    
    sentiment_state = sentiment_agent(state)
    state.update(sentiment_state)
    
    risk_state = risk_volatility_agent(state)
    state.update(risk_state)
    
    final_state = aggregator_agent(state)
    state.update(final_state)
    
    return state

app = run_analysis_workflow

# --- Main Execution ---

if __name__ == "__main__":
    print("=" * 70)
    print(" RAVEN Multi-Agent System: Hypothesis Validation")
    print(" Powered by Google Gemini 2.5 Flash")
    print("=" * 70)
    print()
    
    # Get inputs from the user
    ticker = input("Enter the stock ticker (e.g., AAPL): ").upper()
    metric = input("Enter the metric to analyze (e.g., stock price): ")
    target_movement = input("Enter the target movement (e.g., surge by 10%): ")
    time_horizon = input("Enter the time horizon (e.g., in the next quarter): ")

    user_hypothesis = f"The hypothesis is that {ticker}'s {metric} will {target_movement} {time_horizon}."
    print(f"\n📋 Analyzing hypothesis: {user_hypothesis}\n")
    
    initial_state = {
        "hypothesis": user_hypothesis,
        "parsed_hypothesis": {
            "ticker": ticker,
            "metric": metric,
            "target": target_movement,
            "time_horizon": time_horizon
        },
        "market_analysis": {},
        "fundamental_analysis": {},
        "sentiment_analysis": {},
        "risk_analysis": {},
        "final_report": {}
    }
    
    print("🚀 Starting RAVEN Multi-Agent System...\n")
    final_state = app(initial_state)
    
    if 'final_report' in final_state and final_state['final_report']:
        report = final_state['final_report']
        
        print("\n\n" + "="*70)
        print(" FINAL REPORT")
        print("="*70)
        print(f"Hypothesis: {user_hypothesis}")
        print("-" * 70)
        
        print(f"\n📊 Final Verdict: {report.get('final_verdict', 'Not available')}")
        print(f"🎯 Overall Confidence: {report.get('overall_confidence', 'Not available')}")
        
        if 'factor_contributions' in report:
            print("\n📈 Factor Contributions:")
            for factor, score in report['factor_contributions'].items():
                bar = "█" * int(score * 20)
                print(f"  {factor.capitalize():12} [{score:.2f}] {bar}")
        
        if 'supporting_factors' in report and report['supporting_factors']:
            print("\n✅ Supporting Factors:")
            for factor in report['supporting_factors']:
                print(f"  + {factor}")
        
        if 'opposing_factors' in report and report['opposing_factors']:
            print("\n❌ Opposing Factors:")
            for factor in report['opposing_factors']:
                print(f"  - {factor}")
        
        if 'risk_assessment' in report:
            print(f"\n⚠️  Risk Assessment:\n  {report['risk_assessment']}")
        
        if 'time_horizon_analysis' in report:
            print(f"\n⏰ Time Horizon Analysis:\n  {report['time_horizon_analysis']}")
        
        if 'recommendations' in report and report['recommendations']:
            print("\n💡 Recommendations:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        if 'detailed_reasoning' in report:
            print(f"\n📝 Detailed Reasoning:\n{report['detailed_reasoning']}")
        
        if 'data_quality' in report:
            print("\n✓ Data Quality:")
            for source, quality in report['data_quality'].items():
                status = "✓ Complete" if quality else "✗ Incomplete"
                print(f"  {source.capitalize():12} {status}")
        
        if 'error' in report:
            print("\n⚠️  Errors Encountered:")
            print(f"  {report['error']}")
        
        print("\n" + "="*70)
        print(" Analysis Complete")
        print("="*70)
            
    else:
        print("\n--- Execution Finished ---")
        print("Could not generate final report. Final state:")
        print(json.dumps(final_state, indent=2, default=str))