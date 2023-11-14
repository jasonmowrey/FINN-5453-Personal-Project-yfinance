import dash
from dash import dcc, html, Input, Output
from dash.dependencies import Input, Output, State
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from functools import lru_cache
import logging
from concurrent.futures import ThreadPoolExecutor
# from alpha_vantage.fundamentaldata import FundamentalData
# import numpy as np

# Initialize the app
app = dash.Dash(__name__)

logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_sp500_tickers():
    tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df = tables[0]
    tickers = df['Symbol'].tolist()
    tickers = [ticker.replace('.', '-') for ticker in tickers]
    return tickers

sp500_tickers = fetch_sp500_tickers()

fetched_data = None

# ALPHA_VANTAGE_API_KEY = "QBPPPM2BFLOGL82R"
# fd = FundamentalData(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')

@lru_cache(maxsize=10)
def fetch_stock_data_for_ticker(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Calculate Debt/Equity Ratio
        balance_sheet = stock.balance_sheet
        total_liabilities = balance_sheet.loc['Total Liabilities Net Minority Interest'].iloc[0] if 'Total Liabilities Net Minority Interest' in balance_sheet.index else 0
        total_equity = balance_sheet.loc['Stockholders Equity'].iloc[0] if 'Stockholders Equity' in balance_sheet.index else 0
        debt_equity_ratio = total_liabilities / total_equity if total_equity != 0 else None
                
        # Calculate Earnings Yield
        pe_ratio = info.get("trailingPE", None)
        earnings_yield = 1 / pe_ratio if pe_ratio and pe_ratio != 0 else None
        
        # Get PEG Ratio
        peg_ratio = info.get("pegRatio", None)
        
        # Calculate Free Cash Flow Yield
        total_debt = info.get("totalDebt", 0)
        cash_equivalents = info.get("cash", 0)
        market_cap = info.get("marketCap", 0)
        enterprise_value = market_cap + total_debt - cash_equivalents
        free_cash_flow = info.get("freeCashflow", 0)
        free_cash_flow_yield = free_cash_flow / enterprise_value if enterprise_value != 0 else None
        
        # Calculate Operating Margin
        income_statement = stock.financials
        # Fetch Operating Income and Revenue for the most recent year
        operating_income = float(income_statement.loc['Operating Income'].iloc[0])
        revenue = float(income_statement.loc['Total Revenue'].iloc[0])
        if revenue != 0:
            operating_margin = (operating_income / revenue) * 100
        else:
            operating_margin = None
            
        #Calculating Alpha
        # Getting historical data for the stock and market (S&P 500)
        stock_data = stock.history(period="1y")  # Change the period as needed
        market_data = yf.Ticker("^GSPC").history(period="1y")  # S&P 500 data
        # Calculating annual returns
        stock_return = (stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[0]) - 1
        market_return = (market_data['Close'].iloc[-1] / market_data['Close'].iloc[0]) - 1
        # Assume a constant risk-free rate of 3% (or 0.03)
        risk_free_rate = 0.03
        # Fetch Beta for the stock
        beta = info.get("beta", 1)  # Using 1 as a default beta
        # Calculating Alpha
        alpha = stock_return - (risk_free_rate + beta * (market_return - risk_free_rate))  
        
        # Calculate Sharpe Ratio
        # Fetch historical data for the stock
        stock_data = stock.history(period="1y")  # Change the period as needed
        # Calculate daily returns
        stock_data['Daily Return'] = stock_data['Close'].pct_change(1)
        # Calculate average daily return and daily standard deviation
        avg_daily_return = stock_data['Daily Return'].mean()
        std_dev_return = stock_data['Daily Return'].std()
        # Assume a constant risk-free rate of 3% per annum (or 0.03)
        daily_risk_free_rate = (1 + 0.03) ** (1/252) - 1
        # Calculate the Sharpe Ratio
        sharpe_ratio = (avg_daily_return - daily_risk_free_rate) / std_dev_return
        
        data = {
            "Ticker": ticker,
            "Price": info.get("currentPrice", None),
            "P/E Ratio": info.get("trailingPE", None),
            "PEG Ratio": peg_ratio,
            "EPS": info.get("trailingEps", None),
            "Dividend Yield": info.get("dividendYield", None),
            "P/B Ratio": info.get("priceToBook", None),
            "Return on Equity": info.get("returnOnEquity", None),
            "Current Ratio": info.get("currentRatio", None),
            "Revenue Growth": info.get("revenueGrowth", None),
            "Free Cash Flow Yield": free_cash_flow_yield,
            "Operating Margin": operating_margin,
            "P/S Ratio": info.get("priceToSalesTrailing12Months", None),
            "Earnings Yield": earnings_yield,
            "Quick Ratio": info.get("quickRatio", None),
            "Debt/Equity Ratio": debt_equity_ratio,
            "Alpha": alpha,
            "Beta": info.get("beta", None),
            "Sharpe Ratio": sharpe_ratio,
            "Market Cap": info.get("marketCap", None),
        }
        return data
    except Exception as e:
        logging.error(f"Error with ticker {ticker}: {e}")
        return None

def fetch_stock_data_with_metrics():
    global fetched_data
    if fetched_data is not None:
        return fetched_data
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(fetch_stock_data_for_ticker, sp500_tickers))
    data = [result for result in results if result is not None]
    fetched_data = pd.DataFrame(data)
    return fetched_data

# Define layout
def create_field(label, id, value, type='number'):
    return html.Div([
        html.Label(label, style={'color': 'white'}),
        dcc.Input(id=id, value=value, type=type, style={'color': 'black', 'backgroundColor': 'white'}),
        dcc.Checklist(
            id=f'check-{id}',
            options=[{'label': '', 'value': 'on'}],
            value=[],
            inline=True,
            style={'display': 'inline-block', 'margin-left': '10px'}
        )
    ], style={'width': '48%', 'display': 'inline-block'})

fields = [
    ('Min Price:', 'minPrice', 100),
    ('Max Price:', 'maxPrice', 200),
    ('Max P/E Ratio:', 'trailingPE', 20),
    ('Max PEG Ratio', 'pegRatio', 1),
    ('Min EPS:', 'trailingEps', 2),
    ('Min Dividend Yield:', 'dividendYield', .025), # Use decimal format
    ('Min P/B Ratio:', 'priceToBook', 1),
    ('Min Return on Equity:', 'returnOnEquity', 0.15), # Use decimal format
    ('Min Current Ratio:', 'currentRatio', 1.2),
    ('Min Revenue Growth:', 'revenueGrowth', 0.05), # Use decimal format
    ('Min Free Cash Flow Yield:', 'freeCashflowYield', 0.025),
    ('Min Operating Margin:', 'operating_margin', 0.15),
    ('Max Price/Sales Ratio:', 'priceToSalesTrailing12Months', 1),
    ('Min Earnings Yield:', 'earningsYield', 0.10),
    ('Min Quick Ratio:', 'quickRatio', 1.5),
    ('Max Debt/Equity Ratio:', 'debt_equity_ratio', 2),
    ('Min Alpha:', 'alpha', 0.10),
    ('Min Beta', 'beta', 1),
    ('Min Sharpe Ratio', 'sharpe_ratio', 1),
    ('Min Market Cap', 'min_market_cap', 1e9),
]

# Define the app layout
app.layout = html.Div([
    html.Div([create_field(*field) for field in fields]),
    html.Button('Screen Stocks', id='screen-button', style={'color': 'black', 'backgroundColor': 'white'}),
    dcc.Loading(
        id="loading",
        type="circle",  
        color="#FFFFFF",  # color of the loading spinner
        style={'margin-top': '-250px'},  # adjust the top margin to move it up
        children=[
            html.Div(id='output-table', style={'color': 'white'}),
            # html.Div(id='hidden-div', style={'display': 'none'})  # Hidden Div to store filtered DataFrame
        ]
    )
], style={'backgroundColor': '#000000', 'textAlign': 'center'})

# Updated callback, without the graph
@app.callback(
    [Output('output-table', 'children')],
    [Input('screen-button', 'n_clicks')],
    [State(field_id, 'value') for _, field_id, _ in fields] +
    [State(f'check-{field_id}', 'value') for _, field_id, _ in fields]  # Capture checkbox states
)
def update_output(n_clicks, *args):
    if n_clicks is None:
        return [dash.no_update]
    n = len(fields)
    values = args[:n]
    checkboxes = args[n:]
    
    df = fetch_stock_data_with_metrics()
    if df is None or df.empty:
        return ['No data to display']
    
    # Create a mapping from the field identifier to the actual DataFrame column name
    field_to_df_column = {
        'minPrice': 'Price',
        'maxPrice': 'Price',
        'trailingPE': 'P/E Ratio',
        'pegRatio': 'PEG Ratio',
        'trailingEps': 'EPS',
        'dividendYield': 'Dividend Yield',
        'priceToBook': 'P/B Ratio',
        'returnOnEquity': 'Return on Equity',
        'currentRatio': 'Current Ratio',
        'revenueGrowth': 'Revenue Growth',
        'freeCashflowYield': 'Free Cash Flow Yield',
        'operating_margin': 'Operating Margin',
        'priceToSalesTrailing12Months': 'P/S Ratio',
        'earningsYield': 'Earnings Yield',
        'quickRatio': 'Quick Ratio',
        'debt_equity_ratio': 'Debt/Equity Ratio',
        'alpha': 'Alpha',
        'beta': 'Beta',
        'sharpe_ratio': 'Sharpe Ratio',
        'min_market_cap': 'Market Cap',
    }
    
    # Loop through all the criteria, and apply the filters only if the checkbox is checked ('on')
    for value, (label, field_id, _), checkbox in zip(values, fields, checkboxes):
        if 'on' not in checkbox:  # Check if checkbox is checked
            continue  # Skip this criterion if not checked
            
        column = field_to_df_column[field_id]
        
        if 'Min' in label:
            df = df.loc[df[column] >= value].copy()
        elif 'Max' in label:
            df = df.loc[df[column] <= value].copy()
    
    # Round all numeric columns to 4 decimal places
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].round(4)
        
    # Create a Plotly Table to Display Data
    fig = go.Figure(data=[go.Table(
            header=dict(values=list(df.columns),
                        fill_color='#f5f5f5',
                        align='center',
                        font=dict(size=12),  # Adjust header font size here
                        line=dict(color='darkslategray', width=1)),  # Lines for header
            cells=dict(values=[df[col] for col in df.columns],
                       fill_color='white',
                       align='center',
                       font=dict(size=10),  # Adjust cell font size here
                       line=dict(color='darkslategray', width=1))  # Lines for cells 
        )])
    fig.update_layout(
        autosize=True,
        margin=dict(l=20, r=20, b=20, t=20)  # Adds a 20-pixel margin on all sides
    )
    
    return [dcc.Graph(figure=fig, style={"height": "calc(100vh - 40px)"})] 

if __name__ == '__main__':
    app.run_server(debug=True)