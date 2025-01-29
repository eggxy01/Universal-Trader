import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import plotly.graph_objs as go
import threading
import time

# Set page config at the top
st.set_page_config(page_title="Automated Trading System", layout="wide")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'trades' not in st.session_state:
    st.session_state.trades = []

# Global variables for data and model
refresh_data_event = threading.Event()

# Initialize Total balance and open positions
Total_balance = 9753420.0  # Starting demo balance
open_positions = []  # List to keep track of open positions
transaction_history = []  # List to store transaction history

# Symbol mapping for markets and forex pairs
symbol_mapping = {
    'NIFTY 50': '^NSEI',
    'BANKNIFTY': '^NSEBANK',
    'MIDCAP NIFTY': '^NSEMDCP',
    'FINNIFTY': '^NSEFIN',
    'EUR/USD': 'EURUSD=X',
    'USD/JPY': 'JPY=X',
    'GBP/USD': 'GBPUSD=X',
    'USD/CHF': 'CHF=X',
    'AUD/USD': 'AUDUSD=X',
    'USD/CAD': 'CAD=X',
    'NZD/USD': 'NZDUSD=X',
    'S&P 500': '^GSPC',
    'Dow Jones': '^DJI',
    'NASDAQ': '^IXIC',
    'Gold': 'GC=F',
    'Silver': 'SI=F',
    'Oil': 'CL=F',
}

# Function to fetch market data
@st.cache_data(ttl=60)
def fetch_market_data(symbol, interval, period='1mo'):
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        return data if not data.empty else None
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

# Function to calculate EMA
def calculate_ema(data, span):
    data['EMA'] = data['Close'].ewm(span=span, adjust=False).mean()
    return data

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    data['RSI'] = rsi
    return data

# Function to calculate MACD
def calculate_macd(data):
    data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    return data

# Function to calculate signals based on EMA crossover
def calculate_signals(data):
    data['Signal'] = 0
    data['Signal'][1:] = np.where(data['Close'][1:] > data['EMA'][1:], 1, -1)
    data['Position'] = data['Signal'].diff()
    return data

# Function to detect reversal patterns
def detect_reversal_patterns(data):
    data['Reversal'] = 0
    for i in range(2, len(data) - 2):
        if data['Close'][i] > data['Close'][i - 1] and data['Close'][i] > data['Close'][i + 1]:
            data['Reversal'][i] = -1  # Potential sell signal
        elif data['Close'][i] < data['Close'][i - 1] and data['Close'][i] < data['Close'][i + 1]:
            data['Reversal'][i] = 1  # Potential buy signal
    return data

# Function to simulate trades
def simulate_trades(data, initial_capital=10000, risk_per_trade=0.01):
    capital = initial_capital
    trades = []
    position = None
    entry_price = 0

    for i in range(len(data)):
        if data['Position'][i] == 2:  # Buy signal
            if position != 'long':
                entry_price = data['Close'][i]
                position = 'long'
                stop_loss = entry_price * (1 - risk_per_trade)
                take_profit = entry_price * (1 + 2 * risk_per_trade)  # Risk-reward ratio of 1:2
                trades.append({'type': 'buy', 'price': entry_price, 'stop_loss': stop_loss, 'take_profit': take_profit, 'time': data.index[i]})
        elif data['Position'][i] == -2:  # Sell signal
            if position != 'short':
                entry_price = data['Close'][i]
                position = 'short'
                stop_loss = entry_price * (1 + risk_per_trade)
                take_profit = entry_price * (1 - 2 * risk_per_trade)
                trades.append({'type': 'sell', 'price': entry_price, 'stop_loss': stop_loss, 'take_profit': take_profit, 'time': data.index[i]})

        # Check if stop-loss or take-profit is hit
        if position == 'long' and (data['Low'][i] <= stop_loss or data['High'][i] >= take_profit):
            exit_price = stop_loss if data['Low'][i] <= stop_loss else take_profit
            capital = capital * (exit_price / entry_price)
            position = None
        elif position == 'short' and (data['High'][i] >= stop_loss or data['Low'][i] <= take_profit):
            exit_price = stop_loss if data['High'][i] >= stop_loss else take_profit
            capital = capital * (entry_price / exit_price)
            position = None

    return trades, capital

# Function to render the main page content
def main_page():
    st.title("Automated Trading System")

    # Live Balance and Open Trades Display
    st.sidebar.header("Account Information")
    st.sidebar.write(f"**Total Balance:** ${Total_balance:.2f}")
    st.sidebar.write("### Open Trades")
    if open_positions:
        for pos in open_positions:
            st.sidebar.write(f"{pos['action']} {pos['amount']} of {pos['symbol']} at {pos['price']:.2f}")
    else:
        st.sidebar.write("No open positions.")

    # Display total P&L
    total_pnl = sum((pos['current_price'] - pos['price']) * pos['amount'] for pos in open_positions)
    st.sidebar.write(f"**Total P&L:** ${total_pnl:.2f}")

    # Sidebar for user input
    st.sidebar.header("Trading Parameters")
    market_symbols = list(symbol_mapping.keys())
    selected_market = st.sidebar.selectbox("Select Market Symbol", market_symbols)
    interval = st.sidebar.selectbox("Select Interval", ['5m', '15m', '1h', '1d'])
    ema_span = st.sidebar.number_input("EMA Span", min_value=1, max_value=50, value=10)
    risk_per_trade = st.sidebar.number_input("Risk per Trade (%)", min_value=0.1, max_value=5.0, value=1.0) / 100
    initial_capital = st.sidebar.number_input("Initial Capital", min_value=1000, max_value=100000, value=10000)

    # Fetching the selected market data
    with st.spinner("Fetching market data..."):
        st.session_state.data = fetch_market_data(symbol_mapping[selected_market], interval)

    if st.session_state.data is not None:
        st.subheader(f"Historical Market Data for {selected_market}")
        st.write(st.session_state.data)

        # Calculate EMA
        st.session_state.data = calculate_ema(st.session_state.data, ema_span)

        # Calculate RSI
        st.session_state.data = calculate_rsi(st.session_state.data)

        # Calculate MACD
        st.session_state.data = calculate_macd(st.session_state.data)

        # Calculate buy/sell signals based on EMA crossover
        st.session_state.data = calculate_signals(st.session_state.data)

        # Detect reversal patterns
        st.session_state.data = detect_reversal_patterns(st.session_state.data)

        # Simulate trades
        trades, final_capital = simulate_trades(st.session_state.data, initial_capital, risk_per_trade)
        st.session_state.trades = trades

        # Display trading results
        st.subheader("Trading Results")
        st.write(f"Initial Capital: ${initial_capital:,.2f}")
        st.write(f"Final Capital: ${final_capital:,.2f}")
        st.write(f"Total Trades: {len(trades)}")

        # Display trades in a table
        st.write("Trade History:")
        st.write(pd.DataFrame(trades))

        # Plot trades on the chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=st.session_state.data.index, y=st.session_state.data['Close'], mode='lines', name='Close', line=dict(color='blue')))
        for trade in trades:
            if trade['type'] == 'buy':
                fig.add_trace(go.Scatter(x=[trade['time']], y=[trade['price']], mode='markers', name='Buy', marker=dict(color='green', size=10, symbol='triangle-up')))
            elif trade['type'] == 'sell':
                fig.add_trace(go.Scatter(x=[trade['time']], y=[trade['price']], mode='markers', name='Sell', marker=dict(color='red', size=10, symbol='triangle-down')))
        fig.update_layout(title=f'{selected_market} Trades', xaxis_title='Date', yaxis_title='Price', template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

        # Candlestick Chart with Y-axis Zoom
        st.subheader("Candlestick Chart with Y-Axis Zoom")
        candlestick_fig = go.Figure(data=[go.Candlestick(
            x=st.session_state.data.index,
            open=st.session_state.data['Open'],
            high=st.session_state.data['High'],
            low=st.session_state.data['Low'],
            close=st.session_state.data['Close'],
            name='Candlestick'
        )])
        candlestick_fig.update_layout(
            title=f'{selected_market} Candlestick Chart',
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_dark',
            xaxis=dict(rangeslider=dict(visible=True), type="date"),
            yaxis=dict(fixedrange=False)
        )
        st.plotly_chart(candlestick_fig, use_container_width=True)

        # Closing Price & EMA Chart
        st.subheader("Closing Price & EMA")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=st.session_state.data.index, y=st.session_state.data['Close'], mode='lines', name='Close', line=dict(color='blue')))
        fig1.add_trace(go.Scatter(x=st.session_state.data.index, y=st.session_state.data['EMA'], mode='lines', name='EMA', line=dict(color='orange')))
        fig1.update_layout(title=f'{selected_market} Closing Prices with EMA', xaxis_title='Date', yaxis_title='Price', template='plotly_dark')
        st.plotly_chart(fig1, use_container_width=True)

        # Combined Buy/Sell and Reversal Signals Chart
        st.subheader("Combined Buy/Sell and Reversal Signals")
        combined_signal_fig = go.Figure()
        combined_signal_fig.add_trace(go.Scatter(x=st.session_state.data.index, y=st.session_state.data['Close'], mode='lines', name='Close', line=dict(color='blue')))
        combined_signal_fig.add_trace(go.Scatter(x=st.session_state.data.index[st.session_state.data['Position'] == 1],
                                                 y=st.session_state.data['Close'][st.session_state.data['Position'] == 1],
                                                 mode='markers',
                                                 name='Buy Signal',
                                                 marker=dict(color='green', size=10, symbol='triangle-up')))
        combined_signal_fig.add_trace(go.Scatter(x=st.session_state.data.index[st.session_state.data['Position'] == -1],
                                                 y=st.session_state.data['Close'][st.session_state.data['Position'] == -1],
                                                 mode='markers',
                                                 name='Sell Signal',
                                                 marker=dict(color='red', size=10, symbol='triangle-down')))
        combined_signal_fig.add_trace(go.Scatter(x=st.session_state.data.index[st.session_state.data['Reversal'] == 1],
                                                 y=st.session_state.data['Close'][st.session_state.data['Reversal'] == 1],
                                                 mode='markers',
                                                 name='Reversal Buy',
                                                 marker=dict(color='purple', size=10, symbol='x')))
        combined_signal_fig.add_trace(go.Scatter(x=st.session_state.data.index[st.session_state.data['Reversal'] == -1],
                                                 y=st.session_state.data['Close'][st.session_state.data['Reversal'] == -1],
                                                 mode='markers',
                                                 name='Reversal Sell',
                                                 marker=dict(color='orange', size=10, symbol='x')))
        combined_signal_fig.update_layout(title='Combined Signals', xaxis_title='Date', yaxis_title='Price', template='plotly_dark')
        st.plotly_chart(combined_signal_fig, use_container_width=True)

# Function to handle real-time data updates
def refresh_data():
    global data
    while not refresh_data_event.is_set():
        time.sleep(1)  # Pause for a second
        for symbol in symbol_mapping.values():
            new_data = fetch_market_data(symbol, '1m')  # Adjust interval as needed
            if new_data is not None:
                data = new_data

# Create a thread for data refreshing
data_refresh_thread = threading.Thread(target=refresh_data)
data_refresh_thread.start()

# Run the main page function
main_page()

# Stop the data refresh thread when the app ends
refresh_data_event.set()
data_refresh_thread.join()
