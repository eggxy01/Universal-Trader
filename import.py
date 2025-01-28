import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import plotly.graph_objs as go
import threading
import time

# Set page config at the top
st.set_page_config(page_title="Market Data and Forecasting", layout="wide")

# Global variables for data and model
data = None
refresh_data_event = threading.Event()

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

# Function to train a simple model on the fetched data
@st.cache_resource
def train_model(data):
    data['Date'] = data.index.map(pd.Timestamp.toordinal)  # Convert dates to ordinal for modeling
    X = data[['Date']]
    y = data['Close']
    model = LinearRegression()
    model.fit(X, y)
    return model

# Function to predict future prices
def predict_future_prices(model, last_date, days=1):
    next_days = np.array([[pd.Timestamp(last_date + pd.Timedelta(days=i)).toordinal()] for i in range(1, days + 1)])
    predicted_prices = model.predict(next_days)
    return predicted_prices

# Function to calculate EMA
def calculate_ema(data, span):
    data['EMA'] = data['Close'].ewm(span=span, adjust=False).mean()  # Add EMA directly to DataFrame
    return data

# Function to calculate signals based on EMA crossover
def calculate_signals(data):
    data['Signal'] = 0  # Default to no signal
    data['Signal'][1:] = np.where(data['Close'][1:] > data['EMA'][1:], 1, 0)  # 1 for buy signal
    data['Position'] = data['Signal'].diff()  # Position changes indicate buy (1) or sell (-1)
    return data

# Function to identify reversal patterns
def detect_reversal_patterns(data):
    data['Reversal'] = 0  # Default to no reversal
    # Identify double tops and bottoms
    for i in range(2, len(data) - 2):
        if data['Close'][i] > data['Close'][i - 1] and data['Close'][i] > data['Close'][i + 1]:  # Double top
            data['Reversal'][i] = -1  # Sell signal
        elif data['Close'][i] < data['Close'][i - 1] and data['Close'][i] < data['Close'][i + 1]:  # Double bottom
            data['Reversal'][i] = 1  # Buy signal
    return data

# Function to render the main page content
def main_page():
    st.title("Market Data and Forecasting Dashboard")

    # Sidebar for user input
    st.sidebar.header("Market Selection")
    market_symbols = list(symbol_mapping.keys())
    selected_market = st.sidebar.selectbox("Select Market Symbol", market_symbols)

    interval = st.sidebar.selectbox("Select Interval", ['1m', '5m', '15m', '1h', '1d'])
    forecast_days = st.sidebar.number_input("Number of Days to Forecast", min_value=1, max_value=30, value=1)
    ema_span = st.sidebar.number_input("EMA Span", min_value=1, max_value=50, value=10)

    # Fetching the selected market data initially
    global data
    data = fetch_market_data(symbol_mapping[selected_market], interval)

    if data is not None:
        st.subheader(f"Historical Market Data for {selected_market}")
        st.write(data)

        # Train the model on the fetched data
        model = train_model(data)
        st.success("Initial model trained successfully!")

        # Calculate EMA
        data = calculate_ema(data, ema_span)  # Ensure EMA is calculated and added to the DataFrame

        # Calculate buy/sell signals based on EMA crossover
        data = calculate_signals(data)

        # Detect reversal patterns
        data = detect_reversal_patterns(data)

        # Prepare data for the chart
        last_date = data.index[-1]
        future_prices = predict_future_prices(model, last_date, forecast_days)

        # Create a DataFrame for future predictions
        future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, forecast_days + 1)]
        future_data = pd.DataFrame(data=future_prices, index=future_dates, columns=['Predicted Close'])

        # Arrange charts side by side using columns
        col1, col2, col3 = st.columns([1, 1, 1])  # Make columns equal width

        # Candlestick Chart with Y-axis Zoom
        with col1:
            st.subheader("Candlestick Chart with Y-Axis Zoom")
            candlestick_fig = go.Figure(data=[go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
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
        with col2:
            st.subheader("Closing Price & EMA")
            fig1 = go.Figure()
            fig1.add_trace(
                go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close', line=dict(color='blue')) )
            fig1.add_trace(go.Scatter(x=data.index, y=data['EMA'], mode='lines', name='EMA', line=dict(color='orange')) )
            fig1.update_layout(title=f'{selected_market} Closing Prices with EMA', xaxis_title='Date',
                               yaxis_title='Price', template='plotly_dark')
            st.plotly_chart(fig1, use_container_width=True)

        # Future Price Prediction Chart
        with col3:
            st.subheader("Future Price Prediction")
            fig2 = go.Figure()
            fig2.add_trace(
                go.Scatter(x=future_data.index, y=future_data['Predicted Close'], mode='lines', name='Predicted Close',
                           line=dict(color='red')) )
            fig2.update_layout(title=f'{selected_market} Predicted Prices', xaxis_title='Date', yaxis_title='Price',
                               template='plotly_dark')
            st.plotly_chart(fig2, use_container_width=True)

        # Combined Buy/Sell and Reversal Signals Chart
        with st.expander("Combined Buy/Sell and Reversal Signals"):
            combined_signal_fig = go.Figure()
            combined_signal_fig.add_trace(
                go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close', line=dict(color='blue')) )
            combined_signal_fig.add_trace(go.Scatter(x=data.index[data['Position'] == 1],
                                                     y=data['Close'][data['Position'] == 1],
                                                     mode='markers',
                                                     name='Buy Signal',
                                                     marker=dict(color='green', size=10, symbol='triangle-up')) )
            combined_signal_fig.add_trace(go.Scatter(x=data.index[data['Position'] == -1],
                                                     y=data['Close'][data['Position'] == -1],
                                                     mode='markers',
                                                     name='Sell Signal',
                                                     marker=dict(color='red', size=10, symbol='triangle-down')) )
            combined_signal_fig.add_trace(go.Scatter(x=data.index[data['Reversal'] == 1],
                                                     y=data['Close'][data['Reversal'] == 1],
                                                     mode='markers',
                                                     name='Reversal Buy',
                                                     marker=dict(color='purple', size=10, symbol='x')) )
            combined_signal_fig.add_trace(go.Scatter(x=data.index[data['Reversal'] == -1],
                                                     y=data['Close'][data['Reversal'] == -1],
                                                     mode='markers',
                                                     name='Reversal Sell',
                                                     marker=dict(color='orange', size=10, symbol='x')) )
            combined_signal_fig.update_layout(title='Combined Signals', xaxis_title='Date', yaxis_title='Price',
                                               template='plotly_dark')
            st.plotly_chart(combined_signal_fig, use_container_width=True)

    # Terms and conditions (HTML section)
    with st.expander("Terms and Conditions"):
        html_code = """
        <h2>Terms and Conditions</h2>
        <p>By using this platform, you agree to the following terms:</p>
        <ul>
            <li>All market data is for informational purposes only.</li>
            <li>Trading involves risks, and past performance is not indicative of future results.</li>
            <li>We do not provide financial advice.</li>
        </ul>
        <p>Please read carefully before proceeding with any trading activity.</p>
        """
        st.markdown(html_code, unsafe_allow_html=True)

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
