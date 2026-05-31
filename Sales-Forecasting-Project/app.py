import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
st.set_page_config(
    page_title="Sales Forecast Dashboard",
    layout="wide"
)

st.title("📈 Sales Forecast Dashboard")
# LOAD DATA
df = pd.read_csv(
    "/Users/bhavyasri/Desktop/Sales-Forecasting-Project/dataset/Sample - Superstore.csv",
    encoding='latin1'
)
# Convert date column
df['Order Date'] = pd.to_datetime(df['Order Date'])
# FILTERS
st.sidebar.header("Filters")

region = st.sidebar.selectbox(
    "Select Region",
    ['All'] + list(df['Region'].unique())
)

category = st.sidebar.selectbox(
    "Select Category",
    ['All'] + list(df['Category'].unique())
)

# Apply filters
filtered_df = df.copy()

if region != 'All':
    filtered_df = filtered_df[
        filtered_df['Region'] == region
    ]

if category != 'All':
    filtered_df = filtered_df[
        filtered_df['Category'] == category
    ]

# KPI SECTION
total_sales = filtered_df['Sales'].sum()
avg_sales = filtered_df['Sales'].mean()
max_sales = filtered_df['Sales'].max()
total_orders = filtered_df['Order ID'].nunique()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Sales",
    f"${total_sales:,.0f}"
)

col2.metric(
    "Average Sales",
    f"${avg_sales:,.0f}"
)

col3.metric(
    "Maximum Sales",
    f"${max_sales:,.0f}"
)

col4.metric(
    "Total Orders",
    total_orders
)
# SALES TREND

st.subheader("📊 Historical Sales Trend")

sales_data = filtered_df.groupby(
    'Order Date'
)['Sales'].sum().reset_index()

fig1, ax1 = plt.subplots(figsize=(12,5))

ax1.plot(
    sales_data['Order Date'],
    sales_data['Sales']
)

ax1.set_xlabel("Date")
ax1.set_ylabel("Sales")
ax1.set_title("Sales Trend Over Time")

st.pyplot(fig1)

# MACHINE LEARNING FORECAST

sales_data['Day_Number'] = np.arange(
    len(sales_data)
)

X = sales_data[['Day_Number']]
y = sales_data['Sales']

model = LinearRegression()

model.fit(X, y)

# Future prediction
future_days = pd.DataFrame({
    'Day_Number': np.arange(
        len(sales_data),
        len(sales_data) + 30
    )
})

future_predictions = model.predict(
    future_days
)

# FORECAST GRAPH
st.subheader("🔮 30-Day Sales Forecast")

fig2, ax2 = plt.subplots(figsize=(12,5))

# Historical sales
ax2.plot(
    sales_data['Sales'],
    label='Historical Sales'
)

# Future forecast
ax2.plot(
    range(
        len(sales_data),
        len(sales_data) + 30
    ),
    future_predictions,
    label='Future Forecast'
)

ax2.set_xlabel("Days")
ax2.set_ylabel("Sales")

ax2.legend()

st.pyplot(fig2)
# FORECAST SUMMARY
st.subheader("📌 Forecast Summary")

future_total = future_predictions.sum()

st.write(
    f"### Predicted Total Sales for Next 30 Days: "
    f"${future_total:,.0f}"
)
# SHOW FILTERED DATA
st.subheader("📂 Filtered Dataset")

st.dataframe(filtered_df.head(20))
