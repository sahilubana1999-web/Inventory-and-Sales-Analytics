import streamlit as st
import os
import io
import math
import warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet

# --- 0. Configuration and Environment Fixes ---

# Configure Streamlit page
st.set_page_config(layout="wide", page_title="Inventory & Sales Analytics")

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 50)

# Fix for NumPy 1.24+ compatibility (resolves 'bool8' error)
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

# --- Define MANDATORY columns for data validation ---
MANDATORY_COLUMNS = [
    "Date",
    "Item Sold",
    "Quantity Sold",
]
# ----------------------------------------------------


# --- 1. Core Functions (Updated with Validation and Date Fix) ---

@st.cache_data
def preprocess_data(df):
    """Performs the cleaning and preprocessing steps from Section 3, including validation."""

    # 1. Normalize column names (strip)
    df.columns = [str(c).strip() for c in df.columns]
    normalized_cols = [c.strip().lower() for c in df.columns]

    # --- 1. VALIDATION CHECK ---
    
    # Check for basic mandatory columns
    for col_name in MANDATORY_COLUMNS:
        if col_name.strip().lower() not in normalized_cols:
            st.error(
                f"**Validation Failed:** Your data is missing the required column: **'{col_name}'**."
                " Please ensure the column name is present."
            )
            return None

    # Check for revenue column necessity
    has_total_value = 'Total Sale Value (₹)'.strip().lower() in normalized_cols
    has_price = 'Price per Item (₹)'.strip().lower() in normalized_cols
    
    if not has_total_value and not has_price:
        st.error(
            "**Validation Failed:** To calculate revenue, your data must contain either **'Total Sale Value (₹)'** "
            "or **'Price per Item (₹)'** (since 'Quantity Sold' is present)."
        )
        return None
        
    # --- 2. DATA CLEANING AND PROCESSING ---

    # --- DATE PARSING FIX ---
    date_col = next((col for col in df.columns if col.strip().lower() == 'date'), None)
    
    # List of common formats to try explicitly (most common first)
    date_formats = ['%d-%m-%Y', '%d/%m/%Y', '%m-%d-%Y', '%m/%d/%Y']
    
    for fmt in date_formats:
        try:
            # Attempt parsing with explicit format
            df['Date'] = pd.to_datetime(df[date_col], format=fmt, errors='coerce')
            # Check if parsing was successful for the majority of rows
            if df['Date'].notna().sum() > (len(df) * 0.9):
                break # Success, break the loop
        except ValueError:
            continue
    else:
        # Final attempt with flexible parser (original logic)
        df['Date'] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')


    # --- FINAL CHECK FOR NA DATES ---
    if df['Date'].isna().any():
        st.error(
            "Date parsing failed for multiple rows. Please check the 'Date' column format. "
            "It looks like your data has dates in highly inconsistent formats. The process cannot continue."
        )
        return None

    # Fill missing Price per Item
    if 'Price per Item (₹)' not in df.columns or df['Price per Item (₹)'].isnull().all():
        if 'Total Sale Value (₹)' in df.columns and 'Quantity Sold' in df.columns:
            df['Price per Item (₹)'] = df['Total Sale Value (₹)'] / df['Quantity Sold']
            
    # Ensure numeric columns are numeric
    for col in ['Quantity Sold', 'Price per Item (₹)', 'Total Sale Value (₹)']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Compute Total Sale Value if missing
    if 'Total Sale Value (₹)' not in df.columns or df['Total Sale Value (₹)'].isnull().all():
        df['Total Sale Value (₹)'] = df['Quantity Sold'] * df['Price per Item (₹)']

    # Create Weekday if missing
    if 'Weekday' not in df.columns or df['Weekday'].isnull().all():
        df['Weekday'] = df['Date'].dt.day_name()

    # Trim whitespace in Item names
    df['Item Sold'] = df['Item Sold'].astype(str).str.strip().str.lower()
    
    return df

@st.cache_data
def run_analysis(df):
    """Runs all aggregation and analysis steps (Section 4 & 5)."""
    
    # Daily total revenue and quantity
    daily = df.groupby('Date').agg(
        DailyRevenue=('Total Sale Value (₹)', 'sum'),
        DailyQty=('Quantity Sold', 'sum')
    ).reset_index()
    daily['Weekday'] = daily['Date'].dt.day_name()

    # Compute avg daily qty and sd using pivot for clarity
    pivot_qty = df.pivot_table(index='Date', columns='Item Sold', values='Quantity Sold', aggfunc='sum').fillna(0)
    item_stats = []
    for item in pivot_qty.columns:
        series = pivot_qty[item]
        avg = series.mean()
        sd = series.std(ddof=0)  # population sd
        item_stats.append((item, avg, sd, series.sum()))
    item_stats_df = pd.DataFrame(item_stats, columns=['Item Sold', 'AvgDailyQty', 'SDDailyQty', 'TotalQty'])
    
    # Merge with revenue
    rev = df.groupby('Item Sold')['Total Sale Value (₹)'].sum().reset_index().rename(columns={'Total Sale Value (₹)':'TotalRevenue'})
    item_summary = item_stats_df.merge(rev, on='Item Sold', how='left')
    item_summary = item_summary.sort_values('TotalRevenue', ascending=False).reset_index(drop=True)
    
    # Pareto cumulative percent
    total_rev = item_summary['TotalRevenue'].sum()
    item_summary['CumRevenue'] = item_summary['TotalRevenue'].cumsum()
    item_summary['CumRevenuePct'] = (item_summary['CumRevenue'] / total_rev * 100).round(2)
    
    # Weekday analysis
    sales_by_weekday = df.groupby('Weekday').agg(Revenue=('Total Sale Value (₹)', 'sum'),
                                                 Quantity=('Quantity Sold', 'sum')).reset_index()
    week_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    sales_by_weekday['Weekday'] = pd.Categorical(sales_by_weekday['Weekday'], categories=week_order, ordered=True)
    sales_by_weekday = sales_by_weekday.sort_values('Weekday')
    
    return daily, item_summary, sales_by_weekday, pivot_qty


def compute_reorder_table(item_summary_df, lead_time_days, z_value):
    """Reorder and Safety Stock calculations (Section 6)."""
    rows = []
    for _, r in item_summary_df.iterrows():
        item = r['Item Sold']
        avg = float(r['AvgDailyQty'])
        sd = float(r['SDDailyQty'])
        total_qty = int(r['TotalQty'])
        
        # Statistical method: z * SD * sqrt(LT)
        safety = math.ceil(z_value * sd * math.sqrt(max(1, lead_time_days)))
        reorder_qty = math.ceil(avg * max(1, lead_time_days) + safety)
        
        rows.append({
            'Item Sold': item,
            'AvgDailyQty': round(avg,2),
            'SDDailyQty': round(sd,2),
            'SafetyStock (Units)': safety,
            'ReorderQty (ROP)': reorder_qty,
            'TotalQty (History)': total_qty
        })
    return pd.DataFrame(rows)

@st.cache_data
def forecast_moving_average(daily_series, window=7, periods=14):
    s = daily_series.set_index('Date')['DailyRevenue'].asfreq('D').fillna(0)
    ma = s.rolling(window=window, min_periods=1).mean()
    last_ma = ma.iloc[-1]
    future_dates = pd.date_range(start=s.index[-1] + pd.Timedelta(days=1), periods=periods)
    forecast = pd.Series([last_ma]*periods, index=future_dates)
    fc_df = forecast.reset_index().rename(columns={'index':'Date', 0:'Forecast'})
    return fc_df

@st.cache_data
def forecast_exponential_smoothing(daily_series, periods=14):
    s = daily_series.set_index('Date')['DailyRevenue'].asfreq('D').fillna(0)
    # simple ETS with additive trend
    model = ExponentialSmoothing(s, trend='add', seasonal=None, initialization_method='estimated')
    fit = model.fit(optimized=True)
    pred = fit.forecast(periods)
    fc_df = pred.reset_index().rename(columns={'index':'Date', 0:'Forecast'})
    return fc_df

@st.cache_data
def forecast_prophet(daily_series, periods=14):
    prophet_df = daily_series[['Date','DailyRevenue']].rename(columns={'Date':'ds','DailyRevenue':'y'})
    prophet_df = prophet_df.set_index('ds').asfreq('D').fillna(0).reset_index()
    
    m = Prophet(daily_seasonality=True, yearly_seasonality=False) # Simplified settings
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    fc = forecast[['ds','yhat']].tail(periods).rename(columns={'ds':'Date','yhat':'Forecast'})
    return fc[['Date', 'Forecast']]


# --- 2. Streamlit App Layout (Updated with Guide) ---

st.title("🛒 Sales & Inventory Analytics App")
st.markdown("Upload your transactional Excel file to generate a full analysis, forecasts, and reorder suggestions.")

# --- USER GUIDE & VALIDATION INSTRUCTIONS ---
st.subheader("📝 Data Requirements")
st.info(
    "Your uploaded Excel file **must** contain the following columns (case and spelling are lenient, but structure matters): "
    f"**{', '.join(MANDATORY_COLUMNS)}**"
    "\n\nAdditionally, you must provide either **'Total Sale Value (₹)'** OR **'Price per Item (₹)'**."
)
# --- END USER GUIDE ---

# File Uploader (Section 2)
uploaded_file = st.sidebar.file_uploader("Upload your Excel Sales Data", type=['xlsx', 'csv']) # Added CSV support too

if uploaded_file is not None:
    # Read the file
    try:
        # Use filename to infer if it's CSV or Excel
        if uploaded_file.name.endswith('.csv'):
             df_raw = pd.read_csv(uploaded_file)
        else:
             df_raw = pd.read_excel(uploaded_file)
             
        # Pass the raw DataFrame to the validation/preprocessing function
        df = preprocess_data(df_raw.copy())
    except Exception as e:
        st.error(f"Error reading or processing file: {e}")
        st.stop()

    if df is not None and not df.empty:
        # Run Analysis
        daily, item_summary, sales_by_weekday, pivot_qty = run_analysis(df)
        total_revenue = item_summary['TotalRevenue'].sum()

        # --- 2.1. SIDEBAR INPUTS ---
        with st.sidebar:
            st.header("Inventory Settings")
            lead_time = st.slider("Lead Time (Days)", 1, 10, 3)
            service_level_pct = st.slider("Service Level (%)", 70, 99, 90)
            
            # Convert Service Level to Z-score
            z_score_map = {70: 0.52, 80: 0.84, 90: 1.28, 95: 1.64, 99: 2.33}
            z_value = z_score_map.get(service_level_pct, 1.28)
            st.caption(f"Z-Value used for safety stock: **{z_value}**")

            st.header("Forecasting Settings")
            forecast_horizon = st.slider("Forecast Horizon (Days)", 7, 30, 14)


        # --- 3. Key Metrics (Section 10) ---
        st.header("📊 Key Performance Indicators (KPIs)")
        
        # Calculate ROI/Savings
        monthly_estimated = total_revenue
        potential_savings_10pct = round(monthly_estimated * 0.10, 2)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Revenue (History)", f"₹ {total_revenue:,.2f}")
        col2.metric("Top Selling Item", item_summary['Item Sold'].iloc[0].title())
        col3.metric("Potential 10% Waste Saving", f"₹ {potential_savings_10pct:,.2f}")
        
        st.divider()

        # --- 4. Sales Trend & Peaks (Section 8.1, 8.3) ---
        st.header("📈 Sales Trend Analysis")

        col_trend, col_peak = st.columns([3, 2])
        
        with col_trend:
            st.subheader("Daily Revenue Trend")
            fig_daily = px.line(daily, x='Date', y='DailyRevenue', 
                                title='Daily Revenue Trend', markers=True)
            fig_daily.update_layout(template='plotly_white')
            st.plotly_chart(fig_daily, use_container_width=True)

        with col_peak:
            st.subheader("Revenue by Weekday")
            fig_weekday = px.bar(sales_by_weekday, x='Weekday', y='Revenue', 
                                title='Revenue by Weekday', labels={'Revenue':'Revenue (₹)'})
            fig_weekday.update_layout(template='plotly_white')
            st.plotly_chart(fig_weekday, use_container_width=True)

        # --- 5. Inventory & Reorder Policy (Section 6, 8.5) ---
        st.header("📦 Inventory Optimization")

        # Run Reorder Calculation based on user input
        reorder_table = compute_reorder_table(item_summary, lead_time, z_value)
        
        st.subheader(f"Safety Stock & Reorder Point ({service_level_pct}% Service Level)")
        
        # Display the Reorder Table
        st.dataframe(reorder_table, use_container_width=True, height=300)
        st.caption("Reorder Quantity (ROP) = Avg Daily Demand * Lead Time + Safety Stock")

        st.subheader("Item Demand Variability (Top 6)")
        top_n = 6
        top_items_list = item_summary['Item Sold'].head(top_n).tolist()
        
        fig_items = go.Figure()
        for item in top_items_list:
            fig_items.add_trace(go.Scatter(x=pivot_qty.index, y=pivot_qty[item], 
                                          mode='lines+markers', name=item))
        fig_items.update_layout(title='Daily Quantity Series - Visualize Variability', 
                                template='plotly_white')
        st.plotly_chart(fig_items, use_container_width=True)


        # --- 6. Pareto Analysis (Section 4, 8.2) ---
        st.header("💎 Top Items and Pareto (ABC) Analysis")
        
        # Display Top Items Bar Chart
        fig_top_items = px.bar(item_summary.head(15), x='Item Sold', y='TotalRevenue', 
                             title='Top 15 Items by Revenue', labels={'TotalRevenue':'Revenue (₹)'})
        fig_top_items.update_layout(xaxis_tickangle=-45, template='plotly_white')
        st.plotly_chart(fig_top_items, use_container_width=True)
        
        # Display Pareto Table
        st.subheader("Revenue Contribution")
        pareto_df = item_summary[['Item Sold', 'TotalRevenue', 'CumRevenuePct']].head(30)
        st.dataframe(pareto_df, use_container_width=True, hide_index=True)
        st.caption("Items above 80% Cumulative Revenue are typically considered 'A' items.")

        # --- 7. Forecasting (Section 7, 9) ---
        st.header(f"🔮 Revenue Forecasting ({forecast_horizon} Days)")
        
        # Run all three forecasts
        ma_fc = forecast_moving_average(daily, periods=forecast_horizon)
        ets_fc = forecast_exponential_smoothing(daily, periods=forecast_horizon)
        prophet_fc = forecast_prophet(daily, periods=forecast_horizon)

        # Combine forecasts
        fc_combined = ma_fc.rename(columns={'Forecast':'MA_Forecast'}).merge(
            ets_fc.rename(columns={'Forecast':'ETS_Forecast'}), on='Date').merge(
            prophet_fc.rename(columns={'Forecast':'Prophet_Forecast'}), on='Date')
        
        st.subheader("Forecast Comparison Table (Revenue in ₹)")
        st.dataframe(fc_combined.set_index('Date').round(2), use_container_width=True)
        
        # Create a visual plot for the combined forecast
        forecast_plot_data = fc_combined.melt('Date', var_name='Model', value_name='Forecast')
        
        fig_forecast = px.line(forecast_plot_data, x='Date', y='Forecast', color='Model', 
                               title='Revenue Forecast Comparison')
        fig_forecast.update_layout(template='plotly_white')
        st.plotly_chart(fig_forecast, use_container_width=True)


        # --- 8. Export Option (Replaces Section 11) ---
        st.divider()
        st.header("📥 Download Full Report")
        
        # Create a single Excel file in memory
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Write all tables to separate sheets
            df.to_excel(writer, sheet_name='Raw_Data', index=False)
            daily.to_excel(writer, sheet_name='Daily_Summary', index=False)
            item_summary.to_excel(writer, sheet_name='Item_Summary', index=False)
            reorder_table.to_excel(writer, sheet_name='Reorder_Table', index=False)
            fc_combined.to_excel(writer, sheet_name='Forecasts', index=False)
            sales_by_weekday.to_excel(writer, sheet_name='Weekday_Summary', index=False)

            # Executive Summary
            exec_summary = pd.DataFrame({
                'Metric': ['Total Revenue (History)', 'Potential saving from 10% waste reduction'],
                'Value': [round(total_revenue,2), potential_savings_10pct]
            })
            exec_summary.to_excel(writer, sheet_name='Executive_Summary', index=False)

        # Download button
        st.download_button(
            label="Download Analytics Report (Excel)",
            data=output.getvalue(),
            file_name="Sales_Inventory_Analytics_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        st.caption("Note: Charts are displayed on the app but are not embedded in this download.")


else:
    st.info("Awaiting file upload. Please use the sidebar to upload your sales data Excel file.")
