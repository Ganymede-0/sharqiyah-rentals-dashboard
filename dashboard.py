import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import joblib

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Eastern Province Rental Market",
    page_icon="",
    layout="wide",
)


# ── Asset Loading (Cached for Speed) ──────────────────────────────────────────
@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "cleaned_sharqiyah_rentals.csv")
    
    df = pd.read_csv(file_path)
    
    df["Date"]     = pd.to_datetime(df["Date"])
    df["year"]     = df["Date"].dt.year
    df["quarter"]  = df["Date"].dt.quarter
    df["period"]   = df["year"].astype(str) + "-Q" + df["quarter"].astype(str)
    # We keep the original category in a hidden column to feed the AI later if needed
    df["category_clean"] = df["category"].str.replace(" - Residential", "", regex=False)
    
    return df

@st.cache_resource
def load_ai_models():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "final_model.pkl")
    encoder_path = os.path.join(current_dir, "encoder.pkl")
    
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    
    return model, encoder

# Load everything into memory
df = load_data()
catboost_model, encoder = load_ai_models()

# ── Main UI Header ────────────────────────────────────────────────────────────

st.markdown(
    "<h1><span style='color: #ff4b4b;'>Sharqiyah Rental Monitor</span>: Regional Analysis & Long-Term Forecasting of Eastern Province Housing Trends</h1>", 
    unsafe_allow_html=True
)
st.markdown(
    "<p style='font-size: 20px; '>A comprehensive market analysis and predictive pricing tool powered by Machine Learning.</p>", 
    unsafe_allow_html=True
)

st.divider()

import streamlit as st

# 1. Inject custom CSS to target the tab text
st.markdown(
    """
    <style>
    /* Target the text inside the Streamlit tabs */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 22px; /* Change this value to make it bigger or smaller */
        font-weight: bold; /* Optional: makes the text bold */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ── Tabs Setup ────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Historical Market Analysis", "AI Price Estimator"])


# ==============================================================================
# TAB 1: MARKET ANALYSIS (EDA)
# ==============================================================================
with tab1:
    st.markdown("### Market Filters")
    
    # Put filters in columns for a cleaner layout
    colA, colB, colC = st.columns(3)
    all_cities = sorted(df["city"].unique())
    sel_cities = colA.multiselect("City", all_cities, default=all_cities)

    all_cats = sorted(df["category_clean"].unique())
    sel_cats = colB.multiselect("Property type", all_cats, default=all_cats)

    y_min, y_max = int(df["year"].min()), int(df["year"].max())
    yr = colC.slider("Year range", y_min, y_max, (y_min, y_max))

    # Apply Filters
    dff = df[
        df["city"].isin(sel_cities) &
        df["category_clean"].isin(sel_cats) &
        df["year"].between(yr[0], yr[1])
    ]

    st.caption(f"Showing **{len(dff):,}** filtered records.")
    
    # ── KPIs ──────────────────────────────────────────────────────────────────
    total_deals = int(dff["total_deals"].sum())
    avg_price   = dff["average"].mean() if not dff.empty else 0
    top_city    = dff.groupby("city")["total_deals"].sum().idxmax() if not dff.empty else "—"
    top_cat     = dff.groupby("category_clean")["total_deals"].sum().idxmax() if not dff.empty else "—"
    min_price_city = dff.groupby("city")["average"].mean().idxmin() if not dff.empty else "—"

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total transactions", f"{total_deals:,}")
    k2.metric("Avg rental price",   f"{avg_price:,.0f} SAR")
    k3.metric("Top city by volume", top_city)
    k4.metric("Top property type",  top_cat)
    k5.metric("Most affordable city", min_price_city)

    st.divider()

    # ── Row 1: Trend + Donut ──────────────────────────────────────────────────
    c1, c2 = st.columns([2, 1])

    with c1:
        st.subheader("Price & volume over time")
        st.caption("Prices peaked at ~26,000 SAR in 2020-Q3 likely a COVID-era demand shock then stabilised at 19,000–21,000 SAR.")

        ts = (
            dff.groupby("Date")
            .agg(avg_price=("average", "mean"), total_deals=("total_deals", "sum"))
            .reset_index().sort_values("Date")
        )
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
            x=ts["Date"], y=ts["avg_price"].round(0),
            name="Avg price (SAR)", mode="lines+markers",
            line=dict(color="#185FA5", width=2.5), marker=dict(size=4),
            fill="tozeroy", fillcolor="rgba(24,95,165,0.07)",
        ), secondary_y=False)
        fig.add_trace(go.Bar(
            x=ts["Date"], y=ts["total_deals"],
            name="Transactions", marker_color="rgba(151,196,89,0.65)",
        ), secondary_y=True)
        fig.update_layout(
            height=320, margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", y=1.08, x=0),
            hovermode="x unified",
        )
        fig.update_yaxes(title_text="Avg price (SAR)", secondary_y=False, tickformat=",")
        fig.update_yaxes(title_text="Transactions",    secondary_y=True,  tickformat=",")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Share by property type")
        st.caption("Apartments account for 87.8% of all deals. The market is overwhelmingly budget residential.")
        cat_df = dff.groupby("category_clean")["total_deals"].sum().reset_index()
        fig = px.pie(cat_df, names="category_clean", values="total_deals",
                     hole=0.55, color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_traces(textposition="outside", textinfo="percent+label",
                          hovertemplate="%{label}: %{value:,}<extra></extra>")
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Row 2: City bars ──────────────────────────────────────────────────────
    c3, c4 = st.columns(2)

    city_sum = (
        dff.groupby("city")
        .agg(total_deals=("total_deals", "sum"), avg_price=("average", "mean"))
        .reset_index()
    )
    bar_h = max(300, len(city_sum) * 26 + 60)

    with c3:
        st.subheader("Transactions by city")
        st.caption("Dammam dominates with massive transaction volume.")
        d = city_sum.sort_values("total_deals", ascending=True)
        fig = px.bar(d, x="total_deals", y="city", orientation="h",
                     color="total_deals",
                     color_continuous_scale=[[0,"#B5D4F4"],[1,"#0C447C"]],
                     text=d["total_deals"].apply(lambda v: f"{v/1e3:.0f}K" if v>=1000 else str(v)),
                     labels={"total_deals":"Total deals","city":""})
        fig.update_traces(textposition="outside")
        fig.update_layout(height=bar_h, margin=dict(l=0,r=40,t=10,b=0),
                          coloraxis_showscale=False, xaxis=dict(tickformat=","))
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.subheader("Average Rental price by city (SAR)")
        st.caption("Al Khobar is the most expensive city at 38,673 SAR.")
        d = city_sum.sort_values("avg_price", ascending=True)
        fig = px.bar(d, x="avg_price", y="city", orientation="h",
                     color="avg_price",
                     color_continuous_scale=[[0,"#FAC775"],[1,"#633806"]],
                     text=d["avg_price"].apply(lambda v: f"{v:,.0f}"),
                     labels={"avg_price":"Avg price (SAR)","city":""})
        fig.update_traces(textposition="outside")
        fig.update_layout(height=bar_h, margin=dict(l=0,r=60,t=10,b=0),
                          coloraxis_showscale=False, xaxis=dict(tickformat=","))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Row 3: Outliers & Heatmap ─────────────────────────────────────────────
    st.subheader("Outlier Detection")
    ob1, ob2 = st.columns(2)

    with ob1:
        fig_out1 = px.box(dff, y="average",
                          labels={"average": "Average Rental Price (SAR)"},
                          title="Rental Price Distribution")
        fig_out1.update_traces(marker_color="#185FA5",
                               hovertemplate="Price: %{y:,.0f} SAR<extra></extra>")
        fig_out1.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0),
                               yaxis=dict(tickformat=","))
        st.plotly_chart(fig_out1, use_container_width=True)

    with ob2:
        fig_out2 = px.box(dff, y="total_deals",
                          labels={"total_deals": "Total Deals"},
                          title="Transaction Volume Distribution")
        fig_out2.update_traces(marker_color="#D85A30",
                               hovertemplate="Deals: %{y:,}<extra></extra>")
        fig_out2.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0),
                               yaxis=dict(tickformat=","))
        st.plotly_chart(fig_out2, use_container_width=True)

# ==============================================================================
# TAB 2: AI PRICE ESTIMATOR (CATBOOST)
# ==============================================================================
with tab2:
    st.markdown("### Predictive Pricing Engine")
    st.markdown("Utilize our highly-tuned CatBoost Regressor to estimate fair-market rental prices based on real estate parameters.")
    
    with st.form("ai_prediction_form"):
        col1, col2 = st.columns(2)
        
        # We use the raw category clean names for the UI, but will append "- Residential" for the AI
        user_city = col1.selectbox("City Location", sorted(df['city'].unique()))
        user_category = col2.selectbox("Property Type", sorted(df['category_clean'].unique()))
        
        # col3, col4, col5 = st.columns(3)
        # user_year = col3.selectbox("Forecasting Year", [2024, 2025, 2026, 2027])
        # user_quarter = col4.selectbox("Quarter", [1, 2, 3, 4])
        # user_deals = col5.number_input("Est. Transaction Volume (Demand)", min_value=1, max_value=50000, value=50)
        col3, col4, col5, col6 = st.columns(4)
        user_year = col3.selectbox("Forecasting Year", [2024, 2025, 2026, 2027])
        user_quarter = col4.selectbox("Quarter", [1, 2, 3, 4])
        user_deals = col5.number_input("Est. Demand (Deals)", min_value=1, max_value=50000, value=50)
        user_growth = col6.slider("Expected Annual Growth (%)", min_value=-5.0, max_value=15.0, value=3.5, step=0.5)

        submit_prediction = st.form_submit_button("Calculate Market Price", type="primary", use_container_width=True)
        
if submit_prediction:
        # 1. Cap the AI model year at 2024 (its maximum historical knowledge)
        # This gets the most accurate "Base Value" before forecasting
        model_year = min(user_year, 2024)
        
        input_data = pd.DataFrame({
            'region': ['Eastern Province'],
            'city': [user_city],
            'category': [f"{user_category} - Residential"],
            'total_deals': [user_deals],
            'year': [model_year],       
            'quarter': [user_quarter]  
        })
        
        try:
            # 2. Process through the saved ColumnTransformer
            encoded_input = encoder.transform(input_data)
            
            # 3. Predict the Base Value using CatBoost
            log_prediction = catboost_model.predict(encoded_input)[0]
            base_price = np.expm1(log_prediction)
            

            # 4. THE FORECASTING ENGINE (Dynamic Scenario Analysis)
            if user_year > 2024:
                years_ahead = user_year - 2024
                # Convert the slider percentage (e.g., 3.5) into a decimal (0.035)
                growth_decimal = user_growth / 100.0 
                final_price = base_price * ((1 + growth_decimal) ** years_ahead)
                trend_msg = f"Includes a user-defined {user_growth}% annualized market forecast compounded over {years_ahead} year(s)."
            else:
                final_price = base_price
                trend_msg = "Based purely on historical market data (No future growth applied)."

            st.success("AI Forecasting Complete")
            
            # Output Display
            res_col1, res_col2 = st.columns([1, 2])
            with res_col1:
                st.metric("Estimated Fair Market Value", f"{final_price:,.0f} SAR")
            with res_col2:
                st.info(f"**Insight:** A {user_category} in {user_city} with demand of {user_deals} deals in Q{user_quarter} {user_year}. \n\n**Forecasting Note:** {trend_msg}")
                
        except Exception as e:
            st.error(f"Error during prediction pipeline: {e}")