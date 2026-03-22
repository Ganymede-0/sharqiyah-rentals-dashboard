import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Eastern Province Rental Market",
    page_icon="🏠",
    layout="wide",
)

# # ── Load data ─────────────────────────────────────────────────────────────────
# @st.cache_data
# def load_data():
#     df = pd.read_csv("_Rental indicators for cities in Eastern Province_.csv")
#     df["Date"]     = pd.to_datetime(df["Date"])
#     df["year"]     = df["Date"].dt.year
#     df["quarter"]  = df["Date"].dt.quarter
#     df["period"]   = df["year"].astype(str) + "-Q" + df["quarter"].astype(str)
#     df["category"] = df["category"].str.replace(" - Residential", "", regex=False)
#     return df

# df = load_data()


# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    # 1. Find the exact folder where this dashboard.py file is living
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Glue that folder path to your exact dataset name
    file_path = os.path.join(current_dir, "cleaned_sharqiyah_rentals.csv")
    
    # 3. Force Pandas to open that exact, absolute path
    df = pd.read_csv(file_path)
    
    # 4. Your normal cleaning steps
    df["Date"]     = pd.to_datetime(df["Date"])
    df["year"]     = df["Date"].dt.year
    df["quarter"]  = df["Date"].dt.quarter
    df["period"]   = df["year"].astype(str) + "-Q" + df["quarter"].astype(str)
    df["category"] = df["category"].str.replace(" - Residential", "", regex=False)
    
    return df

df = load_data()

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")

    all_cities = sorted(df["city"].unique())
    sel_cities = st.multiselect("City", all_cities, default=all_cities)

    all_cats = sorted(df["category"].unique())
    sel_cats = st.multiselect("Property type", all_cats, default=all_cats)

    y_min, y_max = int(df["year"].min()), int(df["year"].max())
    yr = st.slider("Year range", y_min, y_max, (y_min, y_max))

    st.divider()
    st.caption("Eastern Province · Saudi Open Data Portal · 2019–2024")

# ── Filter ────────────────────────────────────────────────────────────────────
dff = df[
    df["city"].isin(sel_cities) &
    df["category"].isin(sel_cats) &
    df["year"].between(yr[0], yr[1])
]

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏠 Eastern Province Rental Market")
st.caption(f"Showing **{len(dff):,}** records · {yr[0]}–{yr[1]}")
st.divider()

# ── KPIs ──────────────────────────────────────────────────────────────────────
total_deals = int(dff["total_deals"].sum())
avg_price   = dff["average"].mean() if not dff.empty else 0
top_city    = dff.groupby("city")["total_deals"].sum().idxmax() if not dff.empty else "—"
top_cat     = dff.groupby("category")["total_deals"].sum().idxmax() if not dff.empty else "—"
min_price_city = dff.groupby("city")["average"].mean().idxmin() if not dff.empty else "—"

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total transactions", f"{total_deals:,}")
k2.metric("Avg rental price",   f"{avg_price:,.0f} SAR")
k3.metric("Top city by volume", top_city)
k4.metric("Top property type",  top_cat)
k5.metric("Most affordable city", min_price_city)

st.divider()


# ── Row 1: Trend + Donut ──────────────────────────────────────────────────────
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
    st.caption("Apartments account for 87.8% of all deals, the market is overwhelmingly budget residential. Villas and duplexes are rare but command premium prices.")
    cat_df = dff.groupby("category")["total_deals"].sum().reset_index()
    fig = px.pie(cat_df, names="category", values="total_deals",
                 hole=0.55, color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_traces(textposition="outside", textinfo="percent+label",
                      hovertemplate="%{label}: %{value:,}<extra></extra>")
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Row 2: City bars ──────────────────────────────────────────────────────────
c3, c4 = st.columns(2)

city_sum = (
    dff.groupby("city")
    .agg(total_deals=("total_deals", "sum"), avg_price=("average", "mean"))
    .reset_index()
)
bar_h = max(300, len(city_sum) * 26 + 60)

with c3:
    st.subheader("Transactions by city")
    st.caption("Dammam dominates with 581K deals. Activity drops sharply after the top 5 cities.")

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
    st.caption("Al Khobar is the most expensive city at 38,673 SAR driven by premium and expat housing. Manakh is the most affordable at 6,700 SAR.")
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

# ── Row 3: Category price bar + Scatter ───────────────────────────────────────
c5, c6 = st.columns(2)

with c5:
    st.subheader("Average price by property type (SAR)")
    st.caption("Villas average 37,382 SAR nearly 3× the apartment average. Studios barely undercut apartments with only a 1,228 SAR gap.")

    d = (dff.groupby("category")["average"].mean().reset_index()
           .rename(columns={"average":"avg_price"})
           .sort_values("avg_price", ascending=True))
    fig = px.bar(d, x="avg_price", y="category", orientation="h",
                 color="avg_price",
                 color_continuous_scale=[[0,"#9FE1CB"],[1,"#085041"]],
                 text=d["avg_price"].apply(lambda v: f"{v:,.0f}"),
                 labels={"avg_price":"Avg price (SAR)","category":""})
    fig.update_traces(textposition="outside")
    fig.update_layout(height=260, margin=dict(l=0,r=60,t=10,b=0),
                      coloraxis_showscale=False, xaxis=dict(tickformat=","))
    st.plotly_chart(fig, use_container_width=True)

with c6:

    
    st.subheader("City: deals vs. price")
    st.caption("high-volume cities like Dammam are budget-driven, while premium cities like Al Khobar have fewer but far more expensive deals.")
    
    fig = px.scatter(city_sum, x="total_deals", y="avg_price",
                     text="city", size="total_deals", size_max=45,
                     color="avg_price",
                     color_continuous_scale=[[0,"#B5D4F4"],[1,"#0C447C"]],
                     labels={"total_deals":"Total deals","avg_price":"Avg price (SAR)"})
    fig.update_traces(textposition="top center",
                      hovertemplate="<b>%{text}</b><br>Deals: %{x:,}<br>Price: %{y:,.0f} SAR<extra></extra>")
    fig.update_layout(height=260, margin=dict(l=0,r=0,t=10,b=0),
                      coloraxis_showscale=False,
                      xaxis=dict(tickformat=","), yaxis=dict(tickformat=","))
    st.plotly_chart(fig, use_container_width=True)
    

st.divider()


# ── Row 0b: Outlier Detection (notebook §4.4) ─────────────────────────────────
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

st.caption("Dots beyond the whiskers are outliers. Luxury villas push rental prices up to 180,000 SAR. Major cities (Dammam) drive transaction volume spikes up to 39,000 deals per record.")

st.divider()

# ── Row 0c: Correlation Heatmap (notebook §4.8) ───────────────────────────────
st.subheader("Correlation Between Rental Deals and Average Price")
st.caption("Weak negative correlation (r = −0.075) cities with the most transactions tend to have lower average prices, confirming the market is volume-driven by budget apartments, not premium units.")

cc1, cc2 = st.columns([1, 2])

with cc1:
    corr = dff[["total_deals", "average"]].corr().round(3)
    fig_corr = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale=[[0, "#042C53"], [0.5, "#EBF5FB"], [1, "#A32D2D"]],
        zmin=-1, zmax=1,
        labels=dict(color="Correlation"),
        aspect="auto",
    )
    fig_corr.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0))
    fig_corr.update_traces(textfont=dict(size=14))
    st.plotly_chart(fig_corr, use_container_width=True)

with cc2:
    sample = dff.sample(min(500, len(dff)), random_state=42) if len(dff) > 500 else dff
    fig_sc = px.scatter(
        sample, x="total_deals", y="average",
        color="category",
        opacity=0.55,
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={"total_deals": "Total Deals", "average": "Avg Price (SAR)", "category": "Type"},
    )
    fig_sc.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0),
                         xaxis=dict(tickformat=","), yaxis=dict(tickformat=","),
                         legend=dict(orientation="h", y=1.1, x=0, font=dict(size=10)))
    st.plotly_chart(fig_sc, use_container_width=True)

st.divider()

# ── Row 4: Heatmap — avg price by city × year ─────────────────────────────────
st.subheader("Average price heatmap city × year (SAR)")
st.caption("Al Khobar maintains consistently high prices (37–39K SAR) across all years. Buqayq and Ras Tanura dropped sharply after 2020. Darker blue = higher average price.")

heat = (
    dff.groupby(["city", "year"])["average"].mean().reset_index()
)
# Keep only cities with enough data across years
city_counts = heat.groupby("city")["year"].count()
top_cities  = city_counts[city_counts >= 3].index.tolist()
heat = heat[heat["city"].isin(top_cities)]
pivot = heat.pivot(index="city", columns="year", values="average").round(0)

fig = px.imshow(
    pivot,
    color_continuous_scale=[[0,"#EBF5FB"],[0.5,"#185FA5"],[1,"#042C53"]],
    labels=dict(x="Year", y="City", color="Avg price (SAR)"),
    text_auto=True,
    aspect="auto",
)
fig.update_layout(height=max(300, len(top_cities)*28+80), margin=dict(l=0,r=0,t=10,b=0))
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Row 5: Summary table ──────────────────────────────────────────────────────
st.subheader("Yearly summary")
st.caption("Transaction volume grew 6× from 2019 to 2024. Prices peaked in 2020 (+3.6% YoY) then fell through 2023 before a modest 2024 recovery (+5.6%).")

yearly = (
    dff.groupby("year")
    .agg(total_deals=("total_deals","sum"), avg_price=("average","mean"))
    .reset_index()
    .rename(columns={"year":"Year","total_deals":"Total deals","avg_price":"Avg price (SAR)"})
)
yearly["Year"]            = yearly["Year"].astype(str)
yearly["Total deals"]     = yearly["Total deals"].astype(int)
yearly["Avg price (SAR)"] = yearly["Avg price (SAR)"].round(0).astype(int)
yearly["YoY deals"]       = yearly["Total deals"].pct_change().mul(100).round(1)
yearly["YoY price"]       = yearly["Avg price (SAR)"].pct_change().mul(100).round(1)

st.dataframe(
    yearly.set_index("Year"),
    use_container_width=True,
    column_config={
        "Total deals":     st.column_config.NumberColumn(format="%d"),
        "Avg price (SAR)": st.column_config.NumberColumn(format="%d SAR"),
        "YoY deals":       st.column_config.NumberColumn("YoY deals %",  format="%.1f%%"),
        "YoY price":       st.column_config.NumberColumn("YoY price %",  format="%.1f%%"),
    },
)
