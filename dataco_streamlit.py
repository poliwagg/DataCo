import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from snowflake.snowpark.context import get_active_session

st.set_page_config(page_title="Delivery & Sales Dashboard", layout="wide")
st.title("Delivery & Sales Dashboard")

session = get_active_session()

@st.cache_data(show_spinner=False)

def load_all_orders():
    df = session.sql("""
        SELECT
            order_id,
            order_item_id,
            order_date,
            shipping_date,
            IFF(is_delivery_late, 1, 0) AS late_delivery_risk,
            GREATEST(days_shipping_actual - days_shipping_scheduled, 0) AS delivery_delay,
            market_std AS market,
            delivery_region AS order_region,
            shipping_mode,
            profit_per_order AS order_profit_per_order,
            payment_type,
            order_status_std AS order_status,
            customer_segment,
            category_name_imputed AS category_name,
            product_name
        FROM DATACO.CLEAN.ORDERS_CLEAN
    """).to_pandas()
    for c in ["ORDER_DATE", "SHIPPING_DATE"]:
        df[c] = pd.to_datetime(df[c])

    df.columns = [c.lower() for c in df.columns]

    df["late_flag_label"] = np.where(df["late_delivery_risk"] == 1, "Late", "On Time")
    return df
    
def load_complete_orders():
    df = session.sql("""
        SELECT
            order_id,
            order_item_id,
            order_date,
            shipping_date,
            IFF(is_delivery_late, 1, 0) AS late_delivery_risk,
            GREATEST(days_shipping_actual - days_shipping_scheduled, 0) AS delivery_delay,
            market_std AS market,
            delivery_region AS order_region,
            shipping_mode,
            profit_per_order AS order_profit_per_order,
            payment_type,
            order_status_std AS order_status,
            customer_segment,
            category_name_imputed AS category_name,
            product_name
        FROM DATACO.CLEAN.ORDERS_CLEAN
        WHERE order_status_std = 'COMPLETE'
    """).to_pandas()

    for c in ["ORDER_DATE", "SHIPPING_DATE"]:
        df[c] = pd.to_datetime(df[c])

    df.columns = [c.lower() for c in df.columns]

    df["late_flag_label"] = np.where(df["late_delivery_risk"] == 1, "Late", "On Time")
    return df
df = load_all_orders()
df_complete_only = load_complete_orders()

st.sidebar.header("Filter")
markets = sorted(df_complete_only["market"].dropna().unique().tolist())
selected_markets = st.sidebar.multiselect("Market", markets, default=markets)

min_d = df_complete_only["shipping_date"].min()
max_d = df_complete_only["shipping_date"].max()
start_date, end_date = st.sidebar.date_input(
    "Shipping date range",
    value=[min_d, max_d],
    min_value=min_d,
    max_value=max_d
)

mask = (
    df_complete_only["market"].isin(selected_markets)
    & (df_complete_only["shipping_date"] >= pd.to_datetime(start_date))
    & (df_complete_only["shipping_date"] <= pd.to_datetime(end_date))
)
df_f = df_complete_only.loc[mask].copy()

tab1, tab2, tab3, tab4 = st.tabs(["Delivery", "Sales", "QA", "Customer"])

with tab1:
    c1, c2 = st.columns([1,1])

    with c1:
        st.subheader("Late vs On-time")
        counts = df_f["late_flag_label"].value_counts().reindex(["Late","On Time"]).fillna(0)
        explode = [0.1, 0.0]
        fig, ax = plt.subplots(figsize=(5.5,5.5))
        wedges, texts, autotexts = ax.pie(
            counts.values,
            labels=counts.index,
            autopct="%1.1f%%",
            explode=explode,
            shadow=True,
            startangle=90
        )
        ax.set_title("Late vs On Time Deliveries")
        st.pyplot(fig)

    with c2:
        st.subheader("Days Late Distribution")
        days_late = (
            df_f.loc[df_f["delivery_delay"] > 0, "delivery_delay"]
              .astype(int)
              .value_counts()
              .sort_index()
        )
        fig2, ax2 = plt.subplots(figsize=(7,4))
        ax2.bar(days_late.index, days_late.values)
        ax2.set_title("Amount of Late Deliveries (Days)")
        ax2.set_xlabel("Days Late")
        ax2.set_ylabel("Amount of Deliveries")
        ax2.set_xticks(days_late.index)
        fig2.tight_layout()
        st.pyplot(fig2)
    st.subheader("==================================================================================================================================")
    st.subheader("Monthly Average Late Deliveries")
    if not df_f.empty:
        last_month = df_f["shipping_date"].dt.to_period("M").max()
        df_late = df_f.loc[df_f["shipping_date"].dt.to_period("M") < last_month]
        df_delays = df_late.loc[df_late["delivery_delay"] > 0]

        monthly = (
            df_delays
              .resample("MS", on="shipping_date")["delivery_delay"]
              .mean()
              .rename("avg_delay_days")
              .to_frame()
        )

        fig3, ax3 = plt.subplots(figsize=(10,4))
        ax3.plot(monthly.index, monthly["avg_delay_days"], linewidth=1.2, label="Monthly Avg")
        ax3.set_title("Average Delivery Delay Over Time")
        ax3.set_xlabel("Month")
        ax3.set_ylabel("Delay (days)")
        ax3.legend()
        ax3.grid(True, linestyle="--", alpha=0.6)
        fig3.tight_layout()
        st.pyplot(fig3)
    else:
        st.info("Tidak ada data untuk periode ini.")

    st.subheader("==================================================================================================================================")
    
    st.subheader("Shipping Mode Comparison per Region (Delay Count)")
    base = df_f.loc[df_f["delivery_delay"] > 0, ["order_region", "shipping_mode", "delivery_delay"]]
    if not base.empty:
        pivoted = (
            base.groupby(["order_region","shipping_mode"]).size().reset_index(name="cnt")
                .pivot(index="order_region", columns="shipping_mode", values="cnt")
                .fillna(0)
        )
        pivoted = pivoted.sort_values(by=list(pivoted.columns), ascending=True)
        fig4, ax4 = plt.subplots(figsize=(6,8))
        pivoted.plot(kind="barh", ax=ax4)
        ax4.set_title("Shipping Mode Comparison per Region")
        ax4.set_ylabel("Region")
        ax4.set_xlabel("Delivery Delay Count")
        ax4.grid(axis="x", linestyle="-", alpha=0.5)
        fig4.tight_layout()
        st.pyplot(fig4)
        st.dataframe(pivoted.reset_index())
    else:
        st.info("Tidak ada kiriman terlambat pada filter saat ini.")

    c3, c4 = st.columns(2)

    with c3:
        st.subheader("=================================================================")
    
        st.subheader("Top 20 Categories by Late Rate (%)")
        cat_perf = (
            df_f.groupby("category_name")["delivery_delay"]
                .apply(lambda x: (x > 0).mean() * 100)
                .reset_index(name="late_rate")
                .sort_values("late_rate", ascending=False)
                .head(20)
        )
        fig5, ax5 = plt.subplots(figsize=(8,6))
        sns.barplot(data=cat_perf, x="late_rate", y="category_name", color="#1f77b4", ax=ax5)
        ax5.set_xlabel("Late Deliveries (%)")
        ax5.set_ylabel("Category")
        ax5.set_title("Top 20 Category with Highest Late Delivery Rate")
        ax5.grid(axis="y", linestyle="--", alpha=0.6)
        fig5.tight_layout()
        st.pyplot(fig5)
        st.dataframe(cat_perf)

    with c4:
        st.subheader("==================================================================================================================================")
    
        st.subheader("Top 20 Products by Late Rate (%)")
        prod_perf = (
            df_f.groupby("product_name")["delivery_delay"]
                .apply(lambda x: (x > 0).mean() * 100)
                .reset_index(name="late_rate")
                .sort_values("late_rate", ascending=False)
                .head(20)
        )
        fig6, ax6 = plt.subplots(figsize=(8,6))
        sns.barplot(data=prod_perf, x="late_rate", y="product_name", color="#1f77b4", ax=ax6)
        ax6.set_xlabel("Late Deliveries (%)")
        ax6.set_ylabel("Product")
        ax6.set_title("Top 20 Products with Highest Late Delivery Rate")
        ax6.grid(axis="y", linestyle="--", alpha=0.6)
        fig6.tight_layout()
        st.pyplot(fig6)
        st.dataframe(prod_perf)

with tab2:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Markets with the Most Loss")
        loss_by_market = (
            df_f.loc[df_f["order_profit_per_order"] < 0, ["market", "order_profit_per_order"]]
              .assign(loss=lambda d: d["order_profit_per_order"].abs())
              .groupby("market", as_index=False)["loss"].sum()
              .sort_values("loss", ascending=True)
        )
        fig7, ax7 = plt.subplots(figsize=(8,5))
        sns.barplot(data=loss_by_market, y="market", x="loss", color="#1f77b4", ax=ax7)
        ax7.set_title("Markets with the Most Loss (negative profits only)")
        ax7.set_xlabel("Loss (USD)")
        ax7.set_ylabel("Market")
        ax7.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"${x:,.0f}"))
        fig7.tight_layout()
        st.pyplot(fig7)
        st.dataframe(loss_by_market)

    with c2:
        st.subheader("Markets with the Most Profit")
        profit_by_market = (
            df_f.loc[df_f["order_profit_per_order"] > 0, ["market", "order_profit_per_order"]]
              .groupby("market", as_index=False)["order_profit_per_order"].sum()
              .rename(columns={"order_profit_per_order": "total_profit"})
              .sort_values("total_profit", ascending=True)
        )
        fig8, ax8 = plt.subplots(figsize=(8,5))
        sns.barplot(data=profit_by_market, y="market", x="total_profit", color="#1f77b4", ax=ax8)
        ax8.set_title("Markets with the Most Profit")
        ax8.set_xlabel("Profit (USD)")
        ax8.set_ylabel("Market")
        ax8.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"${x:,.0f}"))
        fig8.tight_layout()
        st.pyplot(fig8)
        st.dataframe(profit_by_market)

    product_profit = (
        df_f.loc[:, ["product_name", "order_profit_per_order"]]
          .groupby("product_name", as_index=False)["order_profit_per_order"].sum()
          .rename(columns={"order_profit_per_order": "total_profit"})
    )
    st.subheader("==================================================================================================================================")
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Top 10 Products by Total Profit")
        top_profit = product_profit.loc[product_profit["total_profit"] > 0].nlargest(10, "total_profit")
        fig9, ax9 = plt.subplots(figsize=(8,5))
        sns.barplot(data=top_profit, y="product_name", x="total_profit", color="#1f77b4", ax=ax9)
        ax9.set_xlabel("Total Profit (USD)")
        ax9.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"${x:,.0f}"))
        fig9.tight_layout()
        st.pyplot(fig9)
        st.dataframe(top_profit)

    with c4:
        st.subheader("Top 10 Products by Total Loss")
        top_loss = (
            product_profit.loc[product_profit["total_profit"] < 0]
              .nsmallest(10, "total_profit")
              .assign(total_loss=lambda d: -d["total_profit"])
        )
        fig10, ax10 = plt.subplots(figsize=(8,5))
        sns.barplot(data=top_loss, y="product_name", x="total_loss", color="#1f77b4", ax=ax10)
        ax10.set_xlabel("Total Loss (USD)")
        ax10.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"${x:,.0f}"))
        fig10.tight_layout()
        st.pyplot(fig10)
        st.dataframe(top_loss)

with tab3:
    st.subheader("Order Status vs Payment Type — Count")
    status_order = [
        "COMPLETE","PENDING","CLOSED","PENDING_PAYMENT","CANCELED",
        "PROCESSING","SUSPECTED_FRAUD","ON_HOLD","PAYMENT_REVIEW"
    ]
    ct = pd.crosstab(
        index=pd.Categorical(df["order_status"], categories=status_order, ordered=True),
        columns=df["payment_type"]
    ).fillna(0)
    fig11, ax11 = plt.subplots(figsize=(10,5))
    sns.heatmap(ct, annot=True, fmt=".0f", linewidths=.5, cmap="Blues", ax=ax11)
    ax11.set_xlabel("Payment Type")
    ax11.set_ylabel("Order Status")
    fig11.tight_layout()
    st.pyplot(fig11)
    st.dataframe(ct.reset_index())
    
    st.subheader("==================================================================================================================================")
    st.subheader("Regions with Highest Late Delivery Risk (%)")
    fraud_by_region = (
        df_f.groupby("order_region", dropna=False)
          .agg(total_orders=("order_id", "count"),
               late_orders=("late_delivery_risk", lambda x: (x == 1).sum()))
          .reset_index()
    )
    fraud_by_region["late_rate(%)"] = (fraud_by_region["late_orders"] / fraud_by_region["total_orders"]) * 100
    fraud_by_region = fraud_by_region.sort_values("late_rate(%)", ascending=False)

    fig12, ax12 = plt.subplots(figsize=(8,4))
    sns.barplot(data=fraud_by_region, x="late_rate(%)", y="order_region", palette="Blues_r", ax=ax12)
    ax12.set_xlabel("Late Delivery Risk (%)")
    ax12.set_ylabel("Region")
    fig12.tight_layout()
    st.pyplot(fig12)
    st.dataframe(fraud_by_region.reset_index(drop=True))

with tab4:
    st.subheader("Top 5 Most Ordered Categories by Customer Segment")

    category_orders = (
        df_complete_only
        .groupby(["customer_segment", "category_name"])
        .size()
        .reset_index(name="order_count")
    )

    top5_per_segment = (
        category_orders
        .sort_values(["customer_segment", "order_count"], ascending=[True, False])
        .groupby("customer_segment", as_index=False)
        .head(5)
    )

    top5_per_segment["category_name"] = top5_per_segment.groupby("customer_segment")["category_name"].transform(
        lambda x: pd.Categorical(
            x,
            categories=x.iloc[
                np.argsort(
                    -top5_per_segment.loc[x.index, "order_count"].to_numpy()
                )
            ],
            ordered=True,
        )
    )

    sns.set_style("whitegrid")
    g = sns.catplot(
        data=top5_per_segment,
        kind="bar",
        x="order_count",
        y="category_name",
        col="customer_segment",
        sharex=False,
        height=4,
        aspect=1,
        color="#1f77b4",
    )
    g.set_axis_labels("Order Count", "Category")
    g.set_titles("{col_name}")
    g.fig.suptitle("Top 5 Most Ordered Categories by Customer Segment", fontsize=14, y=1.05)
    plt.tight_layout()

    st.pyplot(g.fig)
    st.dataframe(top5_per_segment.reset_index(drop=True))

    st.subheader("==================================================================================================================================")
    st.subheader("Total Profit by Customer Segment")

    profit_by_segment = (
        df_complete_only.dropna(subset=["order_profit_per_order"])
          .groupby("customer_segment", as_index=False)["order_profit_per_order"]
          .sum()
          .rename(columns={"order_profit_per_order":"total_profit"})
          .sort_values("total_profit", ascending=False)
    )

    fig13, ax13 = plt.subplots(figsize=(7,5))
    sns.barplot(
        data=profit_by_segment,
        x="customer_segment",
        y="total_profit",
        color="#1f77b4",
        ax=ax13
    )
    ax13.set_title("Total Profit by Customer Segment")
    ax13.set_xlabel("Customer Segment")
    ax13.set_ylabel("Total Profit")

    for p in ax13.patches:
        value = p.get_height()
        ax13.annotate(
            f"${value:,.0f}",
            (p.get_x() + p.get_width()/2, value),
            ha="center",
            va="bottom",
            xytext=(0,5),
            textcoords="offset points"
        )

    fig13.tight_layout()
    st.pyplot(fig13)
    st.dataframe(profit_by_segment.reset_index(drop=True))
