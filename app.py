import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="E-Commerce Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

customers = pd.read_csv("customers_clean.csv")
products = pd.read_csv("products_clean.csv")
orders = pd.read_csv("orders_clean.csv")
returns = pd.read_csv("returns_clean.csv")

orders["Order_Date"] = pd.to_datetime(orders["Order_Date"])

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Select Dashboard",
    [
        "Executive Dashboard",
        "Sales Analytics",
        "Customer Analytics",
        "Product Analytics",
        "Forecasting"
    ]
)

# -------------------------------------------------
# EXECUTIVE DASHBOARD
# -------------------------------------------------

if page == "Executive Dashboard":

    st.title("📊 Executive Dashboard")

    revenue = orders["Sales_Amount"].sum()
    total_orders = orders["Order_ID"].nunique()
    total_customers = orders["Customer_ID"].nunique()
    avg_order_value = revenue / total_orders

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Revenue", f"₹{revenue/1000000:.1f}M")
    col2.metric("Orders", f"{total_orders/1000:.1f}K")
    col3.metric("Customers", f"{total_customers/1000:.1f}K")
    col4.metric("Avg Order Value", f"₹{avg_order_value:,.0f}")

    st.markdown("---")

    # Revenue by Category
    category_sales = (
        orders.merge(products, on="Product_ID")
        .groupby("Category")["Sales_Amount"]
        .sum()
        .reset_index()
    )

    colA, colB = st.columns(2)

    with colA:
        fig = px.pie(
            category_sales,
            names="Category",
            values="Sales_Amount",
            title="Revenue Contribution by Category"
        )
        st.plotly_chart(fig, use_container_width=True)

    with colB:
        monthly_sales = (
            orders.groupby(
                orders["Order_Date"].dt.to_period("M")
            )["Sales_Amount"]
            .sum()
            .reset_index()
        )

        monthly_sales["Order_Date"] = monthly_sales["Order_Date"].astype(str)

        fig2 = px.line(
            monthly_sales,
            x="Order_Date",
            y="Sales_Amount",
            title="Monthly Revenue Trend"
        )

        st.plotly_chart(fig2, use_container_width=True)

    # Top Products
    top_products = (
        orders.merge(products, on="Product_ID")
        .groupby("Product_Name")["Sales_Amount"]
        .sum()
        .reset_index()
        .sort_values("Sales_Amount", ascending=False)
        .head(10)
    )

    fig3 = px.bar(
        top_products,
        x="Sales_Amount",
        y="Product_Name",
        orientation="h",
        title="Top 10 Revenue Generating Products"
    )

    st.plotly_chart(fig3, use_container_width=True)

# -------------------------------------------------
# SALES ANALYTICS
# -------------------------------------------------

elif page == "Sales Analytics":

    st.title("📈 Sales Analytics")

    monthly_sales = (
        orders.groupby(
            orders["Order_Date"].dt.to_period("M")
        )["Sales_Amount"]
        .sum()
        .reset_index()
    )

    monthly_sales["Order_Date"] = monthly_sales["Order_Date"].astype(str)

    fig = px.line(
        monthly_sales,
        x="Order_Date",
        y="Sales_Amount",
        title="Monthly Sales Trend"
    )

    st.plotly_chart(fig, use_container_width=True)

    category_sales = (
        orders.merge(products, on="Product_ID")
        .groupby("Category")["Sales_Amount"]
        .sum()
        .reset_index()
    )

    fig2 = px.bar(
        category_sales,
        x="Category",
        y="Sales_Amount",
        title="Revenue by Category"
    )

    st.plotly_chart(fig2, use_container_width=True)

# -------------------------------------------------
# CUSTOMER ANALYTICS
# -------------------------------------------------

elif page == "Customer Analytics":

    st.title("👥 Customer Analytics")

    total_customers = customers["Customer_ID"].nunique()

    st.metric(
        "Total Customers",
        f"{total_customers/1000:.1f}K"
    )

    state_customers = (
        customers.groupby("State")["Customer_ID"]
        .count()
        .reset_index()
    )

    fig = px.bar(
        state_customers,
        x="State",
        y="Customer_ID",
        title="Customers by State"
    )

    st.plotly_chart(fig, use_container_width=True)

    gender_dist = customers["Gender"].value_counts()

    fig2 = px.pie(
        names=gender_dist.index,
        values=gender_dist.values,
        title="Customer Gender Distribution"
    )

    st.plotly_chart(fig2, use_container_width=True)

# -------------------------------------------------
# PRODUCT ANALYTICS
# -------------------------------------------------

elif page == "Product Analytics":

    st.title("📦 Product Analytics")

    total_products = products["Product_ID"].nunique()

    st.metric(
        "Total Products",
        total_products
    )

    category_count = (
        products.groupby("Category")["Product_ID"]
        .count()
        .reset_index()
    )

    fig = px.pie(
        category_count,
        names="Category",
        values="Product_ID",
        title="Products by Category"
    )

    st.plotly_chart(fig, use_container_width=True)

    top_products = (
        orders.merge(products, on="Product_ID")
        .groupby("Product_Name")["Sales_Amount"]
        .sum()
        .reset_index()
        .sort_values("Sales_Amount", ascending=False)
        .head(10)
    )

    fig2 = px.bar(
        top_products,
        x="Sales_Amount",
        y="Product_Name",
        orientation="h",
        title="Top 10 Best Selling Products"
    )

    st.plotly_chart(fig2, use_container_width=True)

# -------------------------------------------------
# FORECASTING
# -------------------------------------------------

elif page == "Forecasting":

    st.title("🔮 Sales Forecasting")

    st.write(
        "Future sales prediction generated using Machine Learning."
    )

    st.image(
        "forecast_chart.png",
        caption="Future Revenue Forecast",
        use_container_width=True
    )

   