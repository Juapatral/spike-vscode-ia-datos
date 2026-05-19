from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from rfm_kmeans_analysis import (  # noqa: E402
    add_kmeans_segments,
    build_order_level_dataset,
    build_rfm,
    clean_data,
    read_data,
)


st.set_page_config(
    page_title="Análisis de Ventas y Clientes",
    page_icon="bar_chart",
    layout="wide",
)


@st.cache_data(show_spinner="Cargando y procesando datos...")
def load_model_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    tables = clean_data(read_data())
    order_level = build_order_level_dataset(tables)
    segments = add_kmeans_segments(build_rfm(order_level))
    return order_level, segments


def format_cop(value: float) -> str:
    return f"${value:,.0f} COP"


def format_segment_name(cluster: int) -> str:
    segment_names = {
        0: "Nuevos",
        1: "Dormidos",
        2: "VIP",
        3: "Leales que se enfrían",
        4: "Ocasionales",
    }
    return segment_names.get(int(cluster), f"Segmento {cluster}")


def get_delivered_orders(order_level: pd.DataFrame, department: str) -> pd.DataFrame:
    delivered = order_level.loc[order_level["order_status"] == "delivered"].copy()
    delivered["customer_department"] = delivered["customer_department"].fillna("unknown")
    delivered["monetary"] = delivered["monetary"].fillna(0)

    if department != "Todos":
        delivered = delivered.loc[delivered["customer_department"] == department].copy()

    return delivered


def build_monthly_sales(delivered: pd.DataFrame) -> pd.DataFrame:
    monthly = delivered.copy()
    monthly["purchase_month"] = (
        monthly["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
    )

    monthly_sales = (
        monthly.groupby("purchase_month", as_index=False)
        .agg(
            sales=("monetary", "sum"),
            orders=("order_id", "nunique"),
            customers=("customer_unique_id", "nunique"),
        )
        .sort_values("purchase_month")
    )
    monthly_sales["avg_ticket"] = monthly_sales["sales"] / monthly_sales["orders"]
    return monthly_sales


def show_kpis(delivered: pd.DataFrame) -> None:
    total_sales = delivered["monetary"].sum()
    total_orders = delivered["order_id"].nunique()
    total_customers = delivered["customer_unique_id"].nunique()
    avg_ticket = total_sales / total_orders if total_orders else 0

    col_sales, col_orders, col_customers, col_ticket = st.columns(4)
    col_sales.metric("Ventas entregadas", format_cop(total_sales))
    col_orders.metric("Ordenes entregadas", f"{total_orders:,}")
    col_customers.metric("Clientes unicos", f"{total_customers:,}")
    col_ticket.metric("Ticket promedio", format_cop(avg_ticket))


def show_sales_tab(delivered: pd.DataFrame, department: str) -> None:
    st.subheader("Ventas por mes")
    show_kpis(delivered)

    monthly_sales = build_monthly_sales(delivered)
    if monthly_sales.empty:
        st.info("No hay ordenes entregadas para el departamento seleccionado.")
        return

    fig = px.line(
        monthly_sales,
        x="purchase_month",
        y="sales",
        markers=True,
        title=f"Ventas entregadas por mes - {department}",
        labels={
            "purchase_month": "Mes",
            "sales": "Ventas COP",
            "orders": "Ordenes",
            "customers": "Clientes",
        },
        hover_data={"orders": True, "customers": True, "sales": ":,.0f"},
    )
    fig.update_layout(template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    fig_ticket = px.line(
        monthly_sales,
        x="purchase_month",
        y="avg_ticket",
        markers=True,
        title=f"Ticket promedio por mes - {department}",
        labels={
            "purchase_month": "Mes",
            "avg_ticket": "Ticket promedio COP",
        },
        hover_data={"avg_ticket": ":,.0f"},
    )
    fig_ticket.update_layout(template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig_ticket, use_container_width=True)

    st.dataframe(
        monthly_sales.assign(
            sales=monthly_sales["sales"].round(0),
            avg_ticket=monthly_sales["avg_ticket"].round(0),
        ),
        use_container_width=True,
        hide_index=True,
    )


def show_segments_tab(delivered: pd.DataFrame, segments: pd.DataFrame) -> None:
    st.subheader("Segmentación de clientes")

    customers_in_scope = delivered["customer_unique_id"].dropna().unique()
    scoped_segments = segments.loc[
        segments["customer_unique_id"].isin(customers_in_scope)
    ].copy()

    if scoped_segments.empty:
        st.info("No hay segmentos para el filtro seleccionado.")
        return

    scoped_segments["segment_name"] = scoped_segments["cluster"].apply(format_segment_name)

    summary = (
        scoped_segments.groupby(["cluster", "segment_name"], as_index=False)
        .agg(
            customers=("customer_unique_id", "count"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
        )
        .sort_values("avg_monetary", ascending=False)
    )

    col_bar, col_scatter = st.columns([1, 1])
    with col_bar:
        fig_bar = px.bar(
            summary,
            x="segment_name",
            y="customers",
            color="segment_name",
            title="Clientes por segmento",
            labels={"segment_name": "Segmento", "customers": "Clientes"},
        )
        fig_bar.update_layout(template="plotly_white", showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_scatter:
        fig_scatter = px.scatter(
            scoped_segments,
            x="frequency",
            y="monetary",
            color="segment_name",
            size="recency",
            hover_name="customer_unique_id",
            title="Frecuencia vs valor monetario",
            labels={
                "frequency": "Frecuencia",
                "monetary": "Monetario COP",
                "segment_name": "Segmento",
                "recency": "Recencia",
            },
        )
        fig_scatter.update_layout(template="plotly_white")
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.dataframe(
        summary.round(
            {"avg_recency": 1, "avg_frequency": 2, "avg_monetary": 0}
        ),
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    st.title("Análisis de Ventas y Clientes")

    order_level, segments = load_model_data()

    departments = sorted(
        order_level["customer_department"].fillna("unknown").dropna().unique().tolist()
    )
    department = st.sidebar.selectbox("Departamento", ["Todos", *departments])

    delivered = get_delivered_orders(order_level, department)

    sales_tab, segments_tab = st.tabs(["Ventas por mes", "Segmentación de clientes"])
    with sales_tab:
        show_sales_tab(delivered, department)

    with segments_tab:
        show_segments_tab(delivered, segments)

    st.caption(
        "Dataset para fines educativos. No representa datos reales del mercado colombiano."
    )


if __name__ == "__main__":
    main()
