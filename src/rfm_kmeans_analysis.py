"""
Análisis de segmentación RFM y Clustering KMeans para E-commerce.
Este script carga datos, realiza limpieza según el diccionario, 
calcula métricas de negocio y genera visualizaciones interactivas.
"""

from pathlib import Path
import numpy as np
import pandas as pd

try:
    import plotly.express as px
except ImportError as exc:
    raise SystemExit(
        "Falta instalar plotly. Ejecuta: uv add plotly"
    ) from exc

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise SystemExit(
        "Falta instalar scikit-learn. Ejecuta: uv add scikit-learn"
    ) from exc

# Configuración de rutas relativas al proyecto
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"

def read_data() -> dict[str, pd.DataFrame]:
    """Lee los archivos CSV fuente necesarios para el análisis."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"No se encontró la carpeta de datos en: {DATA_DIR}")
        
    return {
        "customers": pd.read_csv(DATA_DIR / "customers.csv", dtype={"customer_postal_code_prefix": "string"}),
        "orders": pd.read_csv(DATA_DIR / "orders.csv"),
        "order_items": pd.read_csv(DATA_DIR / "order_items.csv"),
        "order_payments": pd.read_csv(DATA_DIR / "order_payments.csv"),
    }

def clean_data(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Aplica limpieza de datos basada en las reglas de data/data_dictionary.csv."""
    customers = tables["customers"].copy()
    orders = tables["orders"].copy()
    order_items = tables["order_items"].copy()
    order_payments = tables["order_payments"].copy()

    # Limpieza de Clientes: Normalización de texto y manejo de nulos
    for column in ["customer_city", "customer_department"]:
        customers[column] = (
            customers[column]
            .fillna("unknown")
            .astype("string")
            .str.strip()
            .str.lower()
        )

    date_columns = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for column in date_columns:
        orders[column] = pd.to_datetime(orders[column], errors="coerce")

    # Limpieza de Items
    order_items["shipping_limit_date"] = pd.to_datetime(
        order_items["shipping_limit_date"], errors="coerce"
    )
    order_items["product_category"] = (
        order_items["product_category"].fillna("unknown").astype("string").str.strip().str.lower()
    )
    order_items["price"] = pd.to_numeric(order_items["price"], errors="coerce").fillna(0)
    order_items["freight_value"] = pd.to_numeric(
        order_items["freight_value"], errors="coerce"
    ).fillna(0)

    # Limpieza de Pagos: Eliminación de duplicados y normalización
    order_payments = order_payments.drop_duplicates()
    order_payments["payment_type"] = (
        order_payments["payment_type"].fillna("unknown").astype("string").str.strip().str.lower()
    )
    order_payments["payment_installments"] = pd.to_numeric(
        order_payments["payment_installments"], errors="coerce"
    ).fillna(1)
    order_payments["payment_value"] = pd.to_numeric(
        order_payments["payment_value"], errors="coerce"
    ).fillna(0)

    return {
        "customers": customers,
        "orders": orders,
        "order_items": order_items,
        "order_payments": order_payments,
    }

def build_order_level_dataset(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Agrega tablas 1:N al grano de orden y une atributos de cliente."""
    item_features = (
        tables["order_items"]
        .groupby("order_id", as_index=False)
        .agg(
            item_count=("order_item_id", "count"),
            gross_merchandise_value=("price", "sum"),
            freight_value=("freight_value", "sum"),
            unique_products=("product_id", "nunique"),
            main_category=("product_category", lambda x: x.mode().iat[0] if not x.mode().empty else "unknown"),
        )
    )

    payment_features = (
        tables["order_payments"]
        .groupby("order_id", as_index=False)
        .agg(
            monetary=("payment_value", "sum"),
            payment_count=("payment_sequential", "count"),
            max_installments=("payment_installments", "max"),
            main_payment_type=("payment_type", lambda x: x.mode().iat[0] if not x.mode().empty else "unknown"),
        )
    )

    return (
        tables["orders"]
        .merge(tables["customers"], on="customer_id", how="left")
        .merge(item_features, on="order_id", how="left")
        .merge(payment_features, on="order_id", how="left")
    )

def build_rfm(order_level: pd.DataFrame) -> pd.DataFrame:
    """Crea una tabla RFM compacta usando solo órdenes entregadas."""
    # Filtro crítico: Solo órdenes entregadas y con ID único de cliente
    delivered = order_level.loc[
        (order_level["order_status"] == "delivered")
        & order_level["customer_unique_id"].notna()
        & order_level["order_purchase_timestamp"].notna()
    ].copy()

    delivered["monetary"] = delivered["monetary"].fillna(0)
    analysis_date = delivered["order_purchase_timestamp"].max() + pd.Timedelta(days=1)

    # Agregación por customer_unique_id (persistente)
    rfm = (
        delivered.groupby("customer_unique_id", as_index=False)
        .agg(
            recency=("order_purchase_timestamp", lambda x: (analysis_date - x.max()).days),
            frequency=("order_id", "nunique"),
            monetary=("monetary", "sum"),
        )
    )

    # Cálculo de scores (Cuartiles)
    rfm["r_score"] = pd.qcut(rfm["recency"], q=4, labels=[4, 3, 2, 1], duplicates="drop")
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), q=4, labels=[1, 2, 3, 4], duplicates="drop")
    rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), q=4, labels=[1, 2, 3, 4], duplicates="drop")
    rfm["rfm_score"] = (
        rfm[["r_score", "f_score", "m_score"]]
        .astype("Int64")
        .astype("string")
        .agg("".join, axis=1)
    )

    return rfm


def evaluate_kmeans_elbow(
    rfm: pd.DataFrame,
    min_clusters: int = 2,
    max_clusters: int = 10,
) -> pd.DataFrame:
    """Calcula inercias para validar el numero de clusters con el metodo del codo."""
    if len(rfm) < 2:
        return pd.DataFrame(columns=["k", "inertia", "inertia_drop_pct", "elbow_distance"])

    features = rfm[["recency", "frequency", "monetary"]].copy()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    max_possible_clusters = min(max_clusters, len(rfm))
    min_possible_clusters = min(min_clusters, max_possible_clusters)
    k_values = list(range(min_possible_clusters, max_possible_clusters + 1))

    inertias = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(scaled_features)
        inertias.append(model.inertia_)

    elbow = pd.DataFrame({"k": k_values, "inertia": inertias})
    elbow["inertia_drop_pct"] = elbow["inertia"].pct_change().mul(-100)

    if len(elbow) <= 2:
        elbow["elbow_distance"] = 0.0
        return elbow

    x = elbow["k"].to_numpy(dtype=float)
    y = elbow["inertia"].to_numpy(dtype=float)
    x_scaled = (x - x.min()) / (x.max() - x.min())
    y_scaled = (y - y.min()) / (y.max() - y.min())

    x1, y1 = x_scaled[0], y_scaled[0]
    x2, y2 = x_scaled[-1], y_scaled[-1]
    numerator = np.abs(
        (y2 - y1) * x_scaled - (x2 - x1) * y_scaled + x2 * y1 - y2 * x1
    )
    denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    distances = numerator / denominator
    elbow["elbow_distance"] = distances

    return elbow


def choose_cluster_count(elbow: pd.DataFrame, fallback: int = 4) -> int:
    """Selecciona k usando la mayor distancia a la linea entre el primer y ultimo punto."""
    if elbow.empty:
        return fallback

    if elbow["elbow_distance"].max() == 0:
        return int(elbow["k"].iloc[0])

    return int(elbow.loc[elbow["elbow_distance"].idxmax(), "k"])


def add_kmeans_segments(rfm: pd.DataFrame, n_clusters: int | None = None) -> pd.DataFrame:
    """Escala features y asigna clusters mediante KMeans."""
    if len(rfm) < 2:
        segmented = rfm.copy()
        segmented["cluster"] = 0
        segmented["segment_name"] = "segmento_1"
        return segmented

    features = rfm[["recency", "frequency", "monetary"]].copy()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    if n_clusters is None:
        elbow = evaluate_kmeans_elbow(rfm)
        n_clusters = choose_cluster_count(elbow, fallback=min(4, len(rfm)))

    fitted_clusters = min(n_clusters, len(rfm))
    model = KMeans(n_clusters=fitted_clusters, random_state=42, n_init=10)

    segmented = rfm.copy()
    segmented["cluster"] = model.fit_predict(scaled_features)

    cluster_summary = (
        segmented.groupby("cluster")
        .agg(
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            customers=("customer_unique_id", "count"),
        )
        .sort_values(["avg_monetary", "avg_frequency"], ascending=False)
        .reset_index()
    )
    cluster_summary["segment_name"] = [
        f"segmento_{rank + 1}" for rank in range(len(cluster_summary))
    ]

    segmented = segmented.merge(
        cluster_summary[["cluster", "segment_name"]], on="cluster", how="left"
    )
    return segmented

def export_monthly_sales_chart(order_level: pd.DataFrame) -> Path:
    """Genera un gráfico de ventas mensuales con Plotly y lo exporta a HTML."""
    delivered = order_level.loc[order_level["order_status"] == "delivered"].copy()
    delivered["monetary"] = delivered["monetary"].fillna(0)
    delivered["purchase_month"] = (
        delivered["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
    )

    monthly_sales = (
        delivered.groupby("purchase_month", as_index=False)
        .agg(sales=("monetary", "sum"), orders=("order_id", "nunique"))
        .sort_values("purchase_month")
    )

    fig = px.line(
        monthly_sales,
        x="purchase_month",
        y="sales",
        markers=True,
        title="Ventas entregadas por mes",
        labels={
            "purchase_month": "Mes",
            "sales": "Ventas COP",
            "orders": "Ordenes",
        },
        hover_data={"orders": True, "sales": ":,.0f"},
    )
    fig.update_layout(template="plotly_white", hovermode="x unified")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "ventas_por_mes.html"
    fig.write_html(output_path, include_plotlyjs="cdn")

    monthly_sales.to_csv(OUTPUT_DIR / "ventas_por_mes.csv", index=False)
    return output_path

def main() -> None:
    """Orquestador principal del análisis."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # Crea la carpeta outputs si no existe

    tables = clean_data(read_data())
    order_level = build_order_level_dataset(tables)
    rfm = build_rfm(order_level)
    segmented_rfm = add_kmeans_segments(rfm)
    chart_path = export_monthly_sales_chart(order_level)

    order_level.to_csv(OUTPUT_DIR / "order_level_dataset.csv", index=False)
    segmented_rfm.to_csv(OUTPUT_DIR / "rfm_kmeans_segments.csv", index=False)

    print(f"Clientes segmentados: {len(segmented_rfm):,}")
    print(f"HTML exportado: {chart_path}")
    print(f"RFM exportado: {OUTPUT_DIR / 'rfm_kmeans_segments.csv'}")

if __name__ == "__main__":
    main()
