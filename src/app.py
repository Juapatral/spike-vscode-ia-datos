import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Configuración de la página
st.set_page_config(page_title="E-commerce Analytics | Gemini", layout="wide")

# Rutas
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

@st.cache_data
def load_data():
    """Carga y limpia los datos base."""
    customers = pd.read_csv(DATA_DIR / "customers.csv")
    orders = pd.read_csv(DATA_DIR / "orders.csv")
    payments = pd.read_csv(DATA_DIR / "order_payments.csv")
    
    # Limpieza básica
    orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
    orders = orders[orders['order_status'] == 'delivered']
    
    # Consolidar pagos
    pagos_orden = payments.groupby('order_id')['payment_value'].sum().reset_index()
    
    # Unir datos
    df = orders.merge(customers, on='customer_id', how='inner')
    df = df.merge(pagos_orden, on='order_id', how='left')
    df['payment_value'] = df['payment_value'].fillna(0)
    
    return df

def get_rfm_segments(df_filtered):
    """Calcula RFM y Clusters sobre los datos filtrados."""
    if df_filtered.empty:
        return pd.DataFrame()
        
    fecha_ref = df_filtered['order_purchase_timestamp'].max() + pd.Timedelta(days=1)
    rfm = df_filtered.groupby('customer_unique_id').agg({
        'order_purchase_timestamp': lambda x: (fecha_ref - x.max()).days,
        'order_id': 'nunique',
        'payment_value': 'sum'
    }).reset_index()
    rfm.columns = ['customer_unique_id', 'recency', 'frequency', 'monetary']
    
    # KMeans
    if len(rfm) >= 4:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(rfm[['recency', 'frequency', 'monetary']])
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        rfm['cluster'] = kmeans.fit_predict(scaled)
        rfm['cluster'] = rfm['cluster'].astype(str)
    else:
        rfm['cluster'] = "0"
        
    return rfm

# --- UI ---
st.title("📊 E-commerce Strategic Dashboard")
st.warning("⚠️ **Advertencia:** El dataset utilizado es sintético y tiene fines exclusivamente educativos.")

# Sidebar
df_raw = load_data()
departamentos = sorted(df_raw['customer_department'].unique())
dept_selected = st.sidebar.multiselect("Filtrar por Departamento", departamentos, default=None)

if dept_selected:
    df_filtered = df_raw[df_raw['customer_department'].isin(dept_selected)].copy()
else:
    df_filtered = df_raw.copy()

# KPIs Principales
col1, col2, col3 = st.columns(3)
with col1:
    total_sales = df_filtered['payment_value'].sum()
    st.metric("Ventas Totales", f"${total_sales:,.0f} COP")
with col2:
    total_orders = df_filtered['order_id'].nunique()
    st.metric("Órdenes Entregadas", f"{total_orders:,}")
with col3:
    ticket_promedio = total_sales / total_orders if total_orders > 0 else 0
    st.metric("Ticket Promedio", f"${ticket_promedio:,.0f} COP")

# TABS
tab_ventas, tab_segmentos = st.tabs(["📈 Análisis de Ventas", "👥 Segmentación RFM"])

with tab_ventas:
    st.subheader("Evolución de Ventas Mensuales")
    
    if not df_filtered.empty:
        df_filtered['mes'] = df_filtered['order_purchase_timestamp'].dt.to_period('M').dt.to_timestamp()
        ventas_mes = df_filtered.groupby('mes')['payment_value'].sum().reset_index()
        
        fig = px.line(
            ventas_mes, x='mes', y='payment_value',
            markers=True,
            labels={'payment_value': 'Ventas (COP)', 'mes': 'Mes'},
            template="plotly_white",
            color_discrete_sequence=['#00CC96']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("Ver datos mensuales"):
            st.dataframe(ventas_mes, use_container_width=True)
    else:
        st.info("No hay datos para mostrar con los filtros actuales.")

with tab_segmentos:
    st.subheader("Explorador de Segmentos de Clientes")
    
    rfm_data = get_rfm_segments(df_filtered)
    
    if not rfm_data.empty:
        # Gráfico 3D de Clusters
        fig_cluster = px.scatter_3d(
            rfm_data, 
            x='recency', y='frequency', z='monetary',
            color='cluster',
            title="Clusters KMeans (R vs F vs M)",
            labels={'recency': 'Recencia (Días)', 'frequency': 'Frecuencia', 'monetary': 'Monetario'},
            opacity=0.7,
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        st.plotly_chart(fig_cluster, use_container_width=True)
        
        # Resumen de segmentos
        st.write("**Resumen por Cluster:**")
        resumen = rfm_data.groupby('cluster').agg({
            'customer_unique_id': 'count',
            'recency': 'mean',
            'frequency': 'mean',
            'monetary': 'mean'
        }).rename(columns={'customer_unique_id': 'Cant. Clientes'}).round(2)
        
        st.dataframe(resumen, use_container_width=True)
    else:
        st.info("Selecciona al menos un departamento para ver la segmentación.")

st.sidebar.markdown("---")
st.sidebar.caption("Senior Data Analyst | Proyect Gemini v1.0")