import pandas as pd
from pathlib import Path
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

# Configuración para una ejecución limpia
warnings.filterwarnings('ignore')

# Definición de rutas relativas al proyecto
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

def procesar_analisis_ecommerce():
    """
    Orquestador de análisis: Carga, Limpieza, RFM, Clustering y Visualización.
    """
    print("🚀 Iniciando pipeline de análisis de datos...")

    # 1. Carga de datos
    try:
        customers = pd.read_csv(DATA_DIR / "customers.csv", dtype={"customer_postal_code_prefix": "string"})
        orders = pd.read_csv(DATA_DIR / "orders.csv")
        order_payments = pd.read_csv(DATA_DIR / "order_payments.csv")
        # order_items se carga por requerimiento, aunque el valor monetario se extraerá de pagos
        order_items = pd.read_csv(DATA_DIR / "order_items.csv")
    except FileNotFoundError as e:
        print(f"❌ Error: No se encontraron los archivos en {DATA_DIR}. {e}")
        return

    # 2. Limpieza y Transformación (Siguiendo reglas del Data Dictionary)
    print("🧹 Limpiando datos y aplicando reglas de negocio...")
    
    # Convertir fechas a datetime
    orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
    
    # Filtrar solo órdenes entregadas para métricas de lealtad y ventas efectivas
    orders = orders[orders['order_status'] == 'delivered'].copy()

    # Eliminar duplicados en pagos según recomendación del diccionario
    order_payments = order_payments.drop_duplicates()
    
    # Consolidar valor total pagado por cada orden
    pagos_por_orden = order_payments.groupby('order_id')['payment_value'].sum().reset_index()

    # Unir tablas para crear el dataset maestro
    # Usamos customer_unique_id para identificar personas reales a través del tiempo
    df_master = orders.merge(
        customers[['customer_id', 'customer_unique_id']], 
        on='customer_id', 
        how='inner'
    ).merge(
        pagos_por_orden, 
        on='order_id', 
        how='left'
    )
    
    df_master['payment_value'] = df_master['payment_value'].fillna(0)

    # 3. Cálculo de Métricas RFM (Recency, Frequency, Monetary)
    print("📊 Calculando métricas RFM por cliente único...")
    fecha_referencia = df_master['order_purchase_timestamp'].max() + pd.Timedelta(days=1)
    
    rfm = df_master.groupby('customer_unique_id').agg({
        'order_purchase_timestamp': lambda x: (fecha_referencia - x.max()).days, # Recencia
        'order_id': 'nunique',                                                 # Frecuencia
        'payment_value': 'sum'                                                # Monetario
    }).reset_index()
    
    rfm.columns = ['customer_unique_id', 'recency', 'frequency', 'monetary']

    # 4. Clustering KMeans
    print("🤖 Segmentando clientes con KMeans...")
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['recency', 'frequency', 'monetary']])

    # Definimos 4 clusters para la segmentación inicial
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm['cluster'] = kmeans.fit_predict(rfm_scaled)

    # 5. Visualización: Evolución de Ventas
    print("📈 Generando gráfico de ventas mensuales...")
    df_master['mes'] = df_master['order_purchase_timestamp'].dt.to_period('M').dt.to_timestamp()
    ventas_mensuales = df_master.groupby('mes')['payment_value'].sum().reset_index()

    fig = px.line(
        ventas_mensuales, x='mes', y='payment_value',
        title='Tendencia de Ventas Mensuales (COP) - Órdenes Entregadas',
        labels={'mes': 'Mes', 'payment_value': 'Ventas Totales'},
        markers=True, template='plotly_white'
    )

    # 6. Exportación de Resultados
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_html(OUTPUT_DIR / "ventas_mensuales_gemini.html")
    rfm.to_csv(OUTPUT_DIR / "segmentacion_clientes_rfm.csv", index=False)

    print(f"✅ Proceso completado.")
    print(f"   - Gráfico: {OUTPUT_DIR}/ventas_mensuales_gemini.html")
    print(f"   - Datos: {OUTPUT_DIR}/segmentacion_clientes_rfm.csv")

if __name__ == "__main__":
    procesar_analisis_ecommerce()