import pandas as pd

def sumar_ventas_por_mes(df, fecha_col='fecha', ventas_col='ventas'):
    """
    Suma las ventas agrupadas por mes en un DataFrame de pandas.
    
    Args:
        df (pd.DataFrame): DataFrame que contiene los datos de ventas.
        fecha_col (str): Nombre de la columna con las fechas.
        ventas_col (str): Nombre de la columna con los montos de ventas.
        
    Returns:
        pd.Series: Un objeto Series con el total de ventas indexado por mes.
    """
    # Asegurarse de que la columna de fecha sea de tipo datetime
    df[fecha_col] = pd.to_datetime(df[fecha_col])
    
    # Agrupar por periodo mensual y sumar la columna de ventas
    ventas_mensuales = df.groupby(df[fecha_col].dt.to_period('M'))[ventas_col].sum()
    
    return ventas_mensuales
