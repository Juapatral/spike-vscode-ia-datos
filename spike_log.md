# Resultado del spike

## Maquina

- SO: Windows 11
- Version: 23H2
- RAM aproximada: 16 GB
- CPU aproximada: Intel i7 / AMD Ryzen 7
- Terminal: PowerShell 7
- Tiempo total: 3.5 horas (aprox. 5:27pm a 8:52pm)

## Cuentas usadas

- Google: Sí (Gemini / Code Assist)
- GitHub: Sí (Control de versiones y deploy de Streamlit)
- Streamlit: Sí (Cloud Deploy)
- Otras: Codex (ChatGPT)

## Herramientas

| Herramienta | Funciono | Friccion | Veredicto |
|---|---|---|---|
| VS Code aislado | Sí | Baja | Aprobado |
| uv | Sí | Baja | Recomendado |
| Python 3.12 | Sí | Baja | Aprobado |
| Gemini Code Assist | Sí | Media | Aprobado (con guía de rutas) |
| Gemini web | Sí | Baja | Aprobado |
| Copilot Free | No | N/A | No probado |
| Jupyter | Sí | Alta | Aprobado con fallback (Codex) |
| Plotly HTML | Sí | Baja | Aprobado |
| Streamlit local | Sí | Baja | Aprobado |
| Streamlit Cloud | Sí | Baja | Aprobado |

## Evidencia

- Notebook: `notebooks/analisis_rfm_kmeans_codex.ipynb`
- HTML: `outputs/ventas_por_mes.html`, `ventas_mensuales_gemini.html`
- Streamlit local: `app_codex/app.py`, `app_gemini/app.py`
- Deploy: Streamlit Cloud Dashboard funcional.
- Capturas utiles: Generación de clusters KMeans y validación del codo.

## Riesgos para clase

- Gemini tiende a crear archivos nuevos en lugar de editar los existentes si no se especifica con Ctrl+I.
- La edición de Notebooks (.ipynb) falla frecuentemente desde el chat; se recomienda enseñar el uso de archivos .py puros.
- Consumo elevado de tokens en sesiones largas puede degradar la calidad de la respuesta.
- Dependencias de visualización (Plotly) en Jupyter requieren configuración adicional del kernel.
- Más notas leer el archivo de comentarios de JP.

## Decision sugerida

- [x] Aprobar ruta principal.
- [x] Aprobar con fallback (Codex).
- [x] Evaluar si Codex se utiliza para vibecoding completo y más grande.

## Notas para produccion

- Priorizar el uso de `uv` para evitar conflictos de librerías en los estudiantes.
- Establecer una convención clara de carpetas (`src/`, `data/`, `outputs/`) desde el primer prompt.
- Enseñar a los estudiantes que Gemini es más robusto en lógica de negocio pero Codex es más rápido en generación de estructura.
- Utilizar el "Método del Codo" automatizado en KMeans para dar rigor profesional al análisis.
- Más notas leer el archivo de comentarios de JP.
- El deploy es sencillo pero hay que conectar streamlit con github y que el repo esté público. Este es mi [app](https://spike-vscode-ia-datos-juapatral-20260516.streamlit.app/)