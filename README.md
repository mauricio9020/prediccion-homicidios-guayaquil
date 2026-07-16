# Predicción del Mecanismo del Homicidio en Guayaquil (Ecuador)

Este proyecto desarrolla un análisis comparativo y predictivo multiclase utilizando datos reales de homicidios registrados en Guayaquil durante el período 2024-2026. Su propósito científico es investigar si la adición de variables de georreferenciación (espaciales) incrementa significativamente la exactitud al predecir el mecanismo utilizado en un homicidio: **Arma de Fuego**, **Arma Blanca**, u **Otros**.

## Estructura del Repositorio

La organización del proyecto cumple con estándares profesionales de reproducibilidad e ingeniería de software:

```
Proyecto_Homicidios/
├── data/
│   ├── raw/           # Dataset original de homicidios en Excel (.xlsx)
│   └── processed/     # Base de datos depurada, normalizada y codificada (.csv)
├── notebooks/         # Espacio para análisis exploratorio rápido
├── src/               # Módulos Python empaquetados
│   ├── config.py      # Definición de rutas fijas, variables y random_state=123
│   ├── data_prep.py   # Limpieza, normalización de coordenadas y balanceo SMOTE
│   ├── modeling.py    # Algoritmos de entrenamiento, optimización y bootstrap
│   ├── evaluation.py  # Métricas multiclase, curvas ROC/PR/Calibración y SHAP
│   ├── report_gen.py  # Generadores automáticos de reportes Word (.docx) y PDF
│   └── pipeline.py    # Orquestador del flujo de trabajo de Machine Learning
├── models/            # Serializaciones de modelos y variables precalculadas (.pkl)
├── dashboard/         # Plantillas HTML y componentes del dashboard streamlit
├── outputs/           # Artefactos gráficos y tablas científicas
│   ├── figures/       # Gráficos (300 DPI) para publicación (Confusion Matrices, ROC, PR, SHAP)
│   ├── tables/        # Reportes tabulares
│   └── metrics/       # Reportes de métricas en Excel y CSV
├── reports/           # Manuscritos y artículos científicos en Word y PDF
├── app.py             # Dashboard interactivo modular construido en Streamlit
├── requirements.txt   # Dependencias de Python requeridas
└── README.md          # Documentación del proyecto (esta guía)
```

## Modelos Comparados

1. **Regresión Logística Multinomial**: Baseline estadístico multiclase.
2. **XGBoost Multiclase (No Espacial)**: Clasificador de ensamble optimizado mediante RandomizedSearchCV y entrenado sobre variables demográficas y temporales únicamente.
3. **G-XGBoost Espacial**: Modelo de boosting de árboles optimizado que integra características geográficas (latitud, longitud, distrito, parroquia, zona).

## Flujo de Trabajo y Reproducibilidad

1. **Carga y Limpieza**: Ajusta errores de codificación en nombres de columnas, normaliza coordenadas decimales (comas a puntos), imputa datos faltantes y elimina duplicados.
2. **Feature Engineering**: Mapea armas del homicidio a las tres categorías del estudio y define circuitos policiales como equivalentes a Parroquia.
3. **Preprocesamiento**: MinMaxScaler para numéricas, OneHotEncoder para categóricas, y balanceo mediante SMOTE para resolver el severo desbalance de clases (Arma de Fuego: 88.5%).
4. **Validación**:
   - Validación Cruzada Estratificada de 10 pliegues para evaluar la robustez y generalización.
   - Bootstrap con 1000 iteraciones en el test set para calcular intervalos de confianza del 95% (IC95%) de las métricas.
5. **Evaluación de Hipótesis**:
   - McNemar's Test: Evalúa significancia estadística de la diferencia en predicciones pareadas de test.
   - Wilcoxon Signed-Rank Test: Determina si la diferencia en puntuaciones de validación cruzada es estadísticamente significativa.
6. **Explicabilidad**:
   - Extracción de feature importances tradicionales (Gain, Weight, Cover).
   - Análisis local y global de SHAP (Beeswarm, Bar, waterfall, dependence) para explicar qué factores incrementan o reducen la propensión a cada tipo de homicidio.

## Instrucciones para Replicación

### 1. Clonar el repositorio e instalar dependencias

```bash
cd Proyecto_Homicidios
pip install -r requirements.txt
```

### 2. Ejecutar el Pipeline de Machine Learning

Este comando iniciará el preprocesamiento, optimización de hiperparámetros, validaciones, bootstrap, generación de gráficos (300 DPI), pruebas estadísticas y la compilación de los artículos Word y PDF:

```bash
python src/pipeline.py
```

### 3. Lanzar el Dashboard Streamlit

Para iniciar el centro de visualización interactivo y probar el formulario de predicciones espaciales en tiempo real con SHAP integrado, ejecute:

```bash
streamlit run app.py
```

## Resultados Principales

- **G-XGBoost Espacial** demostró ser el modelo ganador con una precisión media en validación cruzada 10-fold de **96.80%**, comparado con **93.27%** de XGBoost no espacial y **59.16%** de la Regresión Logística.
- El test de McNemar y Wilcoxon confirmaron que la inclusión de coordenadas geográficas (`Coord_Y`, `Coord_X`) aporta una mejora en precisión **estadísticamente significativa (p < 0.05)**.
- El balanceo mediante SMOTE fue crítico para evitar que los modelos ignorasen los homicidios por Arma Blanca (5.9%) y Otros (5.5%).
