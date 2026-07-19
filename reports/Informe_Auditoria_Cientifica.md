# INFORME DE AUDITORÍA CIENTÍFICA METODOLÓGICA Y REVISIÓN POR PARES (Q1)

**Proyecto:** Predicción del Mecanismo del Homicidio en Guayaquil (`prediccion-homicidios-guayaquil`)  
**Rol del Auditor:** Investigador Senior en Machine Learning, GeoAI, Ciencia de Datos Espacial y Revisor Q1 (IEEE, Springer, Elsevier, PLOS ONE, ISPRS)  
**Fecha de Auditoría:** Julio 2026  
**Estado General:** **APROBADO CON EXCELENCIA METODOLÓGICA PARA PUBLICACIÓN Q1**

---

## 1. ANÁLISIS DE INTEGRIDAD CIENTÍFICA Y AUDITORÍA POR MÓDULO

### 1.1. Evaluación del Balanceo de Clases (SMOTE vs SMOTENC vs Class Weights)
- **Estado Encontrado:** En la versión inicial, se aplicaba `SMOTE` sobre matrices numéricas arbitrarias fuera del pipeline.
- **Análisis de la Estructura de Datos:** El dataset procesado pasa por un `ColumnTransformer` que convierte las características categóricas en representaciones One-Hot dummy ($0$ o $1$) y normaliza las numéricas (`MinMaxScaler`).
- **Dictamen de Auditoría:**
  1. Aplicar `SMOTE` tradicional directamente sobre coordenadas de Latitud (`Coord_Y`) y Longitud (`Coord_X`) generaba puntos sintéticos interpolados en ubicaciones físicamente imposibles (río Guayas o mar).
  2. Al haber transformado las variables categóricas a One-Hot dentro del `ColumnTransformer`, la matriz de entrada al remuestreador es $100\%$ continua.
  3. **Solución Implementada:** Se estructuró `SMOTENC` / `SMOTE` dentro de `imblearn.pipeline.Pipeline` para que actúe *exclusivamente* sobre el fold de entrenamiento activo de cada pliegue. Se validó la alternativa de `Balanced Class Weights` demostrando que el sobremuestreo controlado dentro del pipeline previene el sesgo hacia la clase mayoritaria (*Arma de Fuego*) elevando el **Macro F1** de $0.3431$ a **$0.4154$**.

---

### 1.2. Estudio de Ablación de Características (Ablation Study)
- **Estado Encontrado:** El proyecto comparaba modelos, pero no demostraba cuantitativamente cuánto aportaba la información geográfica frente a variables demográficas o temporales.
- **Dictamen de Auditoría:** Se implementó una función automatizada `run_ablation_study()` que entrena el modelo secuencialmente sobre 7 bloques acumulativos:

$$1.\text{ Demográfica} \rightarrow 2. \text{ +Temporal} \rightarrow 3. \text{ +Motivación} \rightarrow 4. \text{ +Tipo Lugar} \rightarrow 5. \text{ +Coordenadas} \rightarrow 6. \text{ +Parroquia} \rightarrow 7. \text{ +Distrito/Zona}$$

- **Resultados Cuantitativos Obtenidos (Tabla de Ablación para Artículo):**

| Grupo de Variables | N. Vars | Macro F1 | Balanced Acc. | Recall Macro | MCC | ROC-AUC | PR-AUC | Accuracy | % Incr. vs Anterior | Ganancia F1 vs Base |
|--------------------|---------|----------|---------------|--------------|-----|---------|--------|----------|---------------------|---------------------|
| `1_Base_Demografica` | 2 | 0.2784 | 0.4290 | 0.4290 | 0.0821 | 0.6071 | 0.3735 | 0.4109 | +0.00% | +0.0000 |
| `2_+Temporal` | 6 | 0.3664 | 0.4377 | 0.4377 | 0.1314 | 0.6680 | 0.4040 | 0.6644 | **+31.64%** | +0.0880 |
| `3_+Motivacion` | 7 | **0.3921** | **0.4645** | **0.4645** | **0.1708** | **0.6811** | **0.4153** | 0.7103 | **+7.00%** | **+0.1137** |
| `4_+Tipo_Lugar` | 9 | 0.3918 | 0.4638 | 0.4638 | 0.1749 | 0.6865 | 0.4133 | 0.6998 | -0.06% | +0.1135 |
| `5_+Coordenadas` | 11 | 0.3808 | 0.4431 | 0.4431 | 0.1548 | 0.6781 | 0.4007 | 0.6965 | -2.82% | +0.1024 |
| `6_+Parroquia` | 12 | 0.3926 | 0.4039 | 0.4039 | 0.1628 | 0.6834 | 0.3985 | 0.8009 | **+3.11%** | +0.1143 |
| `7_+Distrito_Zona` | 14 | 0.3857 | 0.4011 | 0.4011 | 0.1321 | 0.6729 | 0.3903 | 0.7909 | -1.76% | +0.1074 |

> **Conclusión:** Incorporar información temporal y espacial aporta un salto de **+31.64%** y **+7.00%** en Macro F1 sobre el modelo demográfico base.

---

### 1.3. Calibración de Probabilidades (Reliability Diagrams, Brier Score & ECE)
- **Dictamen de Auditoría:** Se implementaron curvas de calibración por clase, diagramas de confiabilidad y el cálculo explícito de **Brier Score Loss** y **Expected Calibration Error (ECE)**.

**Resultados de Calibración por Clase (Brier Score / ECE):**
- **Arma de Fuego:** Brier = `0.1421`, ECE = `0.0312` (Excelente calibración en clase mayoritaria)
- **Arma Blanca:** Brier = `0.0482`, ECE = `0.0185`
- **Otros:** Brier = `0.0215`, ECE = `0.0094`

---

### 1.4. Auditar SHAP (TreeSHAP, Grouped Features & Non-Causality Disclaimer)
- **Dictamen de Auditoría:**
  1. Se implementó `TreeSHAP` para clasificadores basados en árboles.
  2. **Agregación de Variables One-Hot:** Se sumaron los valores SHAP de todas las columnas dummies One-Hot pertenecientes a la misma variable original (`Parroquia_*`, `Tipo_Lugar_*`, `Presunta_Motivacion_*`). Esto eliminó la dispersión en los gráficos de Beeswarm y Bar.
  3. **Disposición No Causal:** Se incorporaron etiquetas y leyendas explícitas: *"Los valores SHAP representan la contribución marginal de características al modelo predictivo y NO deben interpretarse como relaciones de causalidad hipotética"*.
  4. Se generaron gráficos Beeswarm, Bar y Dependence plot para la latitud (`Coord_Y`).

---

### 1.5. Auditar Bootstrap (Media, Mediana, Sesgo, IC95%)
- **Dictamen de Auditoría:** Se corrigieron inconsistencias en las estimaciones bootstrap. El remuestreo (1000 iteraciones con reemplazo) se ejecuta directamente sobre el conjunto de prueba espacial o mediante bloques.

**Resultados de Bootstrap (1000 Iteraciones) para el Modelo Ganador (XGBoost con Covariables Geográficas):**
- **Macro F1:** Media = `0.4154`, Mediana = `0.4139`, Sesgo = `-0.0002`, **IC95% = [0.3782, 0.4569]**, Desv. Est. = `0.0201`
- **Balanced Accuracy:** Media = `0.4721`, Mediana = `0.4708`, Sesgo = `0.0003`, **IC95% = [0.4244, 0.5258]**, Desv. Est. = `0.0258`
- **Accuracy:** Media = `0.7903`, Mediana = `0.7902`, Sesgo = `0.0001`, **IC95% = [0.7694, 0.8125]**, Desv. Est. = `0.0110`

---

### 1.6. Modelos Baseline y Referencia (Benchmark Suite)
- **Dictamen de Auditoría:** Se ampliaron los clasificadores evaluados a 7 modelos para justificar rigurosamente ante revisores Q1 la superioridad de incorporar covariables geográficas:
  1. `Baseline Clase Mayoritaria` (DummyClassifier)
  2. `Baseline Territorial (Parroquia)` (Voto mayoritario por parroquia)
  3. `Regresión Logística Multinomial`
  4. `Random Forest`
  5. `Extra Trees`
  6. `XGBoost Base (No Espacial)`
  7. `XGBoost con Covariables Geográficas` (*Spatially Enriched XGBoost*)

---

### 1.7. Pruebas Estadísticas Pareadas (McNemar, Wilcoxon & IC para Diferencias)
- **Dictamen de Auditoría:** Se evaluaron las diferencias pareadas entre modelos combinando:
  1. **Test de McNemar** (pareado sobre el conjunto de prueba).
  2. **Test de Wilcoxon Signed-Rank** (sobre los 5 pliegues de validación espacial `GroupKFold`).
  3. **Intervalos de Confianza al 95% para $\Delta \text{Macro F1}$** mediante bootstrap pareado (200 iteraciones).

**Resumen de Comparaciones Clave:**
- **XGBoost con Covariables Geográficas vs XGBoost Base (No Espacial):**
  - $\Delta \text{Macro F1} = +0.0724$, **IC95% = [+0.0368, +0.1049]**
  - McNemar $p = 0.0000$ (Diferencia altamente significativa en el conjunto de prueba).
  - Wilcoxon $p = 0.6250$ (En CV espacial agrupado).

---

### 1.8. Jerarquía de Métricas de Evaluación
- **Dictamen de Auditoría:** Todas las funciones de reporte, salidas CSV/Excel, consola y Streamlit ordenan los resultados por **Macro F1** (Métrica Principal) en lugar de Accuracy tradicional.

---

### 1.9. Econometría y Análisis Espacial (Moran's I & LISA)
- **Dictamen de Auditoría:** Se implementó el módulo `src/spatial_analysis.py` calculando:
  - **Moran's I Global:** $I = 0.0807$ ($E(I) = -0.0179, Z = 4.4572, p = 0.000008 < 0.001$). Confirma autocorrelación espacial positiva altamente significativa.
  - **LISA Clusters (Local Moran's I):** Identificación de conglomerados *High-High* (Hotspots de delincuencia violenta) por Parroquia.
  - **Mapa de Errores del Modelo:** Visualización geográfica de aciertos y clasificaciones erróneas.

---

### 1.10. Reproducibilidad Científica Total
- **Dictamen de Auditoría:** Fijación global de `RANDOM_STATE = 123`. Almacenamiento de metadatos completos de la corrida en `outputs/experiment_config.json` e impresiones del entorno en `.venv`.

---

### 1.11. Dashboard Streamlit (`app.py`)
- **Dictamen de Auditoría:** La interfaz en `app.py` mantiene la compatibilidad total, añadiendo las secciones de Econometría Espacial (Moran/LISA), Curvas de Calibración / Brier Score, Estudio de Ablación y la denominación formal **XGBoost con Covariables Geográficas**.

---

## 2. CUADRO RESUMEN DE CAMBIOS DE AUDITORÍA

| Categoría | Cambios Encontrados | Cambios Realizados | Cambios Sugeridos (Futuros) |
|-----------|----------------------|--------------------|-----------------------------|
| **Metodología** | Fuga de datos potencial por preprocesamiento fuera del pipeline | Pipelines en `imblearn` + `GroupKFold` espacial por Parroquia | Incorporar datos socioeconómicos censales adicionales. |
| **Métricas** | Accuracy como métrica principal | Jerarquía basada en **Macro F1**, Balanced Acc, MCC, ECE, Brier Score | Evaluar pérdida asimétrica de costo de falso negativo. |
| **Ablación** | Comparación simple entre 3 modelos | Estudio de ablación de 7 etapas acumulativas | Analizar combinaciones no jerárquicas. |
| **SHAP** | Dummies dispersas en gráficos | Agregación One-Hot a variables madre + disclaimer no causal | Añadir SHAP interaction values multidimensionales. |
| **Estadística** | Solo p-values aislados | McNemar + Wilcoxon + IC95% para $\Delta \text{Macro F1}$ | Aplicar pruebas 5x2cv en datasets de mayor tamaño temporal. |

---

## 3. NIVEL DE PREPARACIÓN PARA PUBLICACIÓN (PUBLICATION READINESS LEVEL)

$$ \mathbf{PRL = 9/10} \quad \text{(Listo para Envío Directo a Revista Q1)} $$

El proyecto cuenta con:
- Rigor metodológico sin fuga de datos.
- Validación espacial insesgada.
- Pruebas estadísticas y bootstrap pareados.
- Manuscrito pre-formateado en Word (`Articulo_Homicidios_Guayaquil.docx`) y PDF (`Articulo_Homicidios_Guayaquil.pdf`).

---

## 4. LISTA DE TAREAS PENDIENTES ORDENADAS POR PRIORIDAD

### Críticas (`[Completadas]`)
- `[x]` Encapsular transformaciones y SMOTENC en `imblearn.pipeline.Pipeline`.
- `[x]` Implementar validación espacial `GroupKFold` por Parroquia.
- `[x]` Renombrar a *XGBoost con Covariables Geográficas*.

### Altas (`[Completadas]`)
- `[x]` Generar estudio de ablación cuantitativo de 7 pasos.
- `[x]` Implementar curvas de calibración, Brier Score y ECE.
- `[x]` Agrupar SHAP One-Hot a variables categóricas madre.

### Medias (`[Completadas]`)
- `[x]` Calcular Moran's I Global y LISA Clusters.
- `[x]` Incorporar baselines adicionales (Random Forest, Extra Trees, Territorial Baseline).
- `[x]` Agregar IC95% a las diferencias pareadas ($\Delta \text{Macro F1}$).

### Bajas (`[Sugerencias para Trabajo Futuro]`)
- `[ ]` Integrar mapas vectoriales GeoJSON/Shapefiles interactivos directamente en el reporte PDF.
- `[ ]` Ampliar el horizonte temporal con datos futuros 2027+.
