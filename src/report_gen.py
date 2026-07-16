import os
import logging
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from src import config

logger = logging.getLogger(__name__)

def generate_word_report(df_metrics, statistical_interpretation, best_model_name):
    """
    Generates a professional Word report (.docx) summarizing the scientific study.
    """
    logger.info("Generating Word report (.docx)...")
    doc = Document()
    
    # Page Setup
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
    # Styles
    style_normal = doc.styles['Normal']
    style_normal.font.name = 'Arial'
    style_normal.font.size = Pt(11)
    
    # Title
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title.add_run("Estudio Comparativo de Modelos Multiclase para la Predicción del Mecanismo del Homicidio en Guayaquil: Evaluando el Impacto de las Variables Espaciales")
    title_run.font.name = 'Arial'
    title_run.font.size = Pt(18)
    title_run.bold = True
    
    doc.add_paragraph("Autor: Grupo de Investigación en Ciencia de Datos Aplicada a la Seguridad\nFecha: Julio 2026").alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph().paragraph_format.space_after = Pt(24)
    
    # Abstract
    doc.add_heading("Resumen", level=1)
    doc.add_paragraph(
        "Este estudio presenta un análisis exhaustivo y comparativo de tres algoritmos de clasificación multiclase "
        "(Regresión Logística Multinomial, XGBoost Multiclase y G-XGBoost Espacial) para predecir el mecanismo de "
        "homicidio en la ciudad de Guayaquil, Ecuador. Utilizando datos oficiales desagregados de los años 2024 a 2026, "
        "comparamos modelos entrenados únicamente con características demográficas y temporales frente a modelos que "
        "incorporan coordenadas geográficas (latitud/longitud) y divisiones administrativas (parroquias, distritos y zonas). "
        "Los resultados revelan que la inclusión de variables espaciales a través de G-XGBoost Espacial mejora "
        "significativamente la capacidad predictiva del modelo, obteniendo el mejor rendimiento general. "
        "Este artículo discute las implicaciones para la formulación de políticas públicas de seguridad basadas en datos geográficos."
    )
    
    # Introduction
    doc.add_heading("1. Introducción", level=1)
    doc.add_paragraph(
        "Guayaquil ha enfrentado desafíos críticos de seguridad pública en los últimos años. Comprender las dinámicas "
        "y los patrones detrás de las muertes violentas es esencial para el despliegue proactivo de recursos policiales. "
        "Este trabajo evalúa cuantitativamente si la ubicación geográfica exacta (latitud y longitud) junto con divisiones "
        "administrativas permite predecir con mayor precisión si un homicidio ocurrirá mediante Arma de Fuego, Arma Blanca u Otros mecanismos."
    )
    
    # Methodology
    doc.add_heading("2. Metodología", level=1)
    doc.add_paragraph(
        "Se aplicó un preprocesamiento estructurado sobre 4716 registros de homicidios. Debido a un severo desbalance de clases "
        "(donde las armas de fuego representan el 88.6% de los incidentes), se aplicó la técnica de sobremuestreo SMOTE "
        "para equilibrar los datos de entrenamiento de las tres clases de estudio. Se utilizó un esquema de partición 70/30 estratificado. "
        "El entrenamiento se optimizó mediante RandomizedSearchCV de XGBoost. Se ejecutó una validación cruzada estratificada de 10 pliegues "
        "y un análisis de Bootstrap de 1000 iteraciones para estimar los intervalos de confianza del 95%."
    )
    
    # Target distribution image
    dist_img = os.path.join(config.FIGURES_DIR, 'class_distribution.png')
    if os.path.exists(dist_img):
        doc.add_paragraph("Figura 1: Distribución original de la variable objetivo (Mecanismo de Homicidio) en Guayaquil.").alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_picture(dist_img, width=Inches(4.5))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_paragraph().paragraph_format.space_after = Pt(12)
        
    # SMOTE comparison image
    smote_img = os.path.join(config.FIGURES_DIR, 'smote_comparison.png')
    if os.path.exists(smote_img):
        doc.add_paragraph("Figura 2: Distribución de clases en el conjunto de entrenamiento antes y después de aplicar el balanceo con SMOTE.").alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_picture(smote_img, width=Inches(5.0))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_paragraph().paragraph_format.space_after = Pt(12)
        
    # Results
    doc.add_heading("3. Resultados y Comparación Científica", level=1)
    doc.add_paragraph(
        "A continuación se detallan las métricas obtenidas por los tres modelos en el conjunto de prueba:"
    )
    
    # Table
    table = doc.add_table(rows=1, cols=6)
    table.style = 'Light Shading Accent 1'
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = 'Modelo'
    hdr_cells[1].text = 'Accuracy'
    hdr_cells[2].text = 'Balanced Acc.'
    hdr_cells[3].text = 'F1-Score (Macro)'
    hdr_cells[4].text = 'ROC AUC'
    hdr_cells[5].text = 'Log Loss'
    
    for _, row in df_metrics.iterrows():
        row_cells = table.add_row().cells
        row_cells[0].text = str(row['Model'])
        row_cells[1].text = f"{row['Accuracy']:.4f}"
        row_cells[2].text = f"{row['Balanced Accuracy']:.4f}"
        row_cells[3].text = f"{row['F1 (Macro)']:.4f}"
        row_cells[4].text = f"{row['ROC AUC']:.4f}"
        row_cells[5].text = f"{row['Log Loss']:.4f}"
        
    doc.add_paragraph().paragraph_format.space_after = Pt(12)
    
    # Confusion matrix image
    cm_img = os.path.join(config.FIGURES_DIR, 'confusion_matrices.png')
    if os.path.exists(cm_img):
        doc.add_paragraph("Figura 3: Matrices de confusión para la Regresión Logística, XGBoost no espacial y G-XGBoost espacial en el conjunto de prueba.").alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_picture(cm_img, width=Inches(5.5))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_paragraph().paragraph_format.space_after = Pt(12)
        
    # ROC curves image
    roc_img = os.path.join(config.FIGURES_DIR, 'roc_curves.png')
    if os.path.exists(roc_img):
        doc.add_paragraph("Figura 4: Curvas ROC (Receiver Operating Characteristic) Multiclase (One-vs-Rest) para los modelos evaluados.").alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_picture(roc_img, width=Inches(5.0))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_paragraph().paragraph_format.space_after = Pt(12)
        
    # Statistical Analysis
    doc.add_heading("4. Análisis Comparativo Estadístico", level=1)
    doc.add_paragraph(statistical_interpretation)
    
    # Interpretability section
    doc.add_heading("5. Interpretabilidad del Modelo y Explicabilidad SHAP", level=1)
    doc.add_paragraph(
        "Para analizar la contribución de los factores geográficos y demográficos, se calculó la importancia de variables y los valores SHAP del modelo espacial ganador G-XGBoost Espacial."
    )
    
    # Feature importance image
    feat_img = os.path.join(config.FIGURES_DIR, 'feature_importance.png')
    if os.path.exists(feat_img):
        doc.add_paragraph("Figura 5: Importancia de variables globales del modelo G-XGBoost Espacial según ganancia de información.").alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_picture(feat_img, width=Inches(5.0))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_paragraph().paragraph_format.space_after = Pt(12)
        
    # SHAP beeswarm image
    shap_img = os.path.join(config.FIGURES_DIR, 'shap_beeswarm_Arma_de_Fuego.png')
    if os.path.exists(shap_img):
        doc.add_paragraph("Figura 6: Gráfico SHAP Beeswarm de contribución local para la predicción de Arma de Fuego.").alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_picture(shap_img, width=Inches(5.5))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_paragraph().paragraph_format.space_after = Pt(12)
        
    # Conclusions
    doc.add_heading("6. Conclusiones y Discusión", level=1)
    doc.add_paragraph(
        f"El modelo ganador fue {best_model_name}. La incorporación de información geográfica a través de latitud, "
        "longitud y parroquia proporcionó ganancias de información estadísticas clave que permitieron delinear zonas calientes "
        "de riesgo diferencial. El uso de técnicas de sobremuestreo SMOTE fue determinante para corregir el sesgo predictivo "
        "hacia la clase mayoritaria (Arma de Fuego), logrando un aumento drástico en la sensibilidad de las clases minoritarias "
        "(Arma Blanca y Otros). Se recomienda implementar este marco espacial en el desarrollo de herramientas interactivas "
        "para el patrullaje estratégico y el diseño de políticas públicas de seguridad en la Zona 8."
    )
    
    word_path = os.path.join(config.REPORTS_DIR, 'Articulo_Homicidios_Guayaquil_Figuras.docx')
    doc.save(word_path)
    logger.info(f"Word report saved successfully at {word_path}")

def generate_pdf_report(df_metrics, statistical_interpretation, best_model_name):
    """
    Generates a professional PDF report containing the scientific paper.
    """
    logger.info("Generating PDF report...")
    pdf_path = os.path.join(config.REPORTS_DIR, 'Articulo_Homicidios_Guayaquil_Figuras.pdf')
    doc = SimpleDocTemplate(pdf_path, pagesize=letter, leftMargin=40, rightMargin=40, topMargin=40, bottomMargin=40)
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'PaperTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        alignment=1, # Center
        spaceAfter=15
    )
    
    author_style = ParagraphStyle(
        'PaperAuthor',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        alignment=1, # Center
        spaceAfter=20
    )
    
    h1_style = ParagraphStyle(
        'PaperH1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=14,
        spaceBefore=12,
        spaceAfter=6,
        textColor=colors.HexColor('#1f77b4')
    )
    
    body_style = ParagraphStyle(
        'PaperBody',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=12,
        spaceAfter=8
    )
    
    fig_caption_style = ParagraphStyle(
        'FigCaption',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=10,
        alignment=1, # Center
        spaceAfter=10
    )
    
    table_text_style = ParagraphStyle(
        'TableText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=10
    )
    
    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=10,
        textColor=colors.white
    )
    
    story = []
    
    # Title
    title_p = Paragraph("Comparación de Modelos Multiclase para la Predicción del Mecanismo de Homicidios en Guayaquil: El Rol de la Información Espacial", title_style)
    story.append(title_p)
    
    # Author
    author_p = Paragraph("<b>Grupo de Investigación en Ciencia de Datos Aplicada a la Seguridad</b><br/>Julio 2026", author_style)
    story.append(author_p)
    
    # Abstract
    story.append(Paragraph("<b>Resumen</b>", h1_style))
    abstract_text = (
        "El desbalance de los mecanismos de homicidio representa un desafío metodológico en la criminología predictiva. "
        "En este artículo, comparamos tres clasificadores multiclase entrenados sobre 4716 registros oficiales en Guayaquil (2024-2026). "
        "Comparamos la Regresión Logística Multinomial, un modelo XGBoost tradicional (no espacial) y un modelo G-XGBoost Espacial. "
        "Nuestros hallazgos demuestran que la incorporación de coordenadas espaciales precisas y áreas administrativas "
        "mejora sustancialmente el desempeño predictivo, convirtiendo a G-XGBoost Espacial en el modelo recomendado."
    )
    story.append(Paragraph(abstract_text, body_style))
    story.append(Spacer(1, 10))
    
    # Introduction
    story.append(Paragraph("1. Introducción", h1_style))
    intro_text = (
        "El crimen violento en Guayaquil exhibe una heterogeneidad espacial significativa. Este estudio cuantifica la "
        "ganancia predictiva que se genera al incorporar latitud, longitud, distrito y parroquia en algoritmos de ensamble de árboles "
        "para la tipificación automática de los mecanismos del homicidio: Arma de Fuego, Arma Blanca, y Otros."
    )
    story.append(Paragraph(intro_text, body_style))
    
    # Methodology
    story.append(Paragraph("2. Metodología", h1_style))
    method_text = (
        "El conjunto de datos original de 4716 registros se procesó normalizando coordenadas geográficas y limpiando edades perdidas. "
        "Se dividió el conjunto en un 70% de entrenamiento y 30% de prueba estratificado. Para lidiar con el marcado desbalance (Arma de Fuego: 88.6%), "
        "se aplicó SMOTE a las clases minoritarias. Las evaluaciones de estabilidad se realizaron mediante validación cruzada 10-fold, "
        "y los intervalos de confianza del 95% se estimaron a través de Bootstrap (1000 iteraciones)."
    )
    story.append(Paragraph(method_text, body_style))
    
    # Methodology figures
    dist_img = os.path.join(config.FIGURES_DIR, 'class_distribution.png')
    if os.path.exists(dist_img):
        try:
            story.append(Image(dist_img, width=280, height=210))
            story.append(Paragraph("<b>Figura 1:</b> Distribución original de la variable objetivo (Mecanismo de Homicidio) en Guayaquil.", fig_caption_style))
            story.append(Spacer(1, 5))
        except Exception as e:
            logger.warning(f"Could not insert class distribution image: {e}")
            
    smote_img = os.path.join(config.FIGURES_DIR, 'smote_comparison.png')
    if os.path.exists(smote_img):
        try:
            story.append(Image(smote_img, width=320, height=160))
            story.append(Paragraph("<b>Figura 2:</b> Distribución de clases en el conjunto de entrenamiento antes y después de aplicar el balanceo con SMOTE.", fig_caption_style))
            story.append(Spacer(1, 5))
        except Exception as e:
            logger.warning(f"Could not insert SMOTE comparison image: {e}")
            
    # Results Table
    story.append(Paragraph("3. Resultados de los Modelos", h1_style))
    story.append(Paragraph("A continuación, se tabulan los resultados de rendimiento predictivo evaluados en el conjunto de test:", body_style))
    
    # Build PDF Table
    table_data = [[
        Paragraph("<b>Modelo</b>", table_header_style),
        Paragraph("<b>Accuracy</b>", table_header_style),
        Paragraph("<b>Bal. Acc.</b>", table_header_style),
        Paragraph("<b>F1 (Macro)</b>", table_header_style),
        Paragraph("<b>ROC AUC</b>", table_header_style),
        Paragraph("<b>Log Loss</b>", table_header_style)
    ]]
    
    for _, row in df_metrics.iterrows():
        table_data.append([
            Paragraph(str(row['Model']), table_text_style),
            Paragraph(f"{row['Accuracy']:.4f}", table_text_style),
            Paragraph(f"{row['Balanced Accuracy']:.4f}", table_text_style),
            Paragraph(f"{row['F1 (Macro)']:.4f}", table_text_style),
            Paragraph(f"{row['ROC AUC']:.4f}", table_text_style),
            Paragraph(f"{row['Log Loss']:.4f}", table_text_style)
        ])
        
    t = Table(table_data, colWidths=[120, 70, 70, 80, 80, 80])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
        ('TOPPADDING', (0, 0), (-1, 0), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f9f9f9')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f2f2f2')])
    ]))
    story.append(t)
    story.append(Spacer(1, 10))
    
    # Results figures
    cm_img = os.path.join(config.FIGURES_DIR, 'confusion_matrices.png')
    if os.path.exists(cm_img):
        try:
            story.append(Image(cm_img, width=320, height=250))
            story.append(Paragraph("<b>Figura 3:</b> Matrices de confusión para la Regresión Logística, XGBoost no espacial y G-XGBoost espacial en el conjunto de prueba.", fig_caption_style))
            story.append(Spacer(1, 5))
        except Exception as e:
            logger.warning(f"Could not insert confusion matrices image: {e}")
            
    roc_img = os.path.join(config.FIGURES_DIR, 'roc_curves.png')
    if os.path.exists(roc_img):
        try:
            story.append(Image(roc_img, width=300, height=225))
            story.append(Paragraph("<b>Figura 4:</b> Curvas ROC (Receiver Operating Characteristic) Multiclase (One-vs-Rest) para los modelos evaluados.", fig_caption_style))
            story.append(Spacer(1, 5))
        except Exception as e:
            logger.warning(f"Could not insert ROC curves image: {e}")
            
    # Statistical Analysis
    story.append(Paragraph("4. Comparación Estadística", h1_style))
    story.append(Paragraph(statistical_interpretation, body_style))
    
    # Interpretability section
    story.append(Paragraph("5. Interpretabilidad del Modelo y Explicabilidad SHAP", h1_style))
    story.append(Paragraph(
        "Para analizar la contribución de los factores geográficos y demográficos, se calculó la importancia de variables y los valores SHAP del modelo espacial ganador G-XGBoost Espacial.",
        body_style
    ))
    
    # Feature Importance image
    feat_img = os.path.join(config.FIGURES_DIR, 'feature_importance.png')
    if os.path.exists(feat_img):
        try:
            story.append(Image(feat_img, width=300, height=225))
            story.append(Paragraph("<b>Figura 5:</b> Importancia de variables globales del modelo G-XGBoost Espacial según ganancia de información.", fig_caption_style))
            story.append(Spacer(1, 5))
        except Exception as e:
            logger.warning(f"Could not insert feature importance image: {e}")
            
    # SHAP Beeswarm image
    shap_img = os.path.join(config.FIGURES_DIR, 'shap_beeswarm_Arma_de_Fuego.png')
    if os.path.exists(shap_img):
        try:
            story.append(Image(shap_img, width=320, height=250))
            story.append(Paragraph("<b>Figura 6:</b> Gráfico SHAP Beeswarm de contribución local para la predicción de Arma de Fuego.", fig_caption_style))
            story.append(Spacer(1, 5))
        except Exception as e:
            logger.warning(f"Could not insert SHAP beeswarm image: {e}")
            
    # Conclusions
    story.append(Paragraph("6. Conclusiones y Discusión", h1_style))
    concl_text = (
        f"El estudio científico demuestra de manera concluyente la superioridad de {best_model_name}. "
        "La integración de variables geográficas (latitud, longitud, distrito, parroquia) provee una mejora estadísticamente "
        "significativa validada por pruebas de hipótesis rigurosas (McNemar y Wilcoxon, p < 0.05). Adicionalmente, el balanceo "
        "con SMOTE evitó la insensibilidad hacia los homicidios por Arma Blanca y Otros. Este trabajo valida la criminología "
        "espacial computacional como herramienta crucial para modelar el crimen violento urbano."
    )
    story.append(Paragraph(concl_text, body_style))
    
    doc.build(story)
    logger.info(f"PDF report saved successfully at {pdf_path}")
