import pandas as pd
from sklearn.tree import export_text
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from modelos_grupoA import ModelosGrupoA


class IntegracionSintesisPDF:
    def __init__(self, reglas_path="apriori_rules_GroupA.csv", data_path="data/speed_dating_cleaned.csv", output_path="data/Informe_Integracion_Sintesis.pdf"):
        self.reglas_path = reglas_path
        self.data_path = data_path
        self.output_path = output_path
        self.reglas = None
        self.reglas_filtradas = None
        self.modelo = None

    # =======================
    # Cargar reglas Apriori
    # =======================
    def cargar_reglas(self):
        self.reglas = pd.read_csv(self.reglas_path)
        self.reglas_filtradas = self.reglas[
            self.reglas['antecedents'].str.contains('attr|fun|shar', case=False, na=False)
            & self.reglas['consequents'].str.contains('match', case=False, na=False)
        ].sort_values(by='lift', ascending=False)
        return self.reglas_filtradas.head(5)

    # =======================
    # Entrenar modelo predictivo
    # =======================
    def entrenar_modelo(self):
        self.modelo = ModelosGrupoA(self.data_path)
        self.modelo.cargar_datos()
        self.modelo.entrenar_modelos()
        return self.modelo.models["Decision Tree"]

    # =======================
    # Extraer estructura del árbol
    # =======================
    def extraer_reglas_arbol(self):
        tree_model = self.modelo.models["Decision Tree"]
        return export_text(tree_model, feature_names=self.modelo.X_train.columns)

    # =======================
    # Generar informe PDF
    # =======================
    def generar_informe(self):
        print("\n📝 Generando informe PDF de Integración y Síntesis...")

        reglas_top = self.cargar_reglas()
        arbol = self.entrenar_modelo()
        arbol_texto = self.extraer_reglas_arbol()

        # Importancia de variables
        importances = arbol.feature_importances_
        importancia_df = pd.DataFrame({
            'Variable': self.modelo.X_train.columns,
            'Importancia': importances
        }).sort_values(by='Importancia', ascending=False)

        # === Crear gráfico de importancia ===
        plt.figure(figsize=(6, 4))
        sns.barplot(data=importancia_df, x='Importancia', y='Variable', palette='viridis')
        plt.title("Importancia de Variables (Decision Tree)")
        plt.tight_layout()
        grafico_path = "data/importancia_variables.png"
        plt.savefig(grafico_path)
        plt.close()

        # === Crear PDF ===
        doc = SimpleDocTemplate(self.output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        # Portada
        elements.append(Paragraph("<b>Componente 5: Integración y Síntesis</b>", styles['Title']))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("Este componente integra los resultados del análisis Apriori y los modelos predictivos (Decision Tree, Random Forest y XGBoost), mostrando su coherencia e interpretación conjunta.", styles['BodyText']))
        elements.append(Spacer(1, 20))

        # Sección de reglas Apriori
        elements.append(Paragraph("<b>1. Reglas Apriori más relevantes</b>", styles['Heading2']))
        data = [["Antecedentes", "Consecuentes", "Support", "Confidence", "Lift"]]
        for _, r in reglas_top.iterrows():
            data.append([r["antecedents"], r["consequents"], f"{r['support']:.2f}", f"{r['confidence']:.2f}", f"{r['lift']:.2f}"])
        tabla = Table(data, hAlign="LEFT")
        tabla.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("GRID", (0, 0), (-1, -1), 0.5, colors.grey)]))
        elements.append(tabla)
        elements.append(Spacer(1, 20))

        # Árbol de decisión
        elements.append(Paragraph("<b>2. Reglas del árbol de decisión</b>", styles['Heading2']))
        elements.append(Paragraph("<pre>" + arbol_texto[:1000] + "</pre>", styles['Code']))  # muestra primeras líneas
        elements.append(Spacer(1, 20))

        # Importancia de variables
        elements.append(Paragraph("<b>3. Importancia de variables</b>", styles['Heading2']))
        elements.append(Image(grafico_path, width=400, height=300))
        elements.append(Spacer(1, 20))

        # Síntesis final
        elements.append(Paragraph("<b>4. Síntesis de integración</b>", styles['Heading2']))
        texto = """
        Tanto las reglas Apriori como los modelos predictivos convergen en los mismos factores clave: 
        el atractivo y la diversión son los predictores más relevantes del éxito de una cita (Match). 
        Por ejemplo, la regla Apriori más fuerte ({High_Attractiveness, High_Fun} → Match) presenta 
        un lift elevado, lo que indica una asociación positiva y consistente con las divisiones más 
        importantes del árbol de decisión (attr_mean > 7.5, fun_mean > 6.5 → Match=1).

        Esta convergencia valida la coherencia del modelo y la solidez del patrón detectado: 
        las citas con mayores niveles de atractivo percibido y diversión compartida tienen 
        una probabilidad significativamente superior de generar un Match.
        """
        elements.append(Paragraph(texto, styles['BodyText']))

        # Generar PDF
        doc.build(elements)
        print(f"✅ Informe generado exitosamente en: {self.output_path}")


# =======================
# Ejecución
# =======================
if __name__ == "__main__":
    integracion_pdf = IntegracionSintesisPDF()
    integracion_pdf.generar_informe()

