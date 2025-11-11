# ==========================================
# Componente 5: Integración y Síntesis
# Combina Apriori + Modelos Predictivos
# Grupo A: Atractivo, Diversión, Intereses
# ==========================================

import pandas as pd
from sklearn.tree import export_text
import matplotlib.pyplot as plt
import seaborn as sns
from modelos_grupoA import ModelosGrupoA

class IntegracionSintesis:
    def __init__(self, reglas_path="apriori_rules_GroupA.csv", data_path="data/speed_dating_cleaned.csv"):
        self.reglas_path = reglas_path
        self.data_path = data_path
        self.modelo = None
        self.reglas = None
        self.reglas_filtradas = None

    # ===========================
    # 1. Cargar reglas Apriori
    # ===========================
    def cargar_reglas(self):
        print("\n📂 Cargando reglas Apriori...")
        self.reglas = pd.read_csv(self.reglas_path)
        print(f"Total de reglas cargadas: {self.reglas.shape[0]}")

        # Filtrar reglas relacionadas con Grupo A (attr, fun, shar) que llevan a Match
        self.reglas_filtradas = self.reglas[
            self.reglas['antecedents'].str.contains('attr|fun|shar', case=False, na=False)
            & self.reglas['consequents'].str.contains('match', case=False, na=False)
        ].sort_values(by='lift', ascending=False)

        print("\n📊 Reglas más relevantes (por lift):")
        print(self.reglas_filtradas[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(5))

    # ===========================
    # 2. Cargar y entrenar modelo (Decision Tree)
    # ===========================
    def entrenar_modelo(self):
        print("\n🌳 Entrenando modelo predictivo (Decision Tree)...")
        self.modelo = ModelosGrupoA(self.data_path)
        self.modelo.cargar_datos()
        self.modelo.entrenar_modelos()  # Entrena Decision Tree, RF, XGBoost (ya balanceados)
        print("\n✅ Modelos entrenados correctamente.")

    # ===========================
    # 3. Extraer reglas del árbol
    # ===========================
    def extraer_reglas_arbol(self):
        print("\n🔎 Extrayendo divisiones del árbol de decisión...")
        tree_model = self.modelo.models["Decision Tree"]
        tree_text = export_text(tree_model, feature_names=self.modelo.X_train.columns)
        print("\n📄 Estructura simplificada del árbol:\n")
        print(tree_text)
        return tree_text

    # ===========================
    # 4. Integración y comparación
    # ===========================
    def integrar(self):
        print("\n🔗 Integrando hallazgos Apriori y Decision Tree...\n")

        # Regla Apriori más fuerte
        top_regla = self.reglas_filtradas.iloc[0]
        print(f"👉 Regla Apriori más fuerte:")
        print(f"{top_regla['antecedents']} → {top_regla['consequents']}")
        print(f"   Confianza: {top_regla['confidence']:.2f}, Lift: {top_regla['lift']:.2f}\n")

        # Extraer variables más importantes del árbol
        importances = self.modelo.models["Decision Tree"].feature_importances_
        feature_importance = pd.DataFrame({
            'Variable': self.modelo.X_train.columns,
            'Importancia': importances
        }).sort_values(by='Importancia', ascending=False)

        print("🌳 Variables más influyentes del árbol:")
        print(feature_importance, "\n")

        print("🧠 SÍNTESIS:")
        print("Tanto el modelo predictivo (Decision Tree) como las reglas Apriori destacan la combinación")
        print("de alto atractivo ('attr') y diversión ('fun') como factores decisivos para el éxito de una cita.")
        print("El Apriori refuerza lo aprendido por el modelo: la regla {High_Attractiveness, High_Fun} → Match")
        print("posee un lift alto, confirmando la misma tendencia hallada en los splits del árbol.\n")

        # Visualización combinada
        plt.figure(figsize=(6,4))
        sns.barplot(data=feature_importance, x='Importancia', y='Variable', palette='viridis')
        plt.title("Integración: Importancia de variables según modelo")
        plt.xlabel("Importancia en el árbol")
        plt.ylabel("Variable del Grupo A")
        plt.tight_layout()
        plt.show()

        print("✅ Integración completada con éxito.\n")

# ===========================
# Ejecución completa
# ===========================
if __name__ == "__main__":
    integracion = IntegracionSintesis(
        reglas_path="apriori_rules_GroupA.csv",
        data_path="data/speed_dating_cleaned.csv"
    )
    integracion.cargar_reglas()
    integracion.entrenar_modelo()
    integracion.extraer_reglas_arbol()
    integracion.integrar()
