import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules

# ===========================
# 1. Cargar dataset limpio
# ===========================
df = pd.read_csv("data/speed_dating_cleaned.csv")

# ===========================
# 2. Seleccionar columnas relevantes (Grupo A)
# ===========================
cols = ['match', 'attr_o', 'fun_o', 'int_corr']
data = df[cols].dropna()

# ===========================
# 3. Crear columnas binarias
# ===========================
data['High_Attractive'] = (data['attr_o'] >= 7).astype(int)
data['High_Fun'] = (data['fun_o'] >= 7).astype(int)
data['High_SharedInterests'] = (data['int_corr'] >= 0.6).astype(int)
data['Match'] = (data['match'] == 1).astype(int)

basket = data[['High_Attractive', 'High_Fun', 'High_SharedInterests', 'Match']]

# ===========================
# 4. Aplicar Apriori
# ===========================
frequent_itemsets = apriori(basket, min_support=0.1, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Filtrar reglas con 'Match'
rules = rules[rules['consequents'].apply(lambda x: 'Match' in x)]
rules = rules.sort_values(by='lift', ascending=False)

# ===========================
# 5. Mostrar resultados
# ===========================
print("\n📊 Reglas de asociación más relevantes:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())

# Guardar CSV
rules.to_csv("apriori_rules_GroupA.csv", index=False)
print("\n💾 Archivo 'apriori_rules_GroupA.csv' guardado correctamente.")

# ===========================
# 6. Visualizaciones
# ===========================

# --- a) Gráfico de dispersión soporte vs confianza ---
plt.figure(figsize=(8,6))
sns.scatterplot(x='support', y='confidence', size='lift', hue='lift', 
                data=rules, sizes=(50, 300), palette='viridis', alpha=0.7)
plt.title("Reglas de Asociación - Soporte vs Confianza (Grupo A)")
plt.xlabel("Soporte")
plt.ylabel("Confianza")
plt.legend(title='Lift', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# --- b) Top reglas por Lift ---
top_rules = rules.nlargest(5, 'lift')
plt.figure(figsize=(8,5))
sns.barplot(x='lift', y=top_rules['antecedents'].astype(str), data=top_rules, palette='coolwarm')
plt.title("Top 5 Reglas por Lift (Grupo A)")
plt.xlabel("Lift")
plt.ylabel("Antecedentes")
plt.tight_layout()
plt.show()

# --- c) Heatmap de correlación entre atributos binarios ---
plt.figure(figsize=(6,5))
sns.heatmap(basket.corr(), annot=True, cmap='Blues', fmt=".2f")
plt.title("Correlación entre atributos (Grupo A)")
plt.tight_layout()
plt.show()

# ===========================
# 7. Información del grupo
# ===========================
print("\n===============================")
print("💡 GROUP SPECIALIZATION SUMMARY")
print("===============================")
print("Group: A")
print("Focus: Attractiveness + Fun + Shared Interests")
print("Specialization: Tree interpretability and visualization")
print("===============================\n")
