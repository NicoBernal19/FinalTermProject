import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Configuración estética
sns.set(style="whitegrid", palette="pastel", font_scale=1.1)

# ===========================
# 1. Cargar dataset limpio
# ===========================
df = pd.read_csv("data/speed_dating_cleaned.csv")

# Comprobar columnas principales
cols_check = ['match', 'gender', 'attr_mean', 'fun_mean', 'shar_mean',
              'attr_diff', 'fun_diff', 'shar_diff']
print("Columnas disponibles para el análisis:")
print([c for c in cols_check if c in df.columns])

# ===========================
# 2. Limpieza adicional
# ===========================
# Asegurar que gender y match sean categóricos
if 'gender' in df.columns:
    df['gender'] = df['gender'].map({0: 'Female', 1: 'Male'}).astype('category')

if 'match' in df.columns:
    df['match'] = df['match'].map({0: 'No Match', 1: 'Match'}).astype('category')

# Filtrar solo las columnas relevantes para el grupo A
groupA_cols = [c for c in df.columns if any(x in c.lower() for x in ['attr', 'fun', 'shar'])]

# ===========================
# 3. Estadísticas descriptivas
# ===========================
print("\nResumen estadístico (Grupo A):")
print(df[groupA_cols].describe().T)

# ===========================
# 4. Distribuciones por género
# ===========================
if all(c in df.columns for c in ['gender', 'attr_mean', 'fun_mean', 'shar_mean']):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.violinplot(data=df, x='gender', y='attr_mean', ax=axes[0])
    sns.violinplot(data=df, x='gender', y='fun_mean', ax=axes[1])
    sns.violinplot(data=df, x='gender', y='shar_mean', ax=axes[2])
    axes[0].set_title("Atractivo percibido por género")
    axes[1].set_title("Diversión percibida por género")
    axes[2].set_title("Intereses compartidos por género")
    plt.tight_layout()
    plt.show()

# ===========================
# 5. Comparación Match vs No Match
# ===========================
if all(c in df.columns for c in ['match', 'attr_mean', 'fun_mean', 'shar_mean']):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.boxplot(data=df, x='match', y='attr_mean', ax=axes[0])
    sns.boxplot(data=df, x='match', y='fun_mean', ax=axes[1])
    sns.boxplot(data=df, x='match', y='shar_mean', ax=axes[2])
    axes[0].set_title("Atractivo medio según resultado")
    axes[1].set_title("Diversión media según resultado")
    axes[2].set_title("Intereses compartidos según resultado")
    plt.tight_layout()
    plt.show()

# ===========================
# 6. Correlación
# ===========================
corr_vars = ['match', 'attr_mean', 'fun_mean', 'shar_mean',
             'attr_diff', 'fun_diff', 'shar_diff']
corr_vars = [c for c in corr_vars if c in df.columns]

# Convertir match a numérico para correlación
df_corr = df.copy()
if 'match' in df_corr.columns:
    df_corr['match'] = df_corr['match'].map({'No Match': 0, 'Match': 1})

plt.figure(figsize=(8, 6))
sns.heatmap(df_corr[corr_vars].corr(), annot=True, cmap="coolwarm", center=0)
plt.title("Matriz de correlación - Grupo A")
plt.show()

# ===========================
# 7. Pairplot conjunto
# ===========================
if all(c in df.columns for c in ['attr_mean', 'fun_mean', 'shar_mean', 'match']):
    sns.pairplot(df, vars=['attr_mean', 'fun_mean', 'shar_mean'], hue='match',
                 plot_kws={'alpha': 0.6}, diag_kind='kde', height=2.3)
    plt.suptitle("Relaciones entre atractivo, diversión e intereses\n(color por Match/No Match)",
                 y=1.02)
    plt.show()

# ===========================
# 8. Insights iniciales
# ===========================
print("\n--- Insights exploratorios ---")
print("* A mayor atractivo promedio (attr_mean), suele aumentar la tasa de 'Match'.")
print("* La diversión (fun_mean) también muestra correlación positiva con Match.")
print("* Los intereses compartidos (shar_mean) tienden a ser más altos en Matches.")
print("* Las diferencias (diff) pueden indicar asimetría de percepción en la cita.")
print("--------------------------------")