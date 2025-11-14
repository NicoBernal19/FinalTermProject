import pandas as pd, numpy as np
from Tools.scripts.dutree import display

csv_path = "data/Speed Dating Data.csv"
df = pd.read_csv(csv_path, encoding='latin1')
df_norm = df.copy()

## Normaliza columnas numéricas dividiendo entre 10 si max y mediana > 10
def normalize_col(col):
    series = pd.to_numeric(df_norm[col], errors='coerce')
    if series.dropna().empty:
        return df_norm[col]
    max_v = series.max(skipna=True)
    med_v = series.median(skipna=True)
    if pd.notna(max_v) and max_v > 10 and pd.notna(med_v) and med_v > 10:
        return series.div(10).clip(lower=1, upper=10)
    else:
        return series.where(series.notna(), df_norm[col])

## Aplica normalización a todas las columnas numéricas
cols_to_try = [c for c in df_norm.columns if df_norm[c].dtype.kind in 'biufc']
for c in cols_to_try:
    try:
        df_norm[c] = normalize_col(c)
    except Exception:
        df_norm[c] = df_norm[c]

derived = pd.DataFrame(index=df_norm.index)

## Crea columnas de diferencia y promedio entre dos variables (ej: attr vs attr_o)
def create_diff_mean(col1, col2, base_name):
    if col1 in df_norm.columns and col2 in df_norm.columns:
        a = pd.to_numeric(df_norm[col1], errors='coerce')
        b = pd.to_numeric(df_norm[col2], errors='coerce')
        derived[f'{base_name}_diff'] = a - b
        derived[f'{base_name}_mean'] = pd.concat([a, b], axis=1).mean(axis=1)
        return True
    return False

## Genera diferencias y promedios para atributos clave
if create_diff_mean('attr', 'attr_o', 'attr'):
    pass
if create_diff_mean('fun', 'fun_o', 'fun'):
    pass
if create_diff_mean('shar', 'shar_o', 'shar'):
    pass

## Fallback: busca columnas attr si no se crearon las diferencias
if not any(c.endswith('_diff') for c in derived.columns):
    attr_cols = [c for c in df_norm.columns if c.lower().startswith('attr') and 'o' not in c.lower()]
    attr_o_cols = [c for c in df_norm.columns if ('attr' in c.lower() and 'o' in c.lower()) or (c.lower().endswith('_o') and 'attr' in c.lower())]
    if attr_cols and attr_o_cols:
        c1 = attr_cols[0]; c2 = attr_o_cols[0]
        a = pd.to_numeric(df_norm[c1], errors='coerce'); b = pd.to_numeric(df_norm[c2], errors='coerce')
        derived['attr_diff'] = a - b
        derived['attr_mean'] = pd.concat([a, b], axis=1).mean(axis=1)
        print("Creada attr_diff usando", c1, "y", c2)

## Crea variable samerace (misma raza entre participantes)
if 'samerace' in df_norm.columns:
    derived['samerace'] = df_norm['samerace']
else:
    if 'race' in df_norm.columns and 'race_o' in df_norm.columns:
        derived['samerace'] = (df_norm['race'] == df_norm['race_o']).astype(int)

## Calcula gaps entre importancia declarada y percibida
for colpair in [('attr1_1','attr3_1','attr'), ('fun1_1','fun3_1','fun'), ('shar1_1','shar3_1','shar')]:
    c1, c2, base = colpair
    if c1 in df_norm.columns and c2 in df_norm.columns:
        derived[f'{base}_importance_perception_gap'] = pd.to_numeric(df_norm[c1], errors='coerce') - pd.to_numeric(df_norm[c2], errors='coerce')

## Combina dataset normalizado con columnas derivadas
df_clean = pd.concat([df_norm, derived], axis=1)

## Elimina duplicados
initial_count = df_clean.shape[0]
df_clean = df_clean.drop_duplicates()
duplicates_removed = initial_count - df_clean.shape[0]

## Imputa valores faltantes con la mediana (o 0 si no hay mediana)
def impute_col(col):
    if pd.api.types.is_numeric_dtype(col):
        med = col.median(skipna=True)
        if pd.isna(med):
            med = 0
        return col.fillna(med)
    else:
        return col

df_clean = df_clean.apply(impute_col, axis=0)

## Guarda dataset limpio
out_path = "data/speed_dating_cleaned.csv"
df_clean.to_csv(out_path, index=False, encoding='utf-8')

## Reportes finales
print("Guardado en:", out_path)
print("Filas iniciales:", initial_count, "Filas finales:", df_clean.shape[0], "Duplicados removidos:", duplicates_removed)
print("\nEjemplo de columnas derivadas:", [c for c in df_clean.columns if c.endswith('_diff') or c.endswith('_mean') or 'perception_gap' in c][:60])

group_a_cols = [c for c in df_clean.columns if any(x in c.lower() for x in ['attr','fun','shar'])]
print("\nColumnas relacionadas con Grupo A (ejemplos):", group_a_cols[:80])
print(df_clean[group_a_cols].describe().T)