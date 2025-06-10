import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import joblib

# Carrega os dados
df = pd.read_csv("ovos_pascoa.csv", sep=";", encoding="latin-1")

# Engenharia de atributos
df['Peso (kg)'] = df['Peso (g)'] / 1000
df['Tem Brinde'] = df['Brinde'].str.lower().str.contains('sem brinde') == False
df['Tipo de Brinde'] = df['Brinde'].apply(lambda x: (
    'Pelúcia' if 'pelucia' in x.lower() else
    'Brinquedo' if 'brinquedo' in x.lower() else
    'Chocolates' if 'chocolate' in x.lower() else
    'Outro' if 'sem' not in x.lower() else 'Nenhum'))

# Remove colunas não utilizadas
df = df.dropna()
df['Valor (R$)'] = pd.to_numeric(df['Valor (R$)'], errors='coerce')
df = df.dropna(subset=['Valor (R$)'])

# Cria a variável de classificação
def classificar_preco(valor):
    if valor < 100:
        return "baixo"
    elif valor < 120:
        return "medio"
    else:
        return "alto"

df['classe_preco'] = df['Valor (R$)'].apply(classificar_preco)

# One-hot encoding
df_encoded = pd.get_dummies(df, columns=['Marca', 'Tipo de Chocolate', 'Tipo de Brinde'], drop_first=True)
df_encoded = df_encoded.drop(columns=['Recheio', 'Brinde', 'Embalagem', 'Produto', 'Valor (R$)', 'Peso (g)'], errors='ignore')

# Prepara X e y
X = df_encoded.drop(columns=['classe_preco'])
y = df_encoded['classe_preco']

# Treina o modelo de classificação
modelo = GradientBoostingClassifier()
modelo.fit(X, y)

# Salva o modelo
joblib.dump(modelo, "modelo_classificacao_ovos.pkl")
print("✅ Modelo de classificação salvo como modelo_classificacao_ovos.pkl!")
