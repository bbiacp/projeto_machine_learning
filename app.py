
from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
modelo = joblib.load("modelo_classificacao_ovos.pkl")

FEATURES = [
    'Peso (kg)', 'Tem Brinde',
    'Marca_Cacau Show', 'Marca_Ferrero Rocher', 'Marca_Garoto',
    'Marca_Kopenhagen', 'Marca_Lacta', 'Marca_Nestle',
    'Tipo de Chocolate_Belga', 'Tipo de Chocolate_Branco',
    'Tipo de Chocolate_Meio amargo', 'Tipo de Chocolate_Ruby',
    'Tipo de Chocolate_Vegano',
    'Tipo de Brinde_Chocolates', 'Tipo de Brinde_Nenhum', 'Tipo de Brinde_Pelúcia'
]

@app.route("/")
def index():
    return render_template("index.html", resultado=None)

@app.route("/prever", methods=["POST"])
def prever():
    form = request.form
    entrada = {feature: 0 for feature in FEATURES}
    entrada['Peso (kg)'] = float(form.get("peso"))
    entrada['Tem Brinde'] = int(form.get("tem_brinde"))

    marca = f"Marca_{form.get('marca')}"
    if marca in entrada:
        entrada[marca] = 1

    chocolate = f"Tipo de Chocolate_{form.get('tipo_chocolate')}"
    if chocolate in entrada:
        entrada[chocolate] = 1

    brinde = f"Tipo de Brinde_{form.get('tipo_brinde')}"
    if brinde in entrada:
        entrada[brinde] = 1

    X = np.array([list(entrada.values())])
    predicao = modelo.predict(X)[0]

    return render_template("index.html", resultado=f"Faixa de preço prevista: {predicao.upper()}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
