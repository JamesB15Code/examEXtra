from flask import Flask, request, render_template
import joblib
import numpy as np

# Definir etiquetas descriptivas para cada cluster
cluster_labels = {
    0: 'JovenEsPobre',
    1: 'AdultoEsRico',
    2: 'JovenEsRico',
    3: 'AdultoEsPobre',
    4: 'MayorEsRico'
}

# Crear la aplicación Flask
app = Flask(__name__)

# Cargar el modelo y el escalador
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Obtener datos del formulario
        age = float(request.form['age'])
        annual_income = float(request.form['annual_income'])
        spending_score = float(request.form['spending_score'])
        
        # Normalizar los datos de entrada
        data = scaler.transform([[age, annual_income, spending_score]])
        
        # Predecir la categoría
        prediction = model.predict(data)
        cluster_label = cluster_labels[prediction[0]]
        
        return render_template('index.html', prediction=cluster_label)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
