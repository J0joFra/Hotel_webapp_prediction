import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.express as px  # Import Plotly for interactive plots
import matplotlib.pyplot as plt
import seaborn as sns
import io
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.pdfgen import canvas
import os
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb+srv://jofrancalanci:Cf8m2xsQdZgll1hz@element.2o7dxct.mongodb.net/")
db = client['hotel_db']
collection = db['hotel_data'] 

# Fetch data from MongoDB and convert it to a DataFrame
data = pd.DataFrame(list(collection.find()))

# Preprocess the data
data.fillna(data.mean(), inplace=True)

# One-hot encode categorical columns
data = pd.get_dummies(data, drop_first=True)

# Select relevant columns
selected_columns = [
    'hotel', 'meal', 'arrival_date_month', 'is_canceled', 'season',
    'adults', 'children', 'babies', 'total_guests', 'adr', 'lead_time'
]

# Prepare the data
X = data[selected_columns]
y = data['adr']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit UI
st.title("🎯 Predizione del Prezzo Medio Giornaliero (ADR)")
st.write("""
Questa applicazione utilizza un modello di **regressione lineare** per predire il prezzo medio giornaliero di una prenotazione (**ADR**)
basandosi su diversi parametri.
""")

# User input
st.sidebar.header("📊 Inserisci i parametri")
adults = st.sidebar.number_input("Numero di Adulti (max 4)", min_value=1, max_value=4, value=2)
children = st.sidebar.number_input("Numero di Bambini (max 5)", min_value=0, max_value=5, value=0)
lead_time = st.sidebar.number_input("Lead Time (giorni prima della prenotazione)", min_value=0, max_value=1000, value=30)
stay_duration = st.sidebar.number_input("Durata del Soggiorno (notti totali)", min_value=1, max_value=20, value=5)
hotel_type = st.sidebar.selectbox("Tipo di Hotel", ["City Hotel", "Resort Hotel"])

# Encode hotel type as numeric
hotel_encoding = {
    "City Hotel": [0],
    "Resort Hotel": [1]
}[hotel_type]

# Prediction
user_input = pd.DataFrame({
    'adults': [adults],
    'children': [children],
    'lead_time': [lead_time],
    'stay_duration': [stay_duration],
    'Resort Hotel': [hotel_encoding[0]]
})

prediction = model.predict(user_input)[0]

# Results
st.subheader("🎉 Risultato della Predizione")
st.markdown(
    f"""
    <div style="background-color:#f0f9ff; padding:10px; border-radius:10px; border: 2px solid #2196f3;">
        <h2 style="color:#2196f3; text-align:center;">Prezzo Medio Giornaliero Predetto (ADR): <b>{prediction:.2f} €</b></h2>
        <p><i>Tipo di Hotel: {hotel_type}</i></p>
    </div>
    """,
    unsafe_allow_html=True
)

# Model metrics
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📈 Metriche del Modello")
st.markdown(
    f"""
    <div style="background-color:#e8f5e9; padding:10px; border-radius:10px; border: 2px solid #43a047;">
        <p><b>Errore Medio Assoluto (MAE):</b> {mae:.2f}</p>
        <p><b>Coefficiente di Determinazione (R²):</b> {r2:.2f}</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Interactive plot with Plotly
st.subheader("📊 Distribuzione delle Predizioni (Grafico Interattivo)")
fig = px.histogram(y_pred, nbins=20, title="Distribuzione delle Predizioni ADR", labels={"value": "ADR Predetto"})
fig.add_vline(x=prediction, line=dict(dash="dash", color="red"), annotation_text="Valore Utente", annotation_position="top right")
st.plotly_chart(fig)

# Function to save plot image
def save_plot_image():
    plot_dir = 'C:\\tmp'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plot_image_path = os.path.join(plot_dir, 'prediction_plot.png')

    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    sns.histplot(y_pred, bins=20, color='royalblue', label='Predizioni ADR', kde=False)
    sns.kdeplot(y_pred, color='orange', linewidth=2, label='Curva di densità KDE delle Predizioni ADR')
    plt.axvline(prediction, color='red', linestyle='--', label=f'Valore Utente ({prediction:.2f} €)', linewidth=2)
    plt.title('Distribuzione delle Predizioni ADR con il Valore Utente', fontsize=16, fontweight='bold', color='darkblue')
    plt.xlabel('ADR Predetto (€)', fontsize=14)
    plt.ylabel('Frequenza', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_image_path)
    plt.close()

    return plot_image_path

# Function to create PDF
def create_pdf(prediction, mae, r2, plot_image):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)

    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, 750, "📊 Report Predizione ADR")

    c.setFont("Helvetica", 12)
    c.drawString(72, 730, f"Prezzo Medio Giornaliero Predetto (ADR): {prediction:.2f} €")
    c.drawString(72, 710, f"Errore Medio Assoluto (MAE): {mae:.2f}")
    c.drawString(72, 690, f"Coefficiente di Determinazione (R²): {r2:.2f}")

    c.drawImage(plot_image, 72, 400, width=450, height=250)

    c.showPage()
    c.save()

    buf.seek(0)
    return buf

# Button to download the PDF
st.subheader("📥 Scarica il Report PDF")
plot_image = save_plot_image()
pdf_buffer = create_pdf(prediction, mae, r2, plot_image)

st.download_button(
    label="Scarica il Report PDF",
    data=pdf_buffer,
    file_name="report_predizione_adr.pdf",
    mime="application/pdf"
)
