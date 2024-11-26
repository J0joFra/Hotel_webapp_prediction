import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import tempfile
import os

# Simuliamo i dati (sostituire con il vostro dataset)
data = pd.read_csv('Hotel.csv')
df = pd.DataFrame(data)

# Prepariamo i dati
X = df[['adults', 'children', 'babies', 'lead_time', 'stay_duration', 'hotel', 
        'season', 'country_name']]
y = df['adr']

# Gestione delle colonne categoriche
X = pd.get_dummies(X, drop_first=True)

# Gestione dei valori mancanti
X_numeric = X.select_dtypes(include=[np.number])
X[X_numeric.columns] = X_numeric.fillna(X_numeric.mean())
y.fillna(y.mean(), inplace=True)

# Dividiamo i dati in set di addestramento e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Addestramento del modello di regressione lineare
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit UI
st.title("🎯 Predizione del Prezzo Medio Giornaliero (ADR)")
st.write(""" 
Questa applicazione utilizza un modello di **regressione lineare** per predire il prezzo medio giornaliero di una prenotazione (**ADR**) 
basandosi su diversi parametri.
""")

# Spazio tra le sezioni
st.markdown("---")

# Input utente
st.sidebar.header("📊 Inserisci i parametri")
adults = st.sidebar.number_input("Numero di Adulti (max 4)", min_value=1, max_value=4, value=2)
children = st.sidebar.number_input("Numero di Bambini (max 5)", min_value=0, max_value=5, value=0)
lead_time = st.sidebar.number_input("Lead Time (giorni prima della prenotazione)", min_value=0, max_value=1000, value=30)
stay_duration = st.sidebar.number_input("Durata del Soggiorno (notti totali)", min_value=1, max_value=20, value=5)
hotel_type = st.sidebar.selectbox(
    "Tipo di Hotel", ["City Hotel", "Resort Hotel"]
)

# Aggiungere il tipo di hotel come variabile numerica
hotel_encoding = {
    "City Hotel": [0],
    "Resort Hotel": [1]
}[hotel_type]

# Selezione del mese
st.sidebar.header("📅 Seleziona il mese")
season_month = st.sidebar.selectbox(
    "Mese di arrivo",
    ["Gennaio", "Febbraio", "Marzo", "Aprile", "Maggio", "Giugno", "Luglio",
     "Agosto", "Settembre", "Ottobre", "Novembre", "Dicembre"]
)

# Predizione
user_input = pd.DataFrame({
    'adults': [adults],
    'children': [children],
    'lead_time': [lead_time],
    'stay_duration': [stay_duration],
    'Resort Hotel': [hotel_encoding[0]]
})

# Aggiungere tutte le colonne mancanti
for col in X.columns:
    if col not in user_input.columns:
        user_input[col] = 0

# Riordinare le colonne
user_input = user_input[X.columns]

# Predire il valore
prediction = model.predict(user_input)[0]

# Risultati
st.subheader("🎉 Risultato della Predizione")
st.markdown(
    f"""
    <div style="background-color:#f0f9ff; padding:10px; border-radius:10px; border: 2px solid #2196f3;">
        <h2 style="color:#2196f3; text-align:center;">Prezzo Medio Giornaliero Predetto (ADR): <b>{prediction:.2f} €</b></h2>
        <p><i>Tipo di Hotel: {hotel_type}</i></p>
        <p><i>Mese Selezionato: {season_month}</i></p>
    </div>
    """,
    unsafe_allow_html=True
)

# Spazio tra le sezioni
st.markdown("---")

# Metriche del modello
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Spazio tra le sezioni
st.markdown("---")

# Grafico a torta sulle tipologie di hotel
st.subheader("🌈 Distribuzione delle Tipologie di Hotel")
hotel_counts = df['hotel'].value_counts()
fig_pie = px.pie(
    values=hotel_counts.values,
    names=hotel_counts.index,
    title="Distribuzione delle Prenotazioni per Tipologia di Hotel",
    color_discrete_sequence=px.colors.sequential.RdBu
)
st.plotly_chart(fig_pie)

# Spazio tra le sezioni
st.markdown("---")

# Grafico interattivo con Plotly
st.subheader("🔄 Distribuzione delle Predizioni (Grafico Interattivo)")
fig = px.histogram(y_pred, nbins=20, title="Distribuzione delle Predizioni ADR", labels={"value": "ADR Predetto"})
fig.add_vline(x=prediction, line=dict(dash="dash", color="red"), annotation_text="Valore Utente", annotation_position="top right")
st.plotly_chart(fig)


import tempfile

# Funzione per creare il PDF e salvarlo in un file temporaneo
def create_pdf(prediction, mae, r2, plot_image):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    
    # Titolo del documento
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, 750, "📊 Report Predizione ADR")
    
    # Corpo del documento
    c.setFont("Helvetica", 12)
    c.drawString(72, 730, f"Prezzo Medio Giornaliero Predetto (ADR): {prediction:.2f} €")
    c.drawString(72, 710, f"Errore Medio Assoluto (MAE): {mae:.2f}")
    c.drawString(72, 690, f"Coefficiente di Determinazione (R²): {r2:.2f}")
    
    # Aggiungi il grafico nel PDF
    c.drawImage(plot_image, 72, 400, width=450, height=250)
    
    # Salvataggio del PDF
    c.showPage()
    c.save()
    
    buf.seek(0)
    return buf

# Salvataggio del PDF in un file temporaneo
pdf_buffer = create_pdf(prediction, mae, r2, plot_image_path)

# Scrittura del buffer in un file temporaneo
with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
    tmp_file.write(pdf_buffer.getvalue())
    tmp_file_path = tmp_file.name

# Crea il bottone per scaricare il PDF
if os.path.exists(tmp_file_path):
    st.download_button(
        label="Scarica il Report PDF",
        data=open(tmp_file_path, "rb"),
        file_name="report_predizione_adr.pdf",
        mime="application/pdf"
    )
else:
    st.error("Errore nella creazione del PDF. Per favore, riprova.")
