import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.express as px  # Importiamo Plotly per il grafico interattivo
import matplotlib.pyplot as plt
import seaborn as sns
import io
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.pdfgen import canvas
import os

# Caricare dataset
df_csv = pd.read_csv('hotel_bookings.csv')

# Aggiungere una colonna "season" al dataset
def map_season(month):
    if month in ['December', 'January', 'February']:
        return 'Inverno'
    elif month in ['March', 'April', 'May']:
        return 'Primavera'
    elif month in ['June', 'July', 'August']:
        return 'Estate'
    else:
        return 'Autunno'

df_csv['season'] = df_csv['arrival_date_month'].map(map_season)

# Simuliamo i dati (sostituire con il vostro dataset)
np.random.seed(0)
data = {
    'adults': np.random.randint(1, 5, 100),
    'children': np.random.randint(0, 3, 100),
    'lead_time': np.random.randint(0, 500, 100),
    'stays_in_week_nights': np.random.randint(0, 10, 100),
    'stays_in_weekend_nights': np.random.randint(0, 5, 100),
    'adr': np.random.randint(50, 300, 100),
    'hotel': np.random.choice(['City Hotel', 'Resort Hotel'], 100)  # Colonna 'hotel' con City Hotel e Resort Hotel
}
df = pd.DataFrame(data)

# Aggiungiamo la colonna "stay_duration"
df['stay_duration'] = df['stays_in_week_nights'] + df['stays_in_weekend_nights']

# Codifica della colonna 'hotel' (one-hot encoding)
hotel_types = pd.get_dummies(df['hotel'], drop_first=True)  # Dropping the first column to avoid multicollinearity
df = pd.concat([df, hotel_types], axis=1)

# Prepariamo i dati
X = df[['adults', 'children', 'lead_time', 'stay_duration', 'Resort Hotel']]
y = df['adr']

X = X.copy()
X.fillna(X.mean(), inplace=True)
y = y.copy()
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

# Predizione
user_input = pd.DataFrame({
    'adults': [adults],
    'children': [children],
    'lead_time': [lead_time],
    'stay_duration': [stay_duration],
    'Resort Hotel': [hotel_encoding[0]]
})

prediction = model.predict(user_input)[0]

# Risultati
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

# Metriche del modello
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

# Grafico interattivo con Plotly
st.subheader("📊 Distribuzione delle Predizioni (Grafico Interattivo)")
fig = px.histogram(y_pred, nbins=20, title="Distribuzione delle Predizioni ADR", labels={"value": "ADR Predetto"})
fig.add_vline(x=prediction, line=dict(dash="dash", color="red"), annotation_text="Valore Utente", annotation_position="top right")
st.plotly_chart(fig)

# Funzione per salvare il grafico come immagine con miglioramenti estetici
def save_plot_image():
    # Crea la directory se non esiste
    plot_dir = 'C:\\tmp'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    plot_image_path = os.path.join(plot_dir, 'prediction_plot.png')

    # Crea il grafico con un tema migliorato
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    sns.histplot(y_pred, bins=20, color='royalblue', label='Predizioni ADR', kde=False)
    # Curva di densità KDE con il colore arancione
    sns.kdeplot(y_pred, color='orange', linewidth=2, label='Curva di densità KDE delle Predizioni ADR')
    plt.axvline(prediction, color='red', linestyle='--', label=f'Valore Utente ({prediction:.2f} €)', linewidth=2)
    plt.title('Distribuzione delle Predizioni ADR con il Valore Utente', fontsize=16, fontweight='bold', color='darkblue')
    plt.xlabel('ADR Predetto (€)', fontsize=14)
    plt.ylabel('Frequenza', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_image_path)  # Salva direttamente l'immagine come file PNG
    plt.close()
    
    return plot_image_path

# Funzione per generare il PDF
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
    c.drawImage(plot_image, 72, 400, width=450, height=250)  # Posiziona il grafico nel PDF
    
    # Salvataggio del PDF
    c.showPage()
    c.save()
    
    buf.seek(0)
    return buf

# Bottone per scaricare il PDF
st.subheader("📥 Scarica il Report PDF")
plot_image = save_plot_image()
pdf_buffer = create_pdf(prediction, mae, r2, plot_image)

# Crea il link per scaricare il PDF
st.download_button(
    label="Scarica il Report PDF",
    data=pdf_buffer,
    file_name="report_predizione_adr.pdf",
    mime="application/pdf"
)

# Sidebar per selezionare la stagione
st.sidebar.header("🌦️ Filtra per Stagione")
season_filter = st.sidebar.selectbox(
    "Seleziona una stagione", 
    options=['Tutte', 'Inverno', 'Primavera', 'Estate', 'Autunno']
)

# Filtrare il dataset in base alla stagione selezionata
if season_filter != 'Tutte':
    df = df[df['season'] == season_filter]

# (Resto del codice rimane invariato)

