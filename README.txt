================================================================================
ML MARKETING OPTIMIZER - SETUP GUIDE
================================================================================

Developed by: Mike O. Wiesener
Version: 2.0
Date: 2024

================================================================================
SCHNELLSTART
================================================================================

1. VORAUSSETZUNGEN
   - Python 3.8 oder höher
   - pip (Python Package Manager)

2. INSTALLATION

   Schritt 1: Repository klonen oder herunterladen
   -----------------------------------------------
   git clone <repository-url>
   cd "campaign_optimizer"

   Schritt 2: Abhängigkeiten installieren
   ---------------------------------------
   pip install -r requirements.txt

   Schritt 3: Streamlit-App starten
   ---------------------------------
   streamlit run streamlit_app.py

   Die App öffnet sich automatisch im Browser unter:
   http://localhost:8501

3. ERSTE SCHRITTE IN DER APP

   a) Daten laden:
      - Klicken Sie in der Sidebar auf "CSV-Datei hochladen"
      - Wählen Sie Ihre Kundendaten-CSV aus
      - Starten Sie den Prozess über "Daten laden & Pipeline starten"

   b) Ohne eigene Daten:
      - Klicken Sie direkt auf "Daten laden & Pipeline starten"
      - Die App lädt automatisch den Standard-Datensatz

   c) Parameter anpassen:
      - Nutzen Sie die Karten in der Sidebar:
        * "Modell-Tuning": ML-Parameter wie Anzahl Bäume, Tiefe, Lernrate
        * "Kundensegmentierung": Cluster-Anzahl für die Segmentierung
        * "Business-Parameter": Kosten- und Umsatzannahmen
        * "Live-Filter": Alters- und Einkommensbereiche

   d) Tabs erkunden:
      - ML Dashboard: Modell-Performance, KPI-Heldensektion und Feature Importance
      - Interaktiver ROI: ROI-Szenarien und Optimierung
      - Modell-Vergleich: Performance-Vergleiche verschiedener Modelle
      - Live Analytics: 3D-Visualisierungen und Live-Statistiken

================================================================================
DATEISTRUKTUR
================================================================================

campaign_optimizer/
├── streamlit_app.py          # Main web app
├── requirements.txt           # Python dependencies
├── README.txt                 # This file
├── Marktkampagne.csv          # Sample dataset
└── src/                       # Core modules
    ├── config.py
    ├── data_processor.py
    ├── ml_models.py
    ├── business_analytics.py
    └── visualization.py

================================================================================
CSV-DATENFORMAT
================================================================================

Ihre CSV-Datei sollte folgende Spalten enthalten:

PFLICHTFELDER:
- Geburtsjahr
- Bildungsniveau
- Familienstand
- Einkommen
- Kinder_zu_Hause
- Teenager_zu_Hause
- Datum_Kunde
- Ausgaben_Wein
- Ausgaben_Obst
- Ausgaben_Fleisch
- Ausgaben_Fisch
- Ausgaben_Süßigkeiten
- Ausgaben_Gold
- Anzahl_Webkäufe
- Anzahl_Katalogkäufe
- Anzahl_Ladeneinkäufe
- Kampagne_1_Akzeptiert
- Kampagne_2_Akzeptiert
- Kampagne_3_Akzeptiert
- Kampagne_4_Akzeptiert
- Kampagne_5_Akzeptiert
- Antwort_Letzte_Kampagne

BEISPIEL:
Geburtsjahr,Bildungsniveau,Familienstand,Einkommen,...
1970,Bachelor,Verheiratet,50000,...

================================================================================
FEATURES
================================================================================

✓ ML Models:
  - Random Forest
  - Gradient Boosting
  - Logistic Regression

✓ Customer Segmentation:
  - K-Means clustering
  - Auto segment labeling

✓ ROI Optimization:
  - 5 targeting scenarios
  - Real-time filtering
  - Profit forecasting

✓ Interactive Viz:
  - 3D scatter plots
  - Feature importance
  - ROI comparisons
  - Segment performance

✓ Live Parameter Tuning:
  - Real-time updates
  - Dynamic filters
  - Instant visualization

================================================================================
HÄUFIGE PROBLEME & LÖSUNGEN
================================================================================

Issue: "ModuleNotFoundError"
Fix: pip install -r requirements.txt

Issue: "Port already in use"
Fix: streamlit run streamlit_app.py --server.port 8502

Issue: "File not found"
Fix: Make sure you're in campaign_optimizer directory

Issue: App won't load
Fix: Ctrl+C and restart

================================================================================
SYSTEMANFORDERUNGEN
================================================================================

Minimum:
- Python 3.8+
- 4 GB RAM
- 500 MB freier Speicherplatz

Empfohlen:
- Python 3.10+
- 8 GB RAM
- 1 GB freier Speicherplatz

================================================================================
SUPPORT & KONTAKT
================================================================================

Bei Fragen oder Problemen:
- Erstellen Sie ein Issue auf GitHub
- Kontaktieren Sie: Mike O. Wiesener

================================================================================
LIZENZ
================================================================================

Dieses Projekt ist für Bildungszwecke entwickelt worden.
Alle Rechte vorbehalten.

================================================================================
VERSION HISTORY
================================================================================

v2.0 (2024)
- Modular architecture
- Streamlit web interface
- Live parameter tuning
- Interactive visualizations
- ROI optimization

v1.0 (2024)
- Initial version
- Jupyter notebook
- Basic ML models

================================================================================
