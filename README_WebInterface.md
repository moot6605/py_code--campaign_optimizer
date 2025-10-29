# Marketing Campaign Optimizer - Web Interface

## Overview

Streamlit-based web app for interactive marketing campaign analysis and optimization using ML.

## Features

### 1. **Dashboard**
- KPI overview
- Conversion probability distributions
- Customer segment visualizations
- Real-time metrics

### 2. **ROI Optimizer**
- Interactive parameter config
- Multiple targeting scenarios
- Automated ROI calculation
- Profit optimization

### 3. **Customer Segmentation**
- K-Means clustering results
- Segment-specific stats
- Comparative visualizations
- Per-segment analysis

### 4. **Single Customer Scoring**
- Real-time customer evaluation
- Interactive input form
- Instant conversion prediction
- Action recommendations

### 5. **Batch Analysis**
- Top customer identification
- Filterable lists
- CSV export functionality
- Bulk ROI calculations

## Installation & Setup

### Voraussetzungen
- Python 3.8 oder höher
- Windows/Linux/macOS

### Quick Start (Windows)
1. Double-click `start_webapp.bat`
2. Script auto-installs dependencies
3. Browser opens automatically

### Manual Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start app
streamlit run streamlit_app.py
```

### URL
Nach dem Start ist die Anwendung verfügbar unter:
**http://localhost:8501**

## Dateien

| Datei | Beschreibung |
|-------|-------------|
| `streamlit_app.py` | Hauptanwendung mit vollständigem UI |
| `start_webapp.bat` | Windows Start-Script |
| `requirements.txt` | Python-Abhängigkeiten |
| `Marktkampagne.csv` | Standard-Datensatz |

## Bedienung

### 1. Load Data
- **Sidebar** → Click "Load and process data"
- Optional: Upload custom CSV
- Auto data processing and model training

### 2. Configure Parameters
- **Campaign cost per customer**: Sidebar slider
- **Revenue per conversion**: Sidebar slider
- Parameters update all calculations in real-time

### 3. Use Dashboard
- **Tab "Dashboard"**: KPI overview
- **Tab "ROI Optimizer"**: Find best targeting strategy
- **Tab "Customer Segments"**: Segment analysis
- **Tab "Single Customer Scoring"**: Evaluate new customers
- **Tab "Batch Analysis"**: Export top customers

## Technische Details

### Architecture
```
streamlit_app.py
├── MarketingCampaignOptimizer (main class)
│   ├── load_data() - data import
│   ├── preprocess_data() - feature engineering
│   ├── perform_clustering() - customer segmentation
│   ├── train_model() - ML model training
│   └── calculate_roi_scenarios() - ROI calculation
└── main() - Streamlit UI layout
```

### ML Pipeline
1. **Data cleaning**: Outlier handling, missing values
2. **Feature engineering**: 15+ new features
3. **Clustering**: K-Means for customer segmentation
4. **Modeling**: Random Forest classifier
5. **Scoring**: Real-time conversion predictions

### Business Logic
- **ROI-Berechnung**: `(Umsatz - Kosten) / Kosten * 100`
- **Targeting-Szenarien**: Top 10%, 25%, 50%, 75%, Alle Kunden
- **Conversion-Scoring**: Wahrscheinlichkeiten 0-100%

## Verwendete Metriken

| Metrik | Beschreibung | Berechnung |
|--------|-------------|------------|
| **Conversion-Wahrscheinlichkeit** | ML-Vorhersage für Kampagnen-Response | Random Forest Modell |
| **ROI** | Return on Investment | (Gewinn / Kosten) × 100 |
| **Kundenwert** | Gesamtausgaben pro Kunde | Summe aller Produktkategorien |
| **Engagement-Score** | Kampagnen-Teilnahme Historie | Anzahl akzeptierter Kampagnen |

## Business Use Cases

### Marketing Manager
- **Kampagnen-Budgets** optimal allokieren
- **Zielgruppen** datenbasiert auswählen
- **ROI-Prognosen** für verschiedene Strategien

### Data Analyst
- **Kundensegmente** analysieren und verstehen
- **Performance-Metriken** überwachen
- **A/B-Test Ergebnisse** evaluieren

### Sales Team
- **High-Value Kunden** identifizieren
- **Conversion-Wahrscheinlichkeiten** nutzen
- **Prioritäten** bei der Kundenansprache setzen

## Datenschutz & Sicherheit

- **Lokale Verarbeitung**: Alle Daten bleiben auf dem lokalen System
- **Keine Cloud-Übertragung**: Keine Daten werden an externe Server gesendet
- **Session-basiert**: Daten werden nur während der Browser-Session gespeichert

## Troubleshooting

### Häufige Probleme

**Issue**: "File not found"
**Fix**: Ensure `Marktkampagne.csv` is in same directory

**Issue**: "Module not found"
**Fix**: Run `pip install -r requirements.txt`

**Issue**: "Port already in use"
**Fix**: Use different port: `streamlit run streamlit_app.py --server.port 8502`

### Support
Bei Problemen oder Fragen:
1. Prüfen Sie die Konsolen-Ausgabe auf Fehlermeldungen
2. Stellen Sie sicher, dass alle Abhängigkeiten installiert sind
3. Überprüfen Sie die Datenformat-Kompatibilität

## Erweiterte Funktionen

### Custom Data Upload
- Unterstützt CSV-Dateien mit gleicher Spaltenstruktur
- Automatische Datenvalidierung
- Fehlerbehandlung bei inkompatiblen Formaten

### Export-Funktionen
- **CSV-Export** für Top-Kunden Listen
- **Zeitstempel** in Dateinamen
- **Filterbare Exporte** basierend auf Kriterien

### Real-time Updates
- **Parameter-Änderungen** werden sofort übernommen
- **Interaktive Visualisierungen** mit Plotly
- **Responsive Design** für verschiedene Bildschirmgrößen

---

**Developed for ML final project - Marketing Campaign Analysis**

*Version 1.0 - Production-ready web interface*