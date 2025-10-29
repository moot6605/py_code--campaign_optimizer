# Campaign Performance Optimizer

## Overview

ML pipeline for marketing campaign optimization. Combines data preprocessing, model training, customer segmentation, ROI analysis and Streamlit frontend. Runs locally without external dependencies.

## Core Features
- **Data Processing**: ETL pipeline in `src/data_processor.py`
- **ML Models**: Training, scoring, persistence via `src/ml_models.py`
- **Business Analytics**: ROI scenarios, segmentation analysis (`src/business_analytics.py`)
- **Visualization**: Plotly charts and dashboards (`src/visualization.py`)
- **Web Interface**: Interactive dashboard via `streamlit_app.py`
- **CLI**: Full pipeline orchestration through `main.py`

## Verzeichnisstruktur
```
campaign_optimizer/
├── README.md                 # Projektuebersicht (diese Datei)
├── README.txt                # Schritt-fuer-Schritt-Anleitung fuer Endnutzer
├── README_WebInterface.md    # Vertiefte Dokumentation der Streamlit-App
├── requirements.txt          # Python-Abhaengigkeiten
├── Marktkampagne.csv         # Beispieldatensatz (lokal verarbeitbar)
├── main.py                   # CLI-Einstieg fuer die gesamte Pipeline
├── streamlit_app.py          # Interaktives Dashboard
├── streamlit_app_modular.py  # Modulare Referenz-Implementierung
├── src/
│   ├── __init__.py
│   ├── business_analytics.py
│   ├── config.py
│   ├── data_processor.py
│   ├── ml_models.py
│   └── visualization.py
└── start_webapp.bat          # Windows-Starter fuer Streamlit
```

## Quick Start
1. Python >=3.10 required
2. Setup venv (optional):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate   # Linux/macOS
   ```
3. Install deps:
   ```bash
   pip install -r requirements.txt
   ```
4. Run Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```
   Available at `http://localhost:8501`
5. CLI pipeline:
   ```bash
   python main.py --generate-report
   ```

## Data Requirements
- Sample dataset: `Marktkampagne.csv`
- Custom CSV must match required schema (see README.txt)
- All processing runs locally, no external data transmission

## Development
- Code style: PEP8 (use `black` or `ruff` for formatting)
- No unit tests included - test modules via `python -m`
- Exclude `__pycache__` and venv when packaging (see `.gitignore`)

## Git Setup
1. Add `.gitignore` and license file
2. Init repo:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```
3. Add remote:
   ```bash
   git remote add origin <git-url>
   git push -u origin main
   ```
4. Optional: Setup CI/CD for automated testing

## License
No license specified. Add appropriate license file (MIT, Apache 2.0, etc.) before publishing.
