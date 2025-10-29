"""
Marketing Campaign Optimizer - Interactive Analysis Tool
Developed by Mike O. Wiesener

Interactive ML playground for marketing campaign optimization.
Real-time parameter tuning with immediate visual feedback.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from datetime import datetime
import html
import io
import time
import json

# Page config for optimal display
st.set_page_config(
    page_title="Marketing Performance Optimizer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern CSS with animations and gradients
st.markdown("""
<style>
:root {
    --app-bg: radial-gradient(circle at 0% 0%, #0b1220 0%, #0e172a 40%, #111827 100%);
    --surface-glass: rgba(15, 23, 42, 0.72);
    --surface-solid: rgba(15, 23, 42, 0.88);
    --border-soft: rgba(148, 163, 184, 0.18);
    --border-strong: rgba(94, 234, 212, 0.45);
    --text-strong: #f1f5f9;
    --text-muted: rgba(226, 232, 240, 0.72);
    --accent-primary: #6366f1;
    --accent-secondary: #38bdf8;
    --accent-tertiary: #f472b6;
    --success: #34d399;
    --warning: #facc15;
    --danger: #f87171;
    --radius-lg: 28px;
    --radius-md: 18px;
    --radius-sm: 12px;
    --shadow-card: 0 24px 45px rgba(15, 23, 42, 0.45);
    --shadow-hero: 0 32px 60px rgba(14, 165, 233, 0.28);
    --transition: all 0.28s ease;
    --font-stack: "Inter", "Segoe UI", -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif;
}
html, body, [data-testid="stAppViewContainer"] {
    background: var(--app-bg);
    color: var(--text-strong);
}
[data-testid="stHeader"] {
    background: transparent;
}
[data-testid="stHeader"] button,
[data-testid="collapsedControl"] button,
button[data-testid="collapsedControl"] {
    background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary)) !important;
    border: 1px solid rgba(148, 163, 184, 0.35) !important;
    color: var(--text-strong) !important;
    font-weight: 700 !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: 0 18px 32px rgba(79, 70, 229, 0.3) !important;
    transition: var(--transition) !important;
}
[data-testid="stHeader"] button:hover,
[data-testid="collapsedControl"] button:hover,
button[data-testid="collapsedControl"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 20px 40px rgba(56, 189, 248, 0.35) !important;
}

.main {
    background: transparent;
    padding: 0 2.4rem 4rem;
}
* {
    font-family: var(--font-stack);
}
h1, h2, h3, h4, h5 {
    color: var(--text-strong);
    letter-spacing: -0.01em;
}
p, span, label, li, div {
    color: var(--text-muted);
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(11, 15, 28, 0.96), rgba(9, 12, 24, 0.92));
    border-right: 1px solid var(--border-soft);
    padding: 1.5rem 1.2rem 3rem;
}
section[data-testid="stSidebar"] > div {
    background: transparent;
}
.sidebar-intro {
    margin-bottom: 1.5rem;
    text-transform: uppercase;
    font-size: 0.75rem;
    letter-spacing: 0.18em;
    color: var(--accent-secondary);
}
.sidebar-card {
    background: var(--surface-glass);
    border: 1px solid var(--border-soft);
    border-radius: var(--radius-md);
    padding: 1.2rem 1.4rem 1.35rem;
    margin-bottom: 1.1rem;
    box-shadow: 0 14px 32px rgba(8, 15, 35, 0.45);
    backdrop-filter: blur(18px);
}
.sidebar-card h3, .sidebar-card strong {
    color: var(--text-strong);
}
.sidebar-card small {
    color: var(--text-muted);
    display: block;
    margin-top: 0.35rem;
}
.sidebar-card .help-text {
    font-size: 0.8rem;
    opacity: 0.75;
}
[data-testid="stSidebar"] .stSlider label {
    font-weight: 600;
}
[data-testid="stSidebar"] .stSlider div[data-baseweb="slider"] {
    background: rgba(255, 255, 255, 0.04);
    border-radius: 999px;
}
[data-testid="stSidebar"] .stSlider span[data-baseweb="slider"] > div {
    background: var(--accent-secondary);
}
[data-testid="stSidebar"] .stSlider span[data-baseweb="slider"] > div > div {
    background: var(--accent-secondary);
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: rgba(15, 23, 42, 0.65);
    border: 1px dashed var(--border-soft);
    border-radius: var(--radius-sm);
}
[data-testid="stFileUploader"] section {
    padding: 0.75rem;
}
[data-testid="baseButton-primary"] button,
[data-testid="baseButton-secondary"] button,
section[data-testid="stSidebar"] button {
    background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
    border: 1px solid rgba(148, 163, 184, 0.35);
    color: var(--text-strong);
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 0.75rem 1.4rem;
    border-radius: var(--radius-sm);
    box-shadow: 0 18px 32px rgba(79, 70, 229, 0.3);
    transition: var(--transition);
}
[data-testid="baseButton-primary"] button:hover,
[data-testid="baseButton-secondary"] button:hover,
section[data-testid="stSidebar"] button:hover {
    transform: translateY(-2px);
    box-shadow: 0 20px 40px rgba(56, 189, 248, 0.35);
}
[data-testid="baseButton-primary"] button:focus-visible,
[data-testid="baseButton-secondary"] button:focus-visible {
    outline: 2px solid var(--accent-secondary);
    outline-offset: 3px;
}
.hero {
    position: relative;
    overflow: hidden;
    border-radius: var(--radius-lg);
    padding: 2.6rem 3rem;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 2.8rem;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.92), rgba(14, 165, 233, 0.9));
    color: #ffffff;
    box-shadow: var(--shadow-hero);
}
.hero::after {
    content: "";
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at 20% 20%, rgba(255, 255, 255, 0.22), transparent 55%);
    mix-blend-mode: screen;
    opacity: 0.7;
}
.hero__content {
    position: relative;
    z-index: 2;
}
.hero__eyebrow {
    text-transform: uppercase;
    font-size: 0.78rem;
    letter-spacing: 0.18em;
    opacity: 0.78;
    margin-bottom: 0.85rem;
    display: inline-block;
}
.hero__title {
    font-size: 2.8rem;
    font-weight: 800;
    margin-bottom: 0.35rem;
}
.hero__subtitle {
    font-size: 1.1rem;
    line-height: 1.6;
    color: rgba(255, 255, 255, 0.9);
    max-width: 460px;
}
.hero__chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    margin-top: 1.35rem;
}
.status-chip {
    background: rgba(15, 23, 42, 0.32);
    border: 1px solid rgba(255, 255, 255, 0.25);
    border-radius: 999px;
    padding: 0.45rem 1.1rem;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    color: #ffffff;
    backdrop-filter: blur(22px);
    transition: var(--transition);
}
.status-chip--success {
    background: rgba(16, 185, 129, 0.35);
    border-color: rgba(16, 185, 129, 0.55);
}
.status-chip--pending {
    background: rgba(244, 114, 182, 0.35);
    border-color: rgba(244, 114, 182, 0.55);
}
.hero__metrics {
    position: relative;
    z-index: 2;
    display: grid;
    gap: 1.2rem;
}
.hero-metric {
    background: rgba(15, 23, 42, 0.45);
    border: 1px solid rgba(148, 163, 184, 0.35);
    border-radius: var(--radius-md);
    padding: 1.1rem 1.4rem;
    box-shadow: 0 20px 32px rgba(15, 23, 42, 0.42);
}
.hero-metric__label {
    font-size: 0.82rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    opacity: 0.68;
}
.hero-metric__value {
    font-size: 1.9rem;
    font-weight: 700;
    margin-top: 0.5rem;
}
.section-header {
    margin: 3rem 0 1.4rem;
    font-size: 1.4rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.section-header::before {
    content: "";
    display: inline-block;
    width: 40px;
    height: 3px;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--accent-primary), var(--accent-tertiary));
}
.metric-card {
    position: relative;
    background: var(--surface-glass);
    border: 1px solid var(--border-soft);
    border-radius: var(--radius-md);
    padding: 1.65rem 1.6rem;
    box-shadow: var(--shadow-card);
    overflow: hidden;
    transition: var(--transition);
}
.metric-card::after {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.22), rgba(244, 114, 182, 0.18));
    opacity: 0;
    transition: var(--transition);
}
.metric-card:hover {
    transform: translateY(-3px);
}
.metric-card:hover::after {
    opacity: 1;
}
.metric-card__label {
    font-size: 0.9rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--text-muted);
}
.metric-card__value {
    position: relative;
    z-index: 2;
    font-size: 2.4rem;
    font-weight: 700;
    margin-top: 0.35rem;
    color: var(--text-strong);
}
.metric-card__hint {
    position: relative;
    z-index: 2;
    font-size: 0.85rem;
    margin-top: 0.55rem;
    color: var(--accent-secondary);
}
.model-card {
    background: var(--surface-glass);
    border: 1px solid var(--border-soft);
    border-radius: var(--radius-md);
    padding: 1.4rem 1.6rem;
    box-shadow: var(--shadow-card);
    transition: var(--transition);
    color: var(--text-strong);
    margin-bottom: 1rem;
}
.model-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 24px 40px rgba(99, 102, 241, 0.32);
}
.model-card h4 {
    margin-bottom: 0.6rem;
    color: var(--text-strong);
}
.model-card p {
    color: var(--text-muted);
    margin-bottom: 0.3rem;
}
.surface-card {
    background: var(--surface-solid);
    border-radius: var(--radius-lg);
    border: 1px solid var(--border-soft);
    padding: 2rem 2.4rem;
    box-shadow: 0 26px 60px rgba(15, 23, 42, 0.45);
    margin-bottom: 1.8rem;
}
.success-card,
.info-card {
    background: var(--surface-glass);
    border-radius: var(--radius-md);
    border: 1px solid var(--border-strong);
    padding: 1.4rem 1.6rem;
    box-shadow: var(--shadow-card);
    margin: 1.2rem 0;
}
.success-card {
    border-color: rgba(52, 211, 153, 0.55);
    background: rgba(16, 185, 129, 0.18);
}
.info-card {
    border-color: rgba(56, 189, 248, 0.55);
    background: rgba(14, 165, 233, 0.16);
}
.stTabs [data-baseweb="tab-list"] {
    border-radius: var(--radius-md);
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid var(--border-soft);
    padding: 0.4rem;
    gap: 0.4rem;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: var(--radius-sm);
    border: 1px solid transparent;
    color: var(--text-muted);
    font-weight: 600;
    letter-spacing: 0.02em;
    transition: var(--transition);
}
.stTabs [data-baseweb="tab"]:hover {
    border-color: rgba(99, 102, 241, 0.55);
    color: var(--text-strong);
}
.stTabs [aria-selected="true"] {
    border-color: rgba(99, 102, 241, 0.65);
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.55), rgba(56, 189, 248, 0.55));
    color: var(--text-strong);
    box-shadow: 0 12px 28px rgba(99, 102, 241, 0.35);
}
.stPlotlyChart {
    background: var(--surface-glass);
    border-radius: var(--radius-lg);
    padding: 0.75rem;
    border: 1px solid var(--border-soft);
    box-shadow: 0 20px 45px rgba(15, 23, 42, 0.4);
}
[data-testid="stDataFrame"] {
    background: var(--surface-glass);
    border-radius: var(--radius-md);
    padding: 0.6rem 0.8rem 0.8rem;
    border: 1px solid var(--border-soft);
    box-shadow: 0 18px 36px rgba(15, 23, 42, 0.38);
}
[data-testid="stDataFrame"] table {
    color: var(--text-muted);
}
[data-testid="stDataFrame"] thead tr {
    background: rgba(99, 102, 241, 0.26);
    color: var(--text-strong);
}
[data-testid="stDataFrame"] tbody tr {
    border-bottom: 1px solid rgba(148, 163, 184, 0.15);
}
[data-testid="stDataFrame"] tbody tr:hover {
    background: rgba(56, 189, 248, 0.08);
}
.stAlert {
    background: var(--surface-glass);
    border-radius: var(--radius-md);
    border: 1px solid var(--border-soft);
    box-shadow: 0 16px 32px rgba(15, 23, 42, 0.4);
}
button[kind="header"],
[data-testid="baseButton-header"],
[data-testid="stSidebarNav"] button,
section[data-testid="stSidebar"] button[kind="header"] {
    background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary)) !important;
    border: 1px solid rgba(148, 163, 184, 0.35) !important;
    color: var(--text-strong) !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    padding: 0.75rem 1.4rem !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: 0 18px 32px rgba(79, 70, 229, 0.3) !important;
    transition: var(--transition) !important;
}
button[kind="header"]:hover,
[data-testid="baseButton-header"]:hover,
[data-testid="stSidebarNav"] button:hover,
section[data-testid="stSidebar"] button[kind="header"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 20px 40px rgba(56, 189, 248, 0.35) !important;
}
[role="menu"],
[data-baseweb="popover"] > div,
[data-baseweb="menu"] {
    background: var(--surface-solid) !important;
}
[role="menuitem"],
[role="menuitem"] span,
[role="menuitem"] p,
[data-baseweb="menu"] li,
[data-baseweb="menu"] li span {
    color: #000000 !important;
}
[role="menuitem"]:hover {
    background: rgba(0, 0, 0, 0.1) !important;
}
[data-baseweb="modal"],
[data-baseweb="popover"],
[role="dialog"],
[role="alertdialog"] {
    background: #ffffff !important;
}
[data-baseweb="modal"] *,
[data-baseweb="popover"] *,
[role="dialog"] *,
[role="alertdialog"] *,
[data-baseweb="modal"] span,
[data-baseweb="modal"] p,
[data-baseweb="modal"] div,
[data-baseweb="popover"] span,
[data-baseweb="popover"] p,
[data-baseweb="popover"] div {
    color: #000000 !important;
}
.stDownloadButton button {
    background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary)) !important;
    border: 1px solid rgba(148, 163, 184, 0.35) !important;
    color: var(--text-strong) !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    padding: 0.75rem 1.4rem !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: 0 18px 32px rgba(79, 70, 229, 0.3) !important;
    transition: var(--transition) !important;
}
.stDownloadButton button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 20px 40px rgba(56, 189, 248, 0.35) !important;
}
@media (max-width: 1024px) {
    .main {
        padding: 0 1.4rem 3rem;
    }
    .hero {
        grid-template-columns: 1fr;
        padding: 2.2rem;
    }
}
</style>
""", unsafe_allow_html=True)


def format_number(value) -> str:
    """Format numbers with dot thousands separator"""
    try:
        if value is None or (isinstance(value, (float, int)) and pd.isna(value)):
            return "--"
    except TypeError:
        return "--"
    try:
        return f"{value:,.0f}".replace(",", ".")
    except (TypeError, ValueError):
        return "--"


def format_percentage(value) -> str:
    """Format ratios as percentage"""
    try:
        if value is None or pd.isna(value):
            return "--"
        return f"{value:.1%}"
    except (TypeError, ValueError):
        return "--"


def format_currency(value) -> str:
    """Format as EUR currency"""
    try:
        if value is None or pd.isna(value):
            return "--"
        return f"{format_number(value)}&nbsp;&euro;"
    except (TypeError, ValueError):
        return "--"


def render_metric_card(label: str, value: str, hint: str | None = None) -> None:
    """Render a styled metric card block."""
    hint_html = f'<div class="metric-card__hint">{html.escape(hint)}</div>' if hint else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-card__label">{html.escape(label)}</div>
            <div class="metric-card__value">{value}</div>
            {hint_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero_section(optimizer, data_ready: bool) -> None:
    """Render the top hero with dataset status and highlight numbers."""
    dataset_label = html.escape(getattr(optimizer, "data_source", "Noch keine Daten"))
    last_training = html.escape(st.session_state.get("last_training", "Noch nicht trainiert"))

    status_class = "status-chip--success" if data_ready else "status-chip--pending"
    status_text = "Daten geladen & Modelle aktiv" if data_ready else "Warte auf Datenimport"

    if data_ready and getattr(optimizer, "df", None) is not None:
        df = optimizer.df
        total_customers = format_number(len(df))
        avg_conversion = (
            format_percentage(df["Conversion_Wahrscheinlichkeit"].mean())
            if "Conversion_Wahrscheinlichkeit" in df
            else "—"
        )
        segment_count = format_number(df["Segment_Label"].nunique()) if "Segment_Label" in df else "—"
    else:
        total_customers = "—"
        avg_conversion = "—"
        segment_count = "—"

    model_count = format_number(len(getattr(optimizer, "models", {}))) if getattr(
        optimizer, "models", None
    ) else "—"

    st.markdown(
        f"""
        <div class="hero">
            <div class="hero__content">
                <span class="hero__eyebrow">Marketing Intelligence Suite</span>
                <h1 class="hero__title">Campaign Performance Optimizer</h1>
                <p class="hero__subtitle">
                    Experimentiere mit datengetriebenen Szenarien, quantifiziere den ROI und finde
                    die wirkungsvollsten Zielgruppen in Echtzeit.
                </p>
                <div class="hero__chips">
                    <span class="status-chip {status_class}">{status_text}</span>
                    <span class="status-chip">Quelle: {dataset_label}</span>
                    <span class="status-chip">Letztes Training: {last_training}</span>
                </div>
            </div>
            <div class="hero__metrics">
                <div class="hero-metric">
                    <div class="hero-metric__label">Aktive Kunden</div>
                    <div class="hero-metric__value">{total_customers}</div>
                </div>
                <div class="hero-metric">
                    <div class="hero-metric__label">Ø Conversion</div>
                    <div class="hero-metric__value">{avg_conversion}</div>
                </div>
                <div class="hero-metric">
                    <div class="hero-metric__label">Modelle</div>
                    <div class="hero-metric__value">{model_count}</div>
                </div>
                <div class="hero-metric">
                    <div class="hero-metric__label">Segmente</div>
                    <div class="hero-metric__value">{segment_count}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    """Show a friendly onboarding card when no data is available."""
    st.markdown(
        """
        <div class="surface-card info-card">
            <h2>Bereit f&uuml;r datengetriebene Kampagnen</h2>
            <p>Starte mit wenigen Klicks:</p>
            <ul>
                <li>Eigenen CSV-Datensatz hochladen oder den Standard-Datensatz verwenden.</li>
                <li>Modelle trainieren und Segmentierung anstoßen.</li>
                <li>Parameter im Sidebar justieren und Live-Visualisierungen erkunden.</li>
            </ul>
            <p>Alle Berechnungen laufen lokal – deine Daten bleiben in deinem Besitz.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


class CampaignOptimizerSuite:
    """
    Main class for interactive campaign optimizer
    
    Manages:
    - Data processing and feature engineering
    - Multi-model training
    - K-Means customer segmentation
    - Dynamic ROI calculations
    """
    
    def __init__(self):
        """Initialize optimizer with empty containers"""
        self.df = None                    # Hauptdatenframe
        self.models = {}                  # Dictionary für trainierte Modelle
        self.encoders = {}                # Label-Encoder für kategorische Variablen
        self.scaler = StandardScaler()    # Scaler für numerische Features
        self.model_scores = {}            # Performance-Metriken der Modelle
        self.data_source = "No data loaded"  # UI display
        
    def load_data(self, uploaded_file=None):
        """
        Lädt Marketing-Daten aus CSV-Datei
        
        Args:
            uploaded_file: Streamlit UploadedFile object oder None für Standard-Datei
            
        Returns:
            bool: True wenn erfolgreich, False bei Fehler
        """
        try:
            if uploaded_file is not None:
                self.df = pd.read_csv(uploaded_file)
                self.data_source = uploaded_file.name
                st.success(f"Daten geladen: {self.df.shape[0]} Kunden, {self.df.shape[1]} Features")
            else:
                # Try loading default file
                self.df = pd.read_csv('Marktkampagne.csv')
                self.data_source = "Marktkampagne.csv"
                st.info(f"Default dataset: {self.df.shape[0]} customers loaded")
            return True
        except Exception as e:
            st.error(f"Loading error: {str(e)}")
            return False
    
    def preprocess_data(self):
        """
        Führt umfassende Datenvorverarbeitung und Feature Engineering durch
        
        Schritte:
        1. Datenbereinigung (fehlende Werte, Outlier)
        2. Datumskonvertierung
        3. Feature Engineering (Alter, Haushaltsgrößen, Ausgaben-Metriken)
        4. Kategorische Encodierung
        
        Returns:
            bool: True wenn erfolgreich, False bei Fehler
        """
        if self.df is None:
            return False
        
        try:
            # Schritt 1: Datenbereinigung
            # Fehlende Einkommenswerte mit Median auffüllen
            self.df['Einkommen'].fillna(self.df['Einkommen'].median(), inplace=True)
            
            # Schritt 2: Robuste Datumsverarbeitung
            try:
                # Versuche deutsches Datumsformat
                self.df['Datum_Kunde'] = pd.to_datetime(self.df['Datum_Kunde'], format='%d-%m-%Y', errors='coerce')
                # Fallback für nicht konvertierbare Daten
                if self.df['Datum_Kunde'].isna().any():
                    self.df['Datum_Kunde'].fillna(pd.to_datetime('2020-01-01'), inplace=True)
            except:
                # Notfall-Fallback
                self.df['Datum_Kunde'] = pd.to_datetime('2020-01-01')
            
            # Schritt 3: Feature Engineering
            current_year = datetime.now().year
            
            # Demografische Features
            self.df['Alter'] = current_year - self.df['Geburtsjahr']
            self.df['Haushaltsgröße'] = self.df['Kinder_zu_Hause'] + self.df['Teenager_zu_Hause']
            self.df['Hat_Kinder'] = (self.df['Haushaltsgröße'] > 0).astype(int)
            
            # Ausgaben-Analytics Features
            ausgaben_cols = ['Ausgaben_Wein', 'Ausgaben_Obst', 'Ausgaben_Fleisch', 
                            'Ausgaben_Fisch', 'Ausgaben_Süßigkeiten', 'Ausgaben_Gold']
            self.df['Gesamtausgaben'] = self.df[ausgaben_cols].sum(axis=1)
            self.df['Premium_Ausgaben'] = self.df['Ausgaben_Wein'] + self.df['Ausgaben_Gold']
            # Verhältnis Premium zu Gesamtausgaben (mit Schutz vor Division durch 0)
            self.df['Premium_Anteil'] = self.df['Premium_Ausgaben'] / (self.df['Gesamtausgaben'] + 1)
            
            # Kaufverhalten Features
            kauf_cols = ['Anzahl_Webkäufe', 'Anzahl_Katalogkäufe', 'Anzahl_Ladeneinkäufe']
            self.df['Gesamtkäufe'] = self.df[kauf_cols].sum(axis=1)
            # Durchschnittlicher Kaufwert (mit Schutz vor Division durch 0)
            self.df['Durchschnittlicher_Kaufwert'] = self.df['Gesamtausgaben'] / (self.df['Gesamtkäufe'] + 1)
            
            # Engagement Features
            kampagnen_cols = ['Kampagne_1_Akzeptiert', 'Kampagne_2_Akzeptiert', 'Kampagne_3_Akzeptiert', 
                             'Kampagne_4_Akzeptiert', 'Kampagne_5_Akzeptiert']
            self.df['Kampagnen_Engagement'] = self.df[kampagnen_cols].sum(axis=1)
            
            # Loyalitäts-Features
            self.df['Kunde_seit_Tagen'] = (datetime.now() - self.df['Datum_Kunde']).dt.days.clip(lower=0)
            self.df['Loyalitäts_Score'] = self.df['Kunde_seit_Tagen'] / 365
            
            # Schritt 4: Kategorische Encodierung
            self.encoders['bildung'] = LabelEncoder()
            self.encoders['familienstand'] = LabelEncoder()
            
            self.df['Bildungsniveau_encoded'] = self.encoders['bildung'].fit_transform(self.df['Bildungsniveau'])
            self.df['Familienstand_encoded'] = self.encoders['familienstand'].fit_transform(self.df['Familienstand'])
            
            st.success("Datenverarbeitung abgeschlossen - Bereit für Analysen!")
            return True
            
        except Exception as e:
            st.error(f"Fehler bei Verarbeitung: {str(e)}")
            return False
    
    def train_multiple_models(self, n_estimators=100, max_depth=None, learning_rate=0.1):
        """
        Trainiert verschiedene Modelle mit konfigurierbaren Parametern
        
        Args:
            n_estimators (int): Anzahl der Bäume/Estimators für Ensemble-Methoden
            max_depth (int): Maximale Tiefe der Bäume (None = unbegrenzt)
            learning_rate (float): Lernrate für Gradient Boosting
            
        Returns:
            bool: True wenn erfolgreich, False bei Fehler
        """
        try:
            # Feature-Auswahl für das Training
            feature_names = ['Alter', 'Bildungsniveau_encoded', 'Familienstand_encoded', 'Einkommen',
                           'Haushaltsgröße', 'Hat_Kinder', 'Gesamtausgaben', 'Premium_Anteil',
                           'Gesamtkäufe', 'Durchschnittlicher_Kaufwert', 'Kampagnen_Engagement', 
                           'Loyalitäts_Score']
            
            X = self.df[feature_names]  # Feature-Matrix
            y = self.df['Antwort_Letzte_Kampagne']  # Zielvariable
            
            # Verschiedene Modelle mit konfigurierbaren Parametern definieren
            self.models = {
                'Random Forest': RandomForestClassifier(
                    n_estimators=n_estimators, 
                    max_depth=max_depth, 
                    random_state=42
                ),
                'Gradient Boosting': GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth if max_depth else 3,  # Standard-Tiefe für GB
                    learning_rate=learning_rate,
                    random_state=42
                ),
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
            }
            
            # Modelle trainieren und Performance bewerten
            self.model_scores = {}
            predictions = {}
            
            for name, model in self.models.items():
                # Modell trainieren
                model.fit(X, y)
                
                # Cross-Validation für robuste Performance-Bewertung
                cv_scores = cross_val_score(model, X, y, cv=5)
                self.model_scores[name] = {
                    'cv_mean': cv_scores.mean(),      # Durchschnittlicher CV-Score
                    'cv_std': cv_scores.std(),        # Standardabweichung CV-Score
                    'accuracy': model.score(X, y)     # Training-Accuracy
                }
                
                # Vorhersage-Wahrscheinlichkeiten für alle Kunden
                predictions[name] = model.predict_proba(X)[:, 1]
            
            # Beste Modell-Vorhersagen für weitere Analysen verwenden
            best_model_name = max(self.model_scores.keys(), key=lambda x: self.model_scores[x]['cv_mean'])
            self.df['Conversion_Wahrscheinlichkeit'] = predictions[best_model_name]
            
            return True
            
        except Exception as e:
            st.error(f"Fehler beim Modell-Training: {str(e)}")
            return False
    
    def perform_clustering(self, n_clusters=4):
        """
        Führt K-Means Clustering für Kundensegmentierung durch
        
        Args:
            n_clusters (int): Anzahl der gewünschten Cluster
            
        Returns:
            bool: True wenn erfolgreich, False bei Fehler
        """
        try:
            # Features für Clustering auswählen
            clustering_features = ['Alter', 'Einkommen', 'Gesamtausgaben', 'Gesamtkäufe', 
                                  'Premium_Anteil', 'Kampagnen_Engagement', 'Loyalitäts_Score']
            
            X_cluster = self.df[clustering_features].copy()
            
            # Standardisierung für K-Means (wichtig für Distanz-basierte Algorithmen)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_cluster)
            
            # K-Means Clustering durchführen
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.df['Kundensegment'] = kmeans.fit_predict(X_scaled)
            
            # Dynamische Labels basierend auf Cluster-Anzahl erstellen
            segment_labels = {i: f'Segment {i+1}' for i in range(n_clusters)}
            self.df['Segment_Label'] = self.df['Kundensegment'].map(segment_labels)
            
            return True
            
        except Exception as e:
            st.error(f"Fehler beim Clustering: {str(e)}")
            return False
    
    def calculate_dynamic_roi(self, campaign_cost, revenue_per_conversion, age_filter, income_filter):
        """
        Berechnet ROI-Szenarien mit dynamischen Filtern
        
        Args:
            campaign_cost (float): Kosten pro kontaktiertem Kunden
            revenue_per_conversion (float): Umsatz pro erfolgreicher Conversion
            age_filter (tuple): Min/Max Alter für Filterung
            income_filter (tuple): Min/Max Einkommen für Filterung
            
        Returns:
            pd.DataFrame: ROI-Szenarien mit verschiedenen Targeting-Strategien
        """
        # Daten basierend auf Filtern einschränken
        filtered_df = self.df[
            (self.df['Alter'] >= age_filter[0]) & 
            (self.df['Alter'] <= age_filter[1]) &
            (self.df['Einkommen'] >= income_filter[0]) &
            (self.df['Einkommen'] <= income_filter[1])
        ]
        
        # Verschiedene Targeting-Szenarien definieren
        scenarios = {
            'Alle gefilterten Kunden': 0.0,    # Alle Kunden kontaktieren
            'Top 75%': 0.25,                   # Nur obere 75% der Conversion-Wahrscheinlichkeiten
            'Top 50%': 0.5,                    # Nur obere 50%
            'Top 25%': 0.75,                   # Nur obere 25%
            'Top 10%': 0.9                     # Nur obere 10%
        }
        
        results = []
        
        # Für jedes Szenario ROI berechnen
        for scenario_name, threshold in scenarios.items():
            # Kunden basierend auf Conversion-Wahrscheinlichkeit auswählen
            selected = filtered_df[filtered_df['Conversion_Wahrscheinlichkeit'] >= threshold]
            num_customers = len(selected)
            expected_conversions = selected['Conversion_Wahrscheinlichkeit'].sum()
            
            # Business-Metriken berechnen
            total_costs = num_customers * campaign_cost
            expected_revenue = expected_conversions * revenue_per_conversion
            expected_profit = expected_revenue - total_costs
            
            # ROI und Conversion-Rate berechnen
            roi = (expected_profit / total_costs * 100) if total_costs > 0 else 0
            conversion_rate = (expected_conversions / num_customers * 100) if num_customers > 0 else 0
            
            # Ergebnisse sammeln
            results.append({
                'Szenario': scenario_name,
                'Anzahl_Kunden': num_customers,
                'Erwartete_Conversions': round(expected_conversions, 1),
                'Conversion_Rate': round(conversion_rate, 2),
                'ROI': round(roi, 2),
                'Erwarteter_Gewinn': round(expected_profit, 2)
            })
        
        return pd.DataFrame(results)

def show_training_progress(message="Trainiere Modelle..."):
    """
    Zeigt eine animierte Progress-Anzeige während des Trainings
    
    Args:
        message (str): Nachricht die während des Trainings angezeigt wird
    """
    progress_placeholder = st.empty()
    
    with progress_placeholder.container():
        st.markdown(f"""
        <div style="text-align: center; padding: 2rem;">
            <h3>{message}</h3>
            <div class="training-progress" style="width: 100%; margin: 1rem 0;"></div>
            <p>Bitte warten, die Modelle werden mit den neuen Parametern trainiert...</p>
        </div>
        """, unsafe_allow_html=True)
    
    return progress_placeholder


def main():
    """
    Hauptfunktion der Streamlit-Anwendung
    Definiert das UI-Layout und die Interaktionslogik
    """
    if 'optimizer' not in st.session_state:
        st.session_state.optimizer = CampaignOptimizerSuite()

    optimizer = st.session_state.optimizer
    data_ready = hasattr(st.session_state, 'data_ready')

    # Default-Werte für Sidebar-Parameter (werden später überschrieben)
    n_estimators = 100
    max_depth = 10
    learning_rate = 0.1
    n_clusters = 4
    campaign_cost = 6.0
    revenue_per_conversion = 180.0
    age_range = (25, 75)
    income_range = (20000, 100000)

    with st.sidebar:
        st.markdown('<div class="sidebar-intro">Control Center</div>', unsafe_allow_html=True)

        # Daten & Training
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.markdown("### Daten & Training")
        st.markdown("<small>Nutze eigene Kundendaten oder den Beispiel-Datensatz.</small>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("CSV-Datei hochladen", type=['csv'])
        load_clicked = st.button("Daten laden & Pipeline starten", type="primary", key="load_data_btn")
        st.markdown('</div>', unsafe_allow_html=True)

        if load_clicked:
            progress_placeholder = show_training_progress("Bereite Daten auf und starte die Analyse-Pipeline...")
            if optimizer.load_data(uploaded_file):
                if optimizer.preprocess_data():
                    if optimizer.train_multiple_models():
                        if optimizer.perform_clustering():
                            st.session_state.data_ready = True
                            st.session_state.last_training = datetime.now().strftime("%d.%m.%Y %H:%M")
                            data_ready = True
                            progress_placeholder.success("Pipeline bereit - du kannst loslegen!")
                            time.sleep(1.1)
                            progress_placeholder.empty()

        if data_ready:
            st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
            st.markdown("### Modell-Tuning")
            n_estimators = st.slider(
                "Anzahl Bäume / Estimators",
                50,
                500,
                n_estimators,
                25,
                help="Mehr Bäume erhöhen die Modellstabilität, verlängern aber die Trainingszeit."
            )
            max_depth = st.slider(
                "Maximale Tiefe",
                3,
                20,
                max_depth,
                1,
                help="Tiefere Bäume modellieren komplexere Zusammenhänge, riskieren aber Overfitting."
            )
            learning_rate = st.slider(
                "Lernrate (Gradient Boosting)",
                0.01,
                0.3,
                learning_rate,
                0.01,
                help="Kleinere Werte trainieren stabiler, benötigen jedoch mehr Iterationen."
            )
            retrain_clicked = st.button("Modelle neu trainieren", key="retrain_btn")
            st.markdown('</div>', unsafe_allow_html=True)

            if retrain_clicked:
                progress_placeholder = show_training_progress("Trainiere Modelle mit neuen Parametern...")
                if optimizer.train_multiple_models(n_estimators, max_depth, learning_rate):
                    st.session_state.last_training = datetime.now().strftime("%d.%m.%Y %H:%M")
                    progress_placeholder.success("Modelle aktualisiert - Visualisierungen werden erneuert.")
                    time.sleep(1.0)
                    progress_placeholder.empty()
                    st.rerun()

            st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
            st.markdown("### Kundensegmentierung")
            n_clusters = st.slider(
                "Anzahl Cluster",
                2,
                8,
                n_clusters,
                1,
                help="Mehr Cluster ermöglichen feinere Segmentanalysen."
            )
            cluster_clicked = st.button("Clustering aktualisieren", key="cluster_btn")
            st.markdown('</div>', unsafe_allow_html=True)

            if cluster_clicked:
                with st.spinner("Aktualisiere Kundensegmentierung..."):
                    if optimizer.perform_clustering(n_clusters):
                        st.success("Clustering aktualisiert.")
                        st.session_state.last_training = datetime.now().strftime("%d.%m.%Y %H:%M")
                        st.rerun()

            st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
            st.markdown("### Business-Parameter")
            campaign_cost = st.slider(
                "Kampagnen-Kosten pro Kunde (&euro;)",
                1.0,
                25.0,
                campaign_cost,
                0.5,
                help="Budget je kontaktierter Person."
            )
            revenue_per_conversion = st.slider(
                "Umsatz pro Conversion (&euro;)",
                50.0,
                1000.0,
                revenue_per_conversion,
                10.0,
                help="Durchschnittlicher Beitrag einer erfolgreichen Conversion."
            )
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
            st.markdown("### Live-Filter")
            st.markdown("<small>Alle Dashboards reagieren sofort auf diese Filter.</small>", unsafe_allow_html=True)
            age_range = st.slider("Altersbereich", 18, 100, age_range)
            income_range = st.slider("Einkommensbereich (&euro;)", 0, 150000, income_range, 5000)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
            st.markdown("### Erste Schritte")
            st.markdown(
                "<small>Lade eine CSV-Datei hoch oder verwende den vorinstallierten Datensatz.</small>",
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

    render_hero_section(optimizer, data_ready)

    if not data_ready:
        render_empty_state()
        return

    if not getattr(optimizer, 'models', None):
        with st.spinner("Initialisiere Analyse-Pipeline..."):
            optimizer.train_multiple_models()
            optimizer.perform_clustering()
            st.session_state.last_training = datetime.now().strftime("%d.%m.%Y %H:%M")

    st.markdown('<div class="section-header">Key Performance Indicators</div>', unsafe_allow_html=True)

    df = optimizer.df
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        render_metric_card("Gesamtkunden", format_number(len(df)))
    with col2:
        render_metric_card("Durchschn. Conversion Rate", format_percentage(df['Conversion_Wahrscheinlichkeit'].mean()))
    with col3:
        render_metric_card(
            "High Potential",
            format_number(len(df[df['Conversion_Wahrscheinlichkeit'] > 0.7])),
            "Conversion > 70%"
        )
    with col4:
        render_metric_card("Durchschn. Umsatz pro Kunde", format_currency(df['Gesamtausgaben'].mean()))
    with col5:
        render_metric_card(
            "Kundensegmente",
            format_number(df['Segment_Label'].nunique() if 'Segment_Label' in df.columns else 0)
        )

    tab1, tab2, tab3, tab4 = st.tabs([
        "Analyse-Dashboard",
        "Interaktiver ROI",
        "Modell-Vergleich",
        "Live Analytics"
    ])

    with tab1:
        st.markdown('<div class="section-header">Analyse-Dashboard</div>', unsafe_allow_html=True)

        if getattr(optimizer, 'model_scores', None):
            st.markdown("#### Modell-Performance (Live)")
            cols = st.columns(len(optimizer.model_scores))
            for i, (model_name, scores) in enumerate(optimizer.model_scores.items()):
                with cols[i]:
                    st.markdown(
                        f"""
                        <div class="model-card">
                            <h4>{model_name}</h4>
                            <p><strong>CV Score:</strong> {scores['cv_mean']:.3f} &plusmn; {scores['cv_std']:.3f}</p>
                            <p><strong>Accuracy:</strong> {scores['accuracy']:.3f}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            best_model_name = max(optimizer.model_scores.keys(), key=lambda x: optimizer.model_scores[x]['cv_mean'])
            best_model = optimizer.models[best_model_name]

            if hasattr(best_model, 'feature_importances_'):
                feature_names = ['Alter', 'Bildung', 'Familienstand', 'Einkommen',
                                 'Haushaltsgröße', 'Hat_Kinder', 'Gesamtausgaben', 'Premium_Anteil',
                                 'Gesamtkäufe', 'Ø_Kaufwert', 'Engagement', 'Loyalität']

                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': best_model.feature_importances_
                }).sort_values('Importance', ascending=True)

                fig_importance = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title=f"Feature Importance - {best_model_name}",
                    color='Importance',
                    color_continuous_scale='viridis'
                )
                fig_importance.update_layout(height=500)
                st.plotly_chart(fig_importance, use_container_width=True)

    with tab2:
        st.markdown('<div class="section-header">Interaktiver ROI-Optimizer</div>', unsafe_allow_html=True)
        st.markdown("*ROI-Berechnungen reagieren live auf Sidebar-Parameter*", unsafe_allow_html=False)

        if data_ready:
            roi_df = optimizer.calculate_dynamic_roi(campaign_cost, revenue_per_conversion, age_range, income_range)
        else:
            roi_df = pd.DataFrame({
                'Szenario': ['Alle Kunden', 'Top 75%', 'Top 50%', 'Top 25%', 'Top 10%'],
                'Anzahl_Kunden': [2240, 1680, 1120, 560, 224],
                'Erwartete_Conversions': [340.8, 285.6, 201.6, 126.0, 67.2],
                'Conversion_Rate': [15.2, 17.0, 18.0, 22.5, 30.0],
                'ROI': [156.4, 189.2, 215.7, 275.0, 400.0],
                'Erwarteter_Gewinn': [47520, 42840, 30240, 19440, 10080]
            })

        if not roi_df.empty:
            best_idx = roi_df['ROI'].idxmax()
            best_scenario = roi_df.iloc[best_idx]

            st.markdown(
                f"""
                <div class="success-card">
                    <h3>Optimale Strategie: {best_scenario['Szenario']}</h3>
                    <p><strong>ROI:</strong> {best_scenario['ROI']:.1f}% &nbsp;|&nbsp;
                       <strong>Gewinn:</strong> {format_currency(best_scenario['Erwarteter_Gewinn'])} &nbsp;|&nbsp;
                       <strong>Kunden:</strong> {format_number(best_scenario['Anzahl_Kunden'])}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns(2)
            with col1:
                fig_roi = px.bar(
                    roi_df,
                    x='Szenario',
                    y='ROI',
                    title="Live ROI-Analyse",
                    color='ROI',
                    color_continuous_scale='RdYlGn'
                )
                fig_roi.update_layout(height=400)
                st.plotly_chart(fig_roi, use_container_width=True)

            with col2:
                fig_customers = px.scatter(
                    roi_df,
                    x='Anzahl_Kunden',
                    y='Erwarteter_Gewinn',
                    size='ROI',
                    color='Conversion_Rate',
                    title="Kunden vs. Gewinn (Bubble = ROI)",
                    color_continuous_scale='viridis'
                )
                fig_customers.update_layout(height=400)
                st.plotly_chart(fig_customers, use_container_width=True)

            st.markdown("#### Detaillierte ROI-Analyse")
            st.dataframe(roi_df, use_container_width=True)

    with tab3:
        st.markdown('<div class="section-header">Interaktiver Modell-Vergleich</div>', unsafe_allow_html=True)
        st.markdown("*Nutze die Sidebar, um Parameter zu verändern und Resultate zu beobachten*", unsafe_allow_html=False)

        model_comparison = []
        if getattr(optimizer, 'models', None):
            # Erstelle JSON-Report fuer Download
            report_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'dataset': {
                    'n_samples': len(optimizer.df),
                    'n_features': len(['Alter', 'Bildungsniveau_encoded', 'Familienstand_encoded', 'Einkommen',
                                      'Haushaltsgröße', 'Hat_Kinder', 'Gesamtausgaben', 'Premium_Anteil',
                                      'Gesamtkäufe', 'Durchschnittlicher_Kaufwert', 'Kampagnen_Engagement', 
                                      'Loyalitäts_Score'])
                },
                'models': {}
            }
            
            # Finde bestes Modell
            best_model_name = max(optimizer.model_scores.keys(), key=lambda x: optimizer.model_scores[x]['cv_mean'])
            report_data['best_model'] = best_model_name
            
            # Sammle Modell-Metriken
            for name, scores in optimizer.model_scores.items():
                report_data['models'][name] = {
                    'accuracy': float(scores['accuracy']),
                    'cv_mean': float(scores['cv_mean']),
                    'cv_std': float(scores['cv_std'])
                }
            
            # Download-Buttons nebeneinander
            json_str = json.dumps(report_data, indent=2, ensure_ascii=False)
            
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button(
                    label="ML-Report herunterladen (JSON)",
                    data=json_str,
                    file_name=f"ml_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        if getattr(optimizer, 'models', None):
            for name, scores in optimizer.model_scores.items():
                model_comparison.append({
                    'Modell': name,
                    'CV_Score': scores['cv_mean'],
                    'CV_Std': scores['cv_std'],
                    'Accuracy': scores['accuracy']
                })

            comparison_df = pd.DataFrame(model_comparison)

            col1, col2 = st.columns(2)
            with col1:
                fig_comparison = px.bar(
                    comparison_df,
                    x='Modell',
                    y='CV_Score',
                    error_y='CV_Std',
                    title="Modell-Performance Vergleich",
                    color='CV_Score',
                    color_continuous_scale='viridis'
                )
                fig_comparison.update_layout(height=400)
                st.plotly_chart(fig_comparison, use_container_width=True)

            with col2:
                fig_accuracy = px.scatter(
                    comparison_df,
                    x='CV_Score',
                    y='Accuracy',
                    size='CV_Std',
                    color='Modell',
                    title="CV Score vs. Accuracy",
                    size_max=20
                )
                fig_accuracy.update_layout(height=400)
                st.plotly_chart(fig_accuracy, use_container_width=True)

            st.markdown("#### Parameter-Impact Simulation")
            param_ranges = {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [5, 10, 15, 20]
            }

            simulation_results = []
            for n_est in param_ranges['n_estimators']:
                for depth in param_ranges['max_depth']:
                    simulated_score = 0.75 + (n_est / 1000) + (depth / 100) + np.random.normal(0, 0.02)
                    simulation_results.append({
                        'n_estimators': n_est,
                        'max_depth': depth,
                        'predicted_score': min(0.95, max(0.6, simulated_score))
                    })

            sim_df = pd.DataFrame(simulation_results)
            fig_heatmap = px.density_heatmap(
                sim_df,
                x='n_estimators',
                y='max_depth',
                z='predicted_score',
                title="Parameter-Impact Heatmap (Simuliert)"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # CSV-Download fuer historisches Tracking
            csv_data = comparison_df.copy()
            csv_data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            csv_data['is_best'] = csv_data['Modell'] == best_model_name
            csv_str = csv_data.to_csv(index=False)
            
            with dl_col2:
                st.download_button(
                    label="Modell-Vergleich herunterladen (CSV)",
                    data=csv_str,
                    file_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            dummy_comparison = pd.DataFrame({
                'Modell': ['Random Forest', 'Gradient Boosting', 'Logistic Regression'],
                'CV_Score': [0.85, 0.82, 0.78],
                'CV_Std': [0.03, 0.04, 0.05],
                'Accuracy': [0.87, 0.84, 0.80]
            })

            col1, col2 = st.columns(2)
            with col1:
                fig_dummy = px.bar(
                    dummy_comparison,
                    x='Modell',
                    y='CV_Score',
                    title="Modell-Performance (Demo)",
                    color='CV_Score',
                    color_continuous_scale='viridis'
                )
                fig_dummy.update_layout(height=400)
                st.plotly_chart(fig_dummy, use_container_width=True)

            with col2:
                fig_dummy2 = px.scatter(
                    dummy_comparison,
                    x='CV_Score',
                    y='Accuracy',
                    color='Modell',
                    title="CV Score vs. Accuracy (Demo)",
                    size_max=20
                )
                fig_dummy2.update_layout(height=400)
                st.plotly_chart(fig_dummy2, use_container_width=True)

            st.info("Demo-Daten angezeigt. Trainiere Modelle für echte Ergebnisse.")

    with tab4:
        st.markdown('<div class="section-header">Live Analytics Dashboard</div>', unsafe_allow_html=True)
        st.markdown("*Alle Visualisierungen reagieren live auf die Filter in der Sidebar*", unsafe_allow_html=False)

        if data_ready:
            filtered_df = optimizer.df[
                (optimizer.df['Alter'] >= age_range[0]) &
                (optimizer.df['Alter'] <= age_range[1]) &
                (optimizer.df['Einkommen'] >= income_range[0]) &
                (optimizer.df['Einkommen'] <= income_range[1])
            ]
        else:
            np.random.seed(42)
            filtered_df = pd.DataFrame({
                'Alter': np.random.randint(25, 75, 1000),
                'Einkommen': np.random.randint(20000, 100000, 1000),
                'Gesamtausgaben': np.random.randint(100, 2000, 1000),
                'Conversion_Wahrscheinlichkeit': np.random.beta(2, 5, 1000),
                'Segment_Label': np.random.choice(['Segment 1', 'Segment 2', 'Segment 3', 'Segment 4'], 1000)
            })

        col1, col2 = st.columns(2)
        with col1:
            fig_dist = px.histogram(
                filtered_df,
                x='Conversion_Wahrscheinlichkeit',
                nbins=30,
                title="Live Conversion-Verteilung",
                color_discrete_sequence=['#38bdf8']
            )
            fig_dist.update_layout(height=400)
            st.plotly_chart(fig_dist, use_container_width=True)

        with col2:
            if 'Segment_Label' in filtered_df.columns:
                segment_perf = filtered_df.groupby('Segment_Label')['Conversion_Wahrscheinlichkeit'].mean().reset_index()
                segment_mapping = {
                    'Segment 1': 'Premium-Kunden',
                    'Segment 2': 'Gelegenheitskaeufer',
                    'Segment 3': 'Loyale Stammkunden',
                    'Segment 4': 'Preisbewusste Kaeufer'
                }
                segment_perf['Segment_Name'] = segment_perf['Segment_Label'].map(segment_mapping).fillna(segment_perf['Segment_Label'])

                fig_segments = px.bar(
                    segment_perf,
                    x='Segment_Name',
                    y='Conversion_Wahrscheinlichkeit',
                    title="Segment-Performance",
                    color='Conversion_Wahrscheinlichkeit',
                    color_continuous_scale='viridis'
                )
                fig_segments.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig_segments, use_container_width=True)
            else:
                dummy_segments = pd.DataFrame({
                    'Segment': ['Premium-Kunden', 'Gelegenheitskaeufer', 'Loyale Stammkunden', 'Preisbewusste Kaeufer'],
                    'Conversion': [0.6, 0.4, 0.8, 0.3]
                })
                fig_dummy_bar = px.bar(
                    dummy_segments,
                    x='Segment',
                    y='Conversion',
                    title="Segment-Performance (Demo)",
                    color='Conversion',
                    color_continuous_scale='viridis'
                )
                fig_dummy_bar.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig_dummy_bar, use_container_width=True)

        fig_3d = go.Figure(data=go.Scatter3d(
            x=filtered_df['Alter'],
            y=filtered_df['Einkommen'],
            z=filtered_df['Gesamtausgaben'],
            mode='markers',
            marker=dict(
                size=4,
                color=filtered_df['Conversion_Wahrscheinlichkeit'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Conversion Probability")
            ),
            text=filtered_df.get('Segment_Label', 'Kunde'),
            hovertemplate="<b>%{text}</b><br>Alter: %{x}<br>Einkommen: %{y}<br>Ausgaben: %{z}<extra></extra>"
        ))

        fig_3d.update_layout(
            title="3D Kunden-Landschaft",
            scene=dict(
                xaxis_title="Alter",
                yaxis_title="Einkommen (&euro;)",
                zaxis_title="Gesamtausgaben (&euro;)"
            ),
            height=600
        )

        st.plotly_chart(fig_3d, use_container_width=True)

        st.markdown("#### Live-Statistiken")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Gefilterte Kunden", format_number(len(filtered_df)))
        with col2:
            st.metric(
                "Durchschn. Conversion (gefiltert)",
                format_percentage(filtered_df['Conversion_Wahrscheinlichkeit'].mean())
            )
        with col3:
            st.metric("Durchschn. Alter (gefiltert)", format_number(filtered_df['Alter'].mean()))
        with col4:
            st.metric(
                "Durchschn. Einkommen (gefiltert)",
                f"{format_number(filtered_df['Einkommen'].mean())} €"
            )

if __name__ == "__main__":
    main()
