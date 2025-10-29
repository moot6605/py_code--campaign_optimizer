"""
MARKETING CAMPAIGN OPTIMIZATION - INTERACTIVE WEB INTERFACE (MODULAR)
======================================================================

Modular Streamlit-based web application for interactive analysis and 
optimization of marketing campaigns with machine learning.

Uses modular architecture:
- src.data_processor: Data processing and feature engineering
- src.ml_models: Machine learning models
- src.business_analytics: ROI calculations and business intelligence
- src.visualization: Interactive visualizations
- src.config: Central configuration

Author: ML Final Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime
import logging

# Modulare Imports
from src.config import config
from src.data_processor import DataProcessor
from src.ml_models import MLModelManager
from src.business_analytics import BusinessAnalytics
from src.visualization import CampaignVisualizer

# Streamlit Konfiguration
st.set_page_config(
    page_title=config.ui.PAGE_TITLE,
    page_icon=config.ui.PAGE_ICON,
    layout=config.ui.LAYOUT,
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown(f"""
<style>
    .main-header {{
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }}
    .metric-card {{
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }}
    .success-box {{
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 1rem 0;
    }}
</style>
""", unsafe_allow_html=True)

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModularCampaignOptimizer:
    """
    Modulare Hauptklasse für die Marketing-Kampagne-Optimierung
    Nutzt die separaten Module für bessere Wartbarkeit
    """
    
    def __init__(self):
        """Initialisiert alle Module"""
        self.data_processor = DataProcessor()
        self.ml_manager = MLModelManager()
        self.business_analytics = BusinessAnalytics()
        self.visualizer = CampaignVisualizer()
        
        # Status-Tracking
        self.data_ready = False
        self.models_trained = False
        
    def load_and_process_data(self, uploaded_file=None):
        """
        Lädt und verarbeitet Daten mit der modularen Pipeline
        
        Args:
            uploaded_file: Streamlit UploadedFile object oder None
        """
        try:
            with st.spinner("Lade und verarbeite Daten..."):
                # Daten laden
                if uploaded_file is not None:
                    # Temporäre Datei für uploaded file
                    temp_path = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    with open(temp_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    
                    pipeline_result = self.data_processor.process_pipeline(temp_path)
                    
                    # Temporäre Datei löschen
                    import os
                    os.remove(temp_path)
                else:
                    pipeline_result = self.data_processor.process_pipeline()
                
                if pipeline_result is None:
                    st.error("❌ Datenverarbeitung fehlgeschlagen")
                    return False
                
                # Pipeline-Ergebnisse speichern
                self.df_processed, self.X, self.feature_names = pipeline_result
                
                # Kundensegmentierung durchführen
                self.df_segmented = self.ml_manager.perform_clustering(self.df_processed)
                
                # ML-Modelle trainieren
                y = self.df_segmented['Antwort_Letzte_Kampagne']
                self.model_results = self.ml_manager.train_classification_models(self.X, y)
                
                # Conversion-Wahrscheinlichkeiten vorhersagen
                conversion_probs = self.ml_manager.predict_conversion_probability(self.X)
                self.df_segmented['Conversion_Wahrscheinlichkeit'] = conversion_probs
                
                self.data_ready = True
                self.models_trained = True
                
                st.success("Daten erfolgreich verarbeitet und Modelle trainiert!")
                return True
                
        except Exception as e:
            st.error(f"Fehler bei der Datenverarbeitung: {str(e)}")
            logger.error(f"Datenverarbeitungsfehler: {str(e)}")
            return False
    
    def update_business_parameters(self, campaign_cost, revenue_per_conversion):
        """Aktualisiert Business-Parameter"""
        self.business_analytics.set_business_parameters(campaign_cost, revenue_per_conversion)
    
    def get_roi_scenarios(self):
        """Berechnet ROI-Szenarien"""
        if not self.data_ready:
            return None
        return self.business_analytics.calculate_roi_scenarios(self.df_segmented)
    
    def get_segment_analysis(self):
        """Führt Segment-Analyse durch"""
        if not self.data_ready:
            return None
        return self.business_analytics.segment_analysis(self.df_segmented)
    
    def get_campaign_performance(self):
        """Analysiert Kampagnen-Performance"""
        if not self.data_ready:
            return None
        return self.business_analytics.analyze_campaign_performance(self.df_segmented)
    
    def predict_single_customer(self, customer_features):
        """Vorhersage für einzelnen Kunden"""
        if not self.models_trained:
            return None
        return self.ml_manager.predict_single_customer(customer_features)

def main():
    """Hauptfunktion der modularen Streamlit-Anwendung"""
    
    # Header
    st.markdown('<h1 class="main-header">Marketing Kampagne Optimizer</h1>', unsafe_allow_html=True)
    st.markdown("*Modulare Architektur für professionelle ML-Anwendungen*")
    st.markdown("---")
    
    # Optimizer initialisieren
    if 'optimizer' not in st.session_state:
        st.session_state.optimizer = ModularCampaignOptimizer()
    
    optimizer = st.session_state.optimizer
    
    # Sidebar für Navigation und Parameter
    st.sidebar.title("Steuerung")
    
    # Daten-Upload Sektion
    st.sidebar.subheader("Daten laden")
    uploaded_file = st.sidebar.file_uploader(
        "CSV-Datei hochladen (optional)", 
        type=['csv'],
        help="Lade eine CSV-Datei mit Kundendaten hoch oder verwende die Standard-Datei"
    )
    
    if st.sidebar.button("Daten laden und verarbeiten"):
        success = optimizer.load_and_process_data(uploaded_file)
        if success:
            st.session_state.data_ready = True
    
    # Prüfe ob Daten bereit sind
    if not hasattr(st.session_state, 'data_ready') or not st.session_state.data_ready:
        st.info("Bitte zuerst Daten laden und verarbeiten (Sidebar)")
        
        # Zeige Modul-Übersicht
        st.subheader("Modulare Architektur")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Datenverarbeitung** (`src.data_processor`)
            - Automatische Datenbereinigung
            - Feature Engineering (15+ neue Features)
            - Outlier-Behandlung
            - Kategorische Encodierung
            
            **Machine Learning** (`src.ml_models`)
            - Multi-Modell Training & Evaluation
            - SMOTE für Klassenungleichgewicht
            - Hyperparameter-Tuning
            - Kundensegmentierung mit K-Means
            """)
        
        with col2:
            st.markdown("""
            **Business Analytics** (`src.business_analytics`)
            - ROI-Optimierung für verschiedene Szenarien
            - Kampagnen-Performance-Analyse
            - Customer Lifetime Value
            - Executive Summary Generation
            
            **Visualisierung** (`src.visualization`)
            - Interaktive Plotly-Dashboards
            - Executive-Level Reports
            - Segment-Performance Charts
            - Export-Funktionalitäten
            """)
        
        return
    
    # Business-Parameter Konfiguration
    st.sidebar.subheader("Business Parameter")
    campaign_cost = st.sidebar.slider(
        "Kampagnen-Kosten pro Kunde (€)", 
        min_value=config.ui.CAMPAIGN_COST_RANGE[0], 
        max_value=config.ui.CAMPAIGN_COST_RANGE[1], 
        value=config.business.DEFAULT_CAMPAIGN_COST, 
        step=config.ui.CAMPAIGN_COST_RANGE[2],
        help="Kosten für die Kontaktierung eines Kunden"
    )
    
    revenue_per_conversion = st.sidebar.slider(
        "Umsatz pro Conversion (€)", 
        min_value=config.ui.REVENUE_RANGE[0], 
        max_value=config.ui.REVENUE_RANGE[1], 
        value=config.business.DEFAULT_REVENUE_PER_CONVERSION, 
        step=config.ui.REVENUE_RANGE[2],
        help="Durchschnittlicher Umsatz bei erfolgreicher Conversion"
    )
    
    # Business-Parameter aktualisieren
    optimizer.update_business_parameters(campaign_cost, revenue_per_conversion)
    
    # Navigation Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Dashboard", "ROI Optimizer", "Kundensegmente", 
        "Einzelkunden-Scoring", "Batch-Analyse"
    ])
    
    # Tab 1: Dashboard
    with tab1:
        st.header("Kampagnen-Dashboard")
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Gesamtkunden", 
                f"{len(optimizer.df_segmented):,}",
                help="Anzahl Kunden in der Datenbank"
            )
        
        with col2:
            avg_conversion_prob = optimizer.df_segmented['Conversion_Wahrscheinlichkeit'].mean()
            st.metric(
                "Ø Conversion-Wahrscheinlichkeit", 
                f"{avg_conversion_prob:.1%}",
                help="Durchschnittliche Wahrscheinlichkeit für Kampagnen-Response"
            )
        
        with col3:
            high_potential = len(optimizer.df_segmented[optimizer.df_segmented['Conversion_Wahrscheinlichkeit'] > 0.7])
            st.metric(
                "High-Potential Kunden", 
                f"{high_potential:,}",
                help="Kunden mit >70% Conversion-Wahrscheinlichkeit"
            )
        
        with col4:
            avg_customer_value = optimizer.df_segmented['Gesamtausgaben'].mean()
            st.metric(
                "Ø Kundenwert", 
                f"{avg_customer_value:.0f}€",
                help="Durchschnittliche Gesamtausgaben pro Kunde"
            )
        
        # Visualisierungen mit modularem Visualizer
        col1, col2 = st.columns(2)
        
        with col1:
            fig_hist = optimizer.visualizer.create_conversion_distribution_plot(optimizer.df_segmented)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            fig_pie = optimizer.visualizer.create_segment_pie_chart(optimizer.df_segmented)
            st.plotly_chart(fig_pie, use_container_width=True)
    
    # Tab 2: ROI Optimizer
    with tab2:
        st.header("ROI-Optimierung")
        
        # ROI-Szenarien mit modularer Business Analytics
        roi_df = optimizer.get_roi_scenarios()
        
        if roi_df is not None:
            st.subheader("ROI-Szenarien Vergleich")
            
            # Formatierte Tabelle
            formatted_roi = roi_df.copy()
            formatted_roi['Kosten'] = formatted_roi['Kosten'].apply(lambda x: f"{x:,.2f}€")
            formatted_roi['Erwarteter_Umsatz'] = formatted_roi['Erwarteter_Umsatz'].apply(lambda x: f"{x:,.2f}€")
            formatted_roi['Erwarteter_Gewinn'] = formatted_roi['Erwarteter_Gewinn'].apply(lambda x: f"{x:,.2f}€")
            formatted_roi['ROI'] = formatted_roi['ROI'].apply(lambda x: f"{x:.1f}%")
            formatted_roi['Conversion_Rate'] = formatted_roi['Conversion_Rate'].apply(lambda x: f"{x:.1f}%")
            
            st.dataframe(formatted_roi, use_container_width=True)
            
            # Beste Strategie
            best_roi_idx = roi_df['ROI'].idxmax()
            best_scenario = roi_df.iloc[best_roi_idx]
            
            st.success(f"""
            **Beste Strategie: {best_scenario['Szenario']}**
            - ROI: {best_scenario['ROI']:.1f}%
            - Erwarteter Gewinn: {best_scenario['Erwarteter_Gewinn']:,.2f}€
            - Zu kontaktierende Kunden: {best_scenario['Anzahl_Kunden']:,}
            """)
            
            # Visualisierungen
            col1, col2 = st.columns(2)
            
            with col1:
                fig_roi = optimizer.visualizer.create_roi_comparison_chart(roi_df)
                st.plotly_chart(fig_roi, use_container_width=True)
            
            with col2:
                fig_profit = optimizer.visualizer.create_profit_comparison_chart(roi_df)
                st.plotly_chart(fig_profit, use_container_width=True)
    
    # Tab 3: Kundensegmente
    with tab3:
        st.header("Kundensegment-Analyse")
        
        segment_stats = optimizer.get_segment_analysis()
        
        if segment_stats is not None:
            st.subheader("Segment-Übersicht")
            st.dataframe(segment_stats, use_container_width=True)
            
            # Segment-Performance Chart
            fig_segment = optimizer.visualizer.create_segment_performance_chart(segment_stats)
            st.plotly_chart(fig_segment, use_container_width=True)
            
            # Segment-Details
            selected_segment = st.selectbox("Segment für Details auswählen:", 
                                          optimizer.df_segmented['Segment_Label'].unique())
            
            if selected_segment:
                segment_data = optimizer.df_segmented[optimizer.df_segmented['Segment_Label'] == selected_segment]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Kunden im Segment", len(segment_data))
                with col2:
                    st.metric("Ø Conversion-Wahrscheinlichkeit", 
                             f"{segment_data['Conversion_Wahrscheinlichkeit'].mean():.1%}")
                with col3:
                    st.metric("Ø Kundenwert", f"{segment_data['Gesamtausgaben'].mean():.0f}€")
    
    # Tab 4: Einzelkunden-Scoring
    with tab4:
        st.header("Einzelkunden-Scoring")
        
        st.subheader("Neuen Kunden bewerten")
        
        # Input-Formular
        with st.form("customer_scoring"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                alter = st.number_input("Alter", min_value=18, max_value=100, value=45)
                einkommen = st.number_input("Jahreseinkommen (€)", min_value=10000, max_value=200000, value=50000)
                bildung = st.selectbox("Bildungsniveau", 
                                     optimizer.data_processor.encoders['bildung'].classes_)
            
            with col2:
                familienstand = st.selectbox("Familienstand", 
                                           optimizer.data_processor.encoders['familienstand'].classes_)
                kinder = st.number_input("Kinder zu Hause", min_value=0, max_value=10, value=0)
                teenager = st.number_input("Teenager zu Hause", min_value=0, max_value=10, value=0)
            
            with col3:
                gesamtausgaben = st.number_input("Bisherige Gesamtausgaben (€)", min_value=0, max_value=10000, value=500)
                web_besuche = st.number_input("Web-Besuche pro Monat", min_value=0, max_value=50, value=5)
                letzter_kauf = st.number_input("Tage seit letztem Kauf", min_value=0, max_value=365, value=30)
            
            submitted = st.form_submit_button("Kunden bewerten")
            
            if submitted:
                try:
                    # Feature-Dictionary erstellen
                    customer_features = {
                        'Alter': alter,
                        'Bildungsniveau_encoded': optimizer.data_processor.encoders['bildung'].transform([bildung])[0],
                        'Familienstand_encoded': optimizer.data_processor.encoders['familienstand'].transform([familienstand])[0],
                        'Einkommen': einkommen,
                        'Haushaltsgröße': kinder + teenager,
                        'Hat_Kinder': 1 if (kinder + teenager) > 0 else 0,
                        'Gesamtausgaben': gesamtausgaben,
                        'Premium_Anteil': 0.3,  # Annahme
                        'Gesamtkäufe': 5,  # Annahme
                        'Durchschnittlicher_Kaufwert': gesamtausgaben / 6,
                        'Kampagnen_Engagement': 2,  # Annahme
                        'Loyalitäts_Score': 1.5,  # Annahme
                        'Letzter_Kauf_Tage': letzter_kauf,
                        'Anzahl_WebBesuche_Monat': web_besuche,
                        'Kundensegment': 1  # Annahme
                    }
                    
                    # Vorhersage mit modularem ML-Manager
                    result = optimizer.predict_single_customer(customer_features)
                    
                    if result:
                        st.success("Bewertung abgeschlossen!")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Conversion-Wahrscheinlichkeit", f"{result['probability']:.1%}")
                        with col2:
                            st.metric("Risiko-Level", result['risk_level'])
                        with col3:
                            st.metric("Empfehlung", result['recommendation'])
                        
                        # ROI für diesen Kunden
                        expected_revenue = result['probability'] * revenue_per_conversion
                        expected_profit = expected_revenue - campaign_cost
                        customer_roi = (expected_profit / campaign_cost * 100) if campaign_cost > 0 else 0
                        
                        st.info(f"Erwarteter ROI für diesen Kunden: {customer_roi:.1f}%")
                    
                except Exception as e:
                    st.error(f"Fehler bei der Bewertung: {str(e)}")
    
    # Tab 5: Batch-Analyse
    with tab5:
        st.header("Batch-Analyse")
        
        st.subheader("Top-Kunden Analyse")
        
        # Filter-Optionen
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_probability = st.slider(
                "Minimale Conversion-Wahrscheinlichkeit", 
                config.ui.PROBABILITY_RANGE[0], 
                config.ui.PROBABILITY_RANGE[1], 
                0.5, 
                config.ui.PROBABILITY_RANGE[2]
            )
        
        with col2:
            selected_segments = st.multiselect(
                "Kundensegmente", 
                optimizer.df_segmented['Segment_Label'].unique(),
                default=optimizer.df_segmented['Segment_Label'].unique()
            )
        
        with col3:
            top_n = st.number_input("Top N Kunden anzeigen", min_value=10, max_value=500, value=50)
        
        # Gefilterte Daten
        filtered_df = optimizer.df_segmented[
            (optimizer.df_segmented['Conversion_Wahrscheinlichkeit'] >= min_probability) &
            (optimizer.df_segmented['Segment_Label'].isin(selected_segments))
        ].nlargest(top_n, 'Conversion_Wahrscheinlichkeit')
        
        if len(filtered_df) > 0:
            # Zusammenfassung
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Gefilterte Kunden", len(filtered_df))
            with col2:
                expected_conversions = filtered_df['Conversion_Wahrscheinlichkeit'].sum()
                st.metric("Erwartete Conversions", f"{expected_conversions:.1f}")
            with col3:
                total_cost = len(filtered_df) * campaign_cost
                st.metric("Gesamtkosten", f"{total_cost:,.2f}€")
            with col4:
                expected_revenue = expected_conversions * revenue_per_conversion
                roi = ((expected_revenue - total_cost) / total_cost * 100) if total_cost > 0 else 0
                st.metric("ROI", f"{roi:.1f}%")
            
            # Top-Kunden Tabelle
            st.subheader("Top-Kunden Liste")
            
            display_columns = ['ID', 'Segment_Label', 'Conversion_Wahrscheinlichkeit', 
                             'Gesamtausgaben', 'Alter', 'Einkommen', 'Kampagnen_Engagement']
            
            display_df = filtered_df[display_columns].copy()
            display_df['Conversion_Wahrscheinlichkeit'] = display_df['Conversion_Wahrscheinlichkeit'].apply(lambda x: f"{x:.1%}")
            display_df['Gesamtausgaben'] = display_df['Gesamtausgaben'].apply(lambda x: f"{x:.0f}€")
            display_df['Einkommen'] = display_df['Einkommen'].apply(lambda x: f"{x:,.0f}€")
            
            st.dataframe(display_df, use_container_width=True)
            
            # Download-Button
            csv_buffer = io.StringIO()
            filtered_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="Top-Kunden als CSV herunterladen",
                data=csv_data,
                file_name=f"top_kunden_{datetime.now().strftime(config.ui.DATE_FORMAT)}.csv",
                mime="text/csv"
            )
        
        else:
            st.warning("Keine Kunden entsprechen den gewählten Filterkriterien.")

if __name__ == "__main__":
    main()