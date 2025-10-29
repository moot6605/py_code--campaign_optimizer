"""
VISUALISIERUNG - MARKETING KAMPAGNE OPTIMIZER
============================================

Modul für alle Visualisierungs-Funktionen:
- Interaktive Plotly-Dashboards
- Business-Metriken Visualisierungen
- Kampagnen-Performance Charts
- Segment-Analyse Plots
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import config

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CampaignVisualizer:
    """
    Hauptklasse für alle Visualisierungen
    """

    def __init__(self):
        """Initialisiert den Visualizer"""
        self.color_palette = {
            'primary': config.ui.PRIMARY_COLOR,
            'success': config.ui.SUCCESS_COLOR,
            'warning': config.ui.WARNING_COLOR,
            'error': config.ui.ERROR_COLOR
        }

        # Plotly Template konfigurieren
        self.template = "plotly_white"

    def create_conversion_distribution_plot(self, df: pd.DataFrame) -> go.Figure:
        """
        Erstellt Histogramm der Conversion-Wahrscheinlichkeiten
        
        Args:
            df: DataFrame mit Conversion-Wahrscheinlichkeiten
            
        Returns:
            Plotly Figure
        """
        fig = px.histogram(
            df,
            x='Conversion_Wahrscheinlichkeit',
            nbins=30,
            title="Verteilung der Conversion-Wahrscheinlichkeiten",
            labels={
                'Conversion_Wahrscheinlichkeit': 'Conversion-Wahrscheinlichkeit',
                'count': 'Anzahl Kunden'
            },
            template=self.template
        )

        # Durchschnitt als vertikale Linie hinzufügen
        avg_prob = df['Conversion_Wahrscheinlichkeit'].mean()
        fig.add_vline(
            x=avg_prob,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Durchschnitt: {avg_prob:.1%}"
        )

        fig.update_layout(
            showlegend=False,
            height=400
        )

        return fig

    def create_segment_pie_chart(self, df: pd.DataFrame) -> go.Figure:
        """
        Erstellt Pie Chart der Kundensegment-Verteilung
        
        Args:
            df: DataFrame mit Segment-Informationen
            
        Returns:
            Plotly Figure
        """
        if 'Segment_Label' not in df.columns:
            # Fallback: Erstelle dummy Segmente
            segment_counts = pd.Series([len(df)], index=['Alle Kunden'])
        else:
            segment_counts = df['Segment_Label'].value_counts()

        fig = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title="Kundensegment-Verteilung",
            template=self.template
        )

        fig.update_traces(
            textposition='inside',
            textinfo='percent+label'
        )

        fig.update_layout(height=400)

        return fig

    def create_roi_comparison_chart(self, roi_df: pd.DataFrame) -> go.Figure:
        """
        Erstellt ROI-Vergleichschart
        
        Args:
            roi_df: DataFrame mit ROI-Szenarien
            
        Returns:
            Plotly Figure
        """
        # Farben basierend auf ROI-Werten
        ['green' if roi > 0 else 'red' for roi in roi_df['ROI']]

        fig = px.bar(
            roi_df,
            x='Szenario',
            y='ROI',
            title="ROI-Vergleich nach Targeting-Strategie",
            labels={'ROI': 'ROI (%)', 'Szenario': 'Targeting-Strategie'},
            template=self.template,
            color='ROI',
            color_continuous_scale='RdYlGn'
        )

        # Null-Linie hinzufügen
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)

        fig.update_layout(
            showlegend=False,
            height=400
        )

        return fig

    def create_profit_comparison_chart(self, roi_df: pd.DataFrame) -> go.Figure:
        """
        Erstellt Gewinn-Vergleichschart
        
        Args:
            roi_df: DataFrame mit ROI-Szenarien
            
        Returns:
            Plotly Figure
        """
        fig = px.bar(
            roi_df,
            x='Szenario',
            y='Erwarteter_Gewinn',
            title="Erwarteter Gewinn nach Targeting-Strategie",
            labels={'Erwarteter_Gewinn': 'Erwarteter Gewinn (€)', 'Szenario': 'Targeting-Strategie'},
            template=self.template,
            color='Erwarteter_Gewinn',
            color_continuous_scale='Blues'
        )

        fig.update_layout(
            showlegend=False,
            height=400
        )

        return fig

    def create_segment_performance_chart(self, segment_stats: pd.DataFrame) -> go.Figure:
        """
        Erstellt Segment-Performance Chart
        
        Args:
            segment_stats: DataFrame mit Segment-Statistiken
            
        Returns:
            Plotly Figure
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Conversion-Wahrscheinlichkeit',
                'Durchschnittliche Ausgaben',
                'Kampagnen-Engagement',
                'Kundenzahl'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        segments = segment_stats.index

        # Conversion-Wahrscheinlichkeit
        if 'Ø_Conversion_Prob' in segment_stats.columns:
            fig.add_trace(
                go.Bar(x=segments, y=segment_stats['Ø_Conversion_Prob'], name='Conversion Prob.'),
                row=1, col=1
            )

        # Durchschnittliche Ausgaben
        if 'Ø_Ausgaben' in segment_stats.columns:
            fig.add_trace(
                go.Bar(x=segments, y=segment_stats['Ø_Ausgaben'], name='Ausgaben'),
                row=1, col=2
            )

        # Kampagnen-Engagement
        if 'Ø_Engagement' in segment_stats.columns:
            fig.add_trace(
                go.Bar(x=segments, y=segment_stats['Ø_Engagement'], name='Engagement'),
                row=2, col=1
            )

        # Kundenzahl
        if 'Anzahl_Kunden' in segment_stats.columns:
            fig.add_trace(
                go.Bar(x=segments, y=segment_stats['Anzahl_Kunden'], name='Anzahl'),
                row=2, col=2
            )

        fig.update_layout(
            height=600,
            showlegend=False,
            title_text="Segment-Performance Übersicht",
            template=self.template
        )

        return fig

    def create_campaign_performance_chart(self, campaign_df: pd.DataFrame) -> go.Figure:
        """
        Erstellt Kampagnen-Performance Chart
        
        Args:
            campaign_df: DataFrame mit Kampagnen-Performance
            
        Returns:
            Plotly Figure
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Akzeptanzraten', 'Theoretischer ROI'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )

        campaigns = campaign_df.index

        # Akzeptanzraten
        if 'Akzeptanzrate (%)' in campaign_df.columns:
            fig.add_trace(
                go.Bar(
                    x=campaigns,
                    y=campaign_df['Akzeptanzrate (%)'],
                    name='Akzeptanzrate',
                    marker_color=self.color_palette['primary']
                ),
                row=1, col=1
            )

        # ROI
        if 'Theoretischer_ROI (%)' in campaign_df.columns:
            colors = ['green' if roi > 0 else 'red' for roi in campaign_df['Theoretischer_ROI (%)']]
            fig.add_trace(
                go.Bar(
                    x=campaigns,
                    y=campaign_df['Theoretischer_ROI (%)'],
                    name='ROI',
                    marker_color=colors
                ),
                row=1, col=2
            )

        fig.update_layout(
            height=400,
            showlegend=False,
            title_text="Kampagnen-Performance Vergleich",
            template=self.template
        )

        return fig

    def create_correlation_heatmap(self, df: pd.DataFrame, features: List[str] = None) -> go.Figure:
        """
        Erstellt Korrelations-Heatmap
        
        Args:
            df: DataFrame mit numerischen Daten
            features: Liste der zu analysierenden Features (optional)
            
        Returns:
            Plotly Figure
        """
        if features is None:
            # Wähle numerische Spalten
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Begrenze auf wichtigste Features
            features = numeric_cols[:15] if len(numeric_cols) > 15 else numeric_cols

        # Berechne Korrelationsmatrix
        corr_matrix = df[features].corr()

        fig = px.imshow(
            corr_matrix,
            title="Korrelationsmatrix wichtiger Features",
            color_continuous_scale='RdBu_r',
            aspect="auto",
            template=self.template
        )

        fig.update_layout(
            width=800,
            height=600
        )

        return fig

    def create_feature_importance_chart(self, importance_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
        """
        Erstellt Feature Importance Chart
        
        Args:
            importance_df: DataFrame mit Feature Importance
            top_n: Anzahl der Top-Features zu zeigen
            
        Returns:
            Plotly Figure
        """
        top_features = importance_df.head(top_n)

        fig = px.bar(
            top_features,
            x='Importance',
            y='Feature',
            orientation='h',
            title=f"Top {top_n} Feature Importance",
            labels={'Importance': 'Wichtigkeit', 'Feature': 'Feature'},
            template=self.template
        )

        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=400
        )

        return fig

    def create_customer_value_distribution(self, df: pd.DataFrame) -> go.Figure:
        """
        Erstellt Kundenwert-Verteilungsplot
        
        Args:
            df: DataFrame mit Kundenwert-Daten
            
        Returns:
            Plotly Figure
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Gesamtausgaben Verteilung', 'Ausgaben vs. Conversion-Wahrscheinlichkeit']
        )

        # Histogramm der Gesamtausgaben
        fig.add_trace(
            go.Histogram(
                x=df['Gesamtausgaben'],
                nbinsx=30,
                name='Gesamtausgaben'
            ),
            row=1, col=1
        )

        # Scatter Plot: Ausgaben vs. Conversion-Wahrscheinlichkeit
        fig.add_trace(
            go.Scatter(
                x=df['Gesamtausgaben'],
                y=df['Conversion_Wahrscheinlichkeit'],
                mode='markers',
                name='Kunden',
                opacity=0.6
            ),
            row=1, col=2
        )

        fig.update_layout(
            height=400,
            title_text="Kundenwert-Analyse",
            template=self.template
        )

        return fig

    def create_age_income_analysis(self, df: pd.DataFrame) -> go.Figure:
        """
        Erstellt Alter-Einkommen Analyse
        
        Args:
            df: DataFrame mit Alter und Einkommen
            
        Returns:
            Plotly Figure
        """
        fig = px.scatter(
            df,
            x='Alter',
            y='Einkommen',
            size='Gesamtausgaben',
            color='Conversion_Wahrscheinlichkeit',
            title="Alter vs. Einkommen (Größe = Ausgaben, Farbe = Conversion-Wahrscheinlichkeit)",
            labels={
                'Alter': 'Alter (Jahre)',
                'Einkommen': 'Jahreseinkommen (€)',
                'Conversion_Wahrscheinlichkeit': 'Conversion-Wahrscheinlichkeit'
            },
            template=self.template,
            color_continuous_scale='Viridis'
        )

        fig.update_layout(height=500)

        return fig

    def create_channel_preference_analysis(self, df: pd.DataFrame) -> go.Figure:
        """
        Erstellt Kanal-Präferenz Analyse
        
        Args:
            df: DataFrame mit Kanal-Daten
            
        Returns:
            Plotly Figure
        """
        # Berechne Kanal-Anteile
        channel_data = df[config.data.PURCHASE_CHANNELS].sum()

        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "pie"}, {"type": "bar"}]],
            subplot_titles=['Kanal-Verteilung (Gesamt)', 'Durchschnittliche Käufe pro Kanal']
        )

        # Pie Chart für Kanal-Verteilung
        fig.add_trace(
            go.Pie(
                labels=['Web', 'Katalog', 'Laden'],
                values=channel_data.values,
                name="Kanäle"
            ),
            row=1, col=1
        )

        # Bar Chart für durchschnittliche Käufe
        avg_purchases = df[config.data.PURCHASE_CHANNELS].mean()
        fig.add_trace(
            go.Bar(
                x=['Web', 'Katalog', 'Laden'],
                y=avg_purchases.values,
                name="Ø Käufe"
            ),
            row=1, col=2
        )

        fig.update_layout(
            height=400,
            title_text="Kanal-Präferenz Analyse",
            template=self.template
        )

        return fig

    def create_executive_dashboard(self, df: pd.DataFrame, roi_df: pd.DataFrame,
                                 segment_stats: pd.DataFrame = None) -> Dict[str, go.Figure]:
        """
        Erstellt Executive Dashboard mit allen wichtigen Visualisierungen
        
        Args:
            df: Haupt-DataFrame
            roi_df: ROI-Szenarien DataFrame
            segment_stats: Segment-Statistiken (optional)
            
        Returns:
            Dictionary mit allen Dashboard-Plots
        """
        dashboard_plots = {}

        try:
            # Conversion-Verteilung
            dashboard_plots['conversion_distribution'] = self.create_conversion_distribution_plot(df)

            # ROI-Vergleich
            dashboard_plots['roi_comparison'] = self.create_roi_comparison_chart(roi_df)

            # Gewinn-Vergleich
            dashboard_plots['profit_comparison'] = self.create_profit_comparison_chart(roi_df)

            # Segment-Verteilung
            dashboard_plots['segment_pie'] = self.create_segment_pie_chart(df)

            # Kundenwert-Analyse
            dashboard_plots['customer_value'] = self.create_customer_value_distribution(df)

            # Alter-Einkommen Analyse
            if 'Alter' in df.columns and 'Einkommen' in df.columns:
                dashboard_plots['age_income'] = self.create_age_income_analysis(df)

            # Kanal-Analyse
            dashboard_plots['channel_analysis'] = self.create_channel_preference_analysis(df)

            # Segment-Performance (falls verfügbar)
            if segment_stats is not None:
                dashboard_plots['segment_performance'] = self.create_segment_performance_chart(segment_stats)

            logger.info(f"Executive Dashboard mit {len(dashboard_plots)} Plots erstellt")

        except Exception as e:
            logger.error(f"Fehler beim Erstellen des Executive Dashboards: {str(e)}")

        return dashboard_plots

    def save_plots_as_html(self, plots: Dict[str, go.Figure], output_dir: str = "plots/") -> List[str]:
        """
        Speichert Plots als HTML-Dateien
        
        Args:
            plots: Dictionary mit Plotly Figures
            output_dir: Ausgabe-Verzeichnis
            
        Returns:
            Liste der gespeicherten Dateipfade
        """
        import os
        from datetime import datetime

        # Erstelle Ausgabe-Verzeichnis
        os.makedirs(output_dir, exist_ok=True)

        saved_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for plot_name, fig in plots.items():
            try:
                filename = f"{plot_name}_{timestamp}.html"
                filepath = os.path.join(output_dir, filename)

                fig.write_html(filepath)
                saved_files.append(filepath)

                logger.info(f"Plot gespeichert: {filepath}")

            except Exception as e:
                logger.error(f"Fehler beim Speichern von {plot_name}: {str(e)}")

        return saved_files
