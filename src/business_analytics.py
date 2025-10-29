"""
BUSINESS ANALYTICS - MARKETING KAMPAGNE OPTIMIZER
================================================

Modul für alle Business Intelligence und ROI-Analyse Funktionen:
- ROI-Berechnungen für verschiedene Szenarien
- Kampagnen-Performance-Analyse
- Kundenwert-Berechnungen
- Business-Metriken und KPIs
"""

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .config import config

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BusinessAnalytics:
    """
    Hauptklasse für Business Analytics und ROI-Berechnungen
    """

    def __init__(self):
        """Initialisiert Business Analytics"""
        self.campaign_cost = config.business.DEFAULT_CAMPAIGN_COST
        self.revenue_per_conversion = config.business.DEFAULT_REVENUE_PER_CONVERSION

    def set_business_parameters(self, campaign_cost: float, revenue_per_conversion: float) -> None:
        """
        Setzt Business-Parameter für ROI-Berechnungen
        
        Args:
            campaign_cost: Kosten pro kontaktiertem Kunden
            revenue_per_conversion: Umsatz pro erfolgreicher Conversion
        """
        self.campaign_cost = campaign_cost
        self.revenue_per_conversion = revenue_per_conversion
        logger.info(f"Business-Parameter aktualisiert: Kosten={campaign_cost}€, Umsatz={revenue_per_conversion}€")

    def calculate_roi_scenarios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Berechnet ROI für verschiedene Targeting-Szenarien
        
        Args:
            df: DataFrame mit Conversion-Wahrscheinlichkeiten
            
        Returns:
            DataFrame mit ROI-Szenarien
        """
        if 'Conversion_Wahrscheinlichkeit' not in df.columns:
            raise ValueError("DataFrame muss 'Conversion_Wahrscheinlichkeit' Spalte enthalten")

        results = []

        for scenario_name, threshold in config.business.ROI_SCENARIOS.items():
            # Kunden auswählen basierend auf Threshold
            selected_customers = df[df['Conversion_Wahrscheinlichkeit'] >= threshold]

            # Metriken berechnen
            metrics = self._calculate_scenario_metrics(selected_customers)
            metrics['Szenario'] = scenario_name
            metrics['Threshold'] = threshold

            results.append(metrics)

        roi_df = pd.DataFrame(results)

        # Sortiere nach ROI absteigend
        roi_df = roi_df.sort_values('ROI', ascending=False)

        logger.info(f"ROI-Szenarien berechnet für {len(roi_df)} Strategien")
        return roi_df

    def _calculate_scenario_metrics(self, selected_customers: pd.DataFrame) -> Dict[str, float]:
        """
        Berechnet Metriken für ein spezifisches Szenario
        
        Args:
            selected_customers: DataFrame mit ausgewählten Kunden
            
        Returns:
            Dictionary mit berechneten Metriken
        """
        num_customers = len(selected_customers)

        if num_customers == 0:
            return {
                'Anzahl_Kunden': 0,
                'Erwartete_Conversions': 0,
                'Conversion_Rate': 0,
                'Kosten': 0,
                'Erwarteter_Umsatz': 0,
                'Erwarteter_Gewinn': 0,
                'ROI': 0
            }

        expected_conversions = selected_customers['Conversion_Wahrscheinlichkeit'].sum()

        # Kosten und Umsatz berechnen
        total_costs = num_customers * self.campaign_cost
        expected_revenue = expected_conversions * self.revenue_per_conversion
        expected_profit = expected_revenue - total_costs

        # ROI und Conversion Rate
        roi = (expected_profit / total_costs * 100) if total_costs > 0 else 0
        conversion_rate = (expected_conversions / num_customers * 100) if num_customers > 0 else 0

        return {
            'Anzahl_Kunden': num_customers,
            'Erwartete_Conversions': round(expected_conversions, 1),
            'Conversion_Rate': round(conversion_rate, 2),
            'Kosten': round(total_costs, 2),
            'Erwarteter_Umsatz': round(expected_revenue, 2),
            'Erwarteter_Gewinn': round(expected_profit, 2),
            'ROI': round(roi, 2)
        }

    def analyze_campaign_performance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analysiert die Performance historischer Kampagnen
        
        Args:
            df: DataFrame mit Kampagnen-Daten
            
        Returns:
            DataFrame mit Kampagnen-Performance-Statistiken
        """
        campaign_stats = {}

        for col in config.data.CAMPAIGN_COLUMNS + ['Antwort_Letzte_Kampagne']:
            acceptance_rate = df[col].mean() * 100
            total_responses = df[col].sum()

            # Berechne theoretischen ROI für diese Kampagne
            theoretical_cost = len(df) * self.campaign_cost
            theoretical_revenue = total_responses * self.revenue_per_conversion
            theoretical_roi = ((theoretical_revenue - theoretical_cost) / theoretical_cost * 100) if theoretical_cost > 0 else 0

            campaign_stats[col] = {
                'Akzeptanzrate (%)': round(acceptance_rate, 2),
                'Absolute_Antworten': total_responses,
                'Theoretischer_ROI (%)': round(theoretical_roi, 2)
            }

        campaign_df = pd.DataFrame(campaign_stats).T

        logger.info("Kampagnen-Performance analysiert")
        return campaign_df

    def segment_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Führt segment-spezifische Business-Analyse durch
        
        Args:
            df: DataFrame mit Segment-Informationen
            
        Returns:
            DataFrame mit Segment-Analyse
        """
        if 'Segment_Label' not in df.columns:
            raise ValueError("DataFrame muss 'Segment_Label' Spalte enthalten")

        segment_metrics = df.groupby('Segment_Label').agg({
            'Conversion_Wahrscheinlichkeit': ['mean', 'std', 'count'],
            'Gesamtausgaben': ['mean', 'median', 'std'],
            'Kampagnen_Engagement': 'mean',
            'Antwort_Letzte_Kampagne': 'mean',
            'Alter': 'mean',
            'Einkommen': 'mean'
        }).round(3)

        # Spalten-Namen vereinfachen
        segment_metrics.columns = [
            'Ø_Conversion_Prob', 'Std_Conversion_Prob', 'Anzahl_Kunden',
            'Ø_Ausgaben', 'Median_Ausgaben', 'Std_Ausgaben',
            'Ø_Engagement', 'Tatsächliche_Conversion_Rate',
            'Ø_Alter', 'Ø_Einkommen'
        ]

        # ROI pro Segment berechnen
        segment_roi = []
        for segment in segment_metrics.index:
            segment_data = df[df['Segment_Label'] == segment]
            roi_metrics = self._calculate_scenario_metrics(segment_data)
            segment_roi.append(roi_metrics['ROI'])

        segment_metrics['Segment_ROI'] = segment_roi

        logger.info(f"Segment-Analyse für {len(segment_metrics)} Segmente abgeschlossen")
        return segment_metrics

    def calculate_customer_lifetime_value(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Berechnet Customer Lifetime Value (CLV)
        
        Args:
            df: DataFrame mit Kundendaten
            
        Returns:
            DataFrame mit CLV-Berechnungen
        """
        df_clv = df.copy()

        # Basis-CLV Berechnung
        # CLV = (Durchschnittlicher Kaufwert × Kauffrequenz × Kundenlebensdauer) - Akquisitionskosten

        # Annahmen für CLV-Berechnung
        avg_customer_lifespan_years = 3  # Jahre
        acquisition_cost = self.campaign_cost * 2  # Doppelte Kampagnen-Kosten als Akquisitionskosten

        # Kauffrequenz schätzen (basierend auf Gesamtkäufen und Loyalitäts-Score)
        df_clv['Geschätzte_Kauffrequenz_Jahr'] = (
            df_clv['Gesamtkäufe'] / (df_clv['Loyalitäts_Score'] + 0.1)
        ).clip(upper=50)  # Maximum 50 Käufe pro Jahr

        # CLV berechnen
        df_clv['Customer_Lifetime_Value'] = (
            df_clv['Durchschnittlicher_Kaufwert'] *
            df_clv['Geschätzte_Kauffrequenz_Jahr'] *
            avg_customer_lifespan_years
        ) - acquisition_cost

        # CLV-Kategorien
        df_clv['CLV_Kategorie'] = pd.cut(
            df_clv['Customer_Lifetime_Value'],
            bins=[-np.inf, 0, 500, 1500, np.inf],
            labels=['Verlust', 'Niedrig', 'Mittel', 'Hoch']
        )

        logger.info("Customer Lifetime Value berechnet")
        return df_clv

    def optimize_campaign_budget(self, df: pd.DataFrame, total_budget: float) -> Dict[str, Any]:
        """
        Optimiert Budget-Allokation für maximalen ROI
        
        Args:
            df: DataFrame mit Kundendaten
            total_budget: Verfügbares Gesamtbudget
            
        Returns:
            Dictionary mit Optimierungs-Empfehlungen
        """
        max_customers = int(total_budget / self.campaign_cost)

        # Sortiere Kunden nach Conversion-Wahrscheinlichkeit
        df_sorted = df.sort_values('Conversion_Wahrscheinlichkeit', ascending=False)

        # Wähle Top-Kunden basierend auf Budget
        selected_customers = df_sorted.head(max_customers)

        # Berechne optimierte Metriken
        optimized_metrics = self._calculate_scenario_metrics(selected_customers)

        # Segment-Verteilung der ausgewählten Kunden
        if 'Segment_Label' in selected_customers.columns:
            segment_distribution = selected_customers['Segment_Label'].value_counts().to_dict()
        else:
            segment_distribution = {}

        # Vergleich mit "Alle Kunden" Strategie
        all_customers_metrics = self._calculate_scenario_metrics(df.head(max_customers))

        improvement = {
            'ROI_Verbesserung': optimized_metrics['ROI'] - all_customers_metrics['ROI'],
            'Gewinn_Verbesserung': optimized_metrics['Erwarteter_Gewinn'] - all_customers_metrics['Erwarteter_Gewinn'],
            'Conversion_Verbesserung': optimized_metrics['Conversion_Rate'] - all_customers_metrics['Conversion_Rate']
        }

        return {
            'total_budget': total_budget,
            'max_customers': max_customers,
            'optimized_metrics': optimized_metrics,
            'segment_distribution': segment_distribution,
            'improvement': improvement,
            'recommendation': self._generate_budget_recommendation(optimized_metrics, improvement)
        }

    def _generate_budget_recommendation(self, metrics: Dict[str, float], improvement: Dict[str, float]) -> str:
        """
        Generiert Budget-Optimierungs-Empfehlung
        
        Args:
            metrics: Optimierte Metriken
            improvement: Verbesserungs-Metriken
            
        Returns:
            Empfehlungs-Text
        """
        if metrics['ROI'] > 50:
            roi_assessment = "Exzellent"
        elif metrics['ROI'] > 20:
            roi_assessment = "Gut"
        elif metrics['ROI'] > 0:
            roi_assessment = "Akzeptabel"
        else:
            roi_assessment = "Unrentabel"

        recommendation = f"""
        Budget-Optimierung Empfehlung:
        
        ROI-Bewertung: {roi_assessment} ({metrics['ROI']:.1f}%)
        Erwarteter Gewinn: {metrics['Erwarteter_Gewinn']:,.2f}€
        
        Verbesserungen gegenüber Standard-Targeting:
        - ROI: +{improvement['ROI_Verbesserung']:.1f} Prozentpunkte
        - Gewinn: +{improvement['Gewinn_Verbesserung']:,.2f}€
        - Conversion-Rate: +{improvement['Conversion_Verbesserung']:.1f}%
        
        Empfehlung: {"Strategie umsetzen" if metrics['ROI'] > 0 else "Budget überdenken"}
        """

        return recommendation.strip()

    def generate_executive_summary(self, df: pd.DataFrame, roi_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generiert Executive Summary für Management
        
        Args:
            df: Haupt-DataFrame
            roi_df: ROI-Szenarien DataFrame
            
        Returns:
            Dictionary mit Executive Summary
        """
        # Beste Strategie identifizieren
        best_strategy = roi_df.loc[roi_df['ROI'].idxmax()]

        # Key Insights
        total_customers = len(df)
        high_potential_customers = len(df[df['Conversion_Wahrscheinlichkeit'] > 0.7])
        avg_conversion_prob = df['Conversion_Wahrscheinlichkeit'].mean()

        # Segment-Insights
        if 'Segment_Label' in df.columns:
            best_segment = df.groupby('Segment_Label')['Conversion_Wahrscheinlichkeit'].mean().idxmax()
            segment_performance = df.groupby('Segment_Label')['Conversion_Wahrscheinlichkeit'].mean().to_dict()
        else:
            best_segment = "Nicht verfügbar"
            segment_performance = {}

        # Potenzielle Jahres-Impact (bei monatlichen Kampagnen)
        annual_impact = best_strategy['Erwarteter_Gewinn'] * 12

        summary = {
            'total_customers': total_customers,
            'high_potential_customers': high_potential_customers,
            'high_potential_percentage': (high_potential_customers / total_customers * 100),
            'avg_conversion_probability': avg_conversion_prob,
            'best_strategy': {
                'name': best_strategy['Szenario'],
                'roi': best_strategy['ROI'],
                'expected_profit': best_strategy['Erwarteter_Gewinn'],
                'customers_to_target': best_strategy['Anzahl_Kunden']
            },
            'best_segment': best_segment,
            'segment_performance': segment_performance,
            'annual_impact_estimate': annual_impact,
            'key_recommendations': self._generate_key_recommendations(best_strategy, df)
        }

        logger.info("Executive Summary generiert")
        return summary

    def _generate_key_recommendations(self, best_strategy: pd.Series, df: pd.DataFrame) -> List[str]:
        """
        Generiert Schlüssel-Empfehlungen
        
        Args:
            best_strategy: Beste ROI-Strategie
            df: Haupt-DataFrame
            
        Returns:
            Liste mit Empfehlungen
        """
        recommendations = []

        # ROI-basierte Empfehlungen
        if best_strategy['ROI'] > 50:
            recommendations.append(f"Implementierung der '{best_strategy['Szenario']}' Strategie für maximalen ROI")
        elif best_strategy['ROI'] > 0:
            recommendations.append(f"'{best_strategy['Szenario']}' Strategie bietet positiven ROI - Umsetzung empfohlen")
        else:
            recommendations.append("Überarbeitung der Kampagnen-Parameter erforderlich - aktuell negativer ROI")

        # Segment-basierte Empfehlungen
        if 'Segment_Label' in df.columns:
            segment_performance = df.groupby('Segment_Label')['Conversion_Wahrscheinlichkeit'].mean()
            best_segment = segment_performance.idxmax()
            recommendations.append(f"Fokus auf '{best_segment}' Segment für höchste Conversion-Raten")

        # Budget-Empfehlungen
        if best_strategy['Anzahl_Kunden'] < len(df) * 0.5:
            recommendations.append("Selektives Targeting reduziert Kosten bei gleichzeitig höherer Effizienz")

        # Datenqualität-Empfehlungen
        recommendations.append("Kontinuierliche Modell-Updates mit neuen Kampagnen-Daten für bessere Vorhersagen")

        return recommendations
