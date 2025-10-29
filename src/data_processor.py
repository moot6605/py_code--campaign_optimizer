"""
DATA PROCESSING - MARKETING CAMPAIGN OPTIMIZER
==============================================

Module for data processing and feature engineering.
Handles data import, cleaning, transformation and feature creation.
"""

import logging
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Main class for data processing and feature engineering
    """

    def __init__(self):
        """Initialize DataProcessor"""
        self.encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: list = []

    def load_data(self, filepath: str = None) -> Optional[pd.DataFrame]:
        """
        Load marketing campaign data from CSV
        
        Args:
            filepath: Path to CSV file (optional)
            
        Returns:
            DataFrame with loaded data or None on error
        """
        if filepath is None:
            filepath = config.data.DEFAULT_DATA_FILE

        try:
            df = pd.read_csv(filepath)
            logger.info(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")

            # Validate columns
            if not config.validate_data_columns(df.columns.tolist()):
                logger.error("Data validation failed")
                return None

            return df

        except FileNotFoundError:
            logger.error(f"File {filepath} not found")
            return None
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return None

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw data
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()

        # Handle missing values
        df_clean['Einkommen'].fillna(df_clean['Einkommen'].median(), inplace=True)

        # Convert dates - support multiple formats
        try:
            # Try different date formats
            df_clean['Datum_Kunde'] = pd.to_datetime(df_clean['Datum_Kunde'],
                                                    format='%d-%m-%Y', errors='coerce')
            if df_clean['Datum_Kunde'].isna().any():
                df_clean['Datum_Kunde'] = pd.to_datetime(df_clean['Datum_Kunde'],
                                                        format='%Y-%m-%d', errors='coerce')
            if df_clean['Datum_Kunde'].isna().any():
                df_clean['Datum_Kunde'] = pd.to_datetime(df_clean['Datum_Kunde'],
                                                        infer_datetime_format=True, errors='coerce')
        except Exception as e:
            logger.warning(f"Date conversion failed: {e}")
            # Fallback: Set default date
            df_clean['Datum_Kunde'] = pd.to_datetime('2020-01-01')

        # Outlier handling for income
        outliers_removed = self._remove_outliers(df_clean, 'Einkommen')

        logger.info(f"Data cleaning completed. {outliers_removed} outliers removed.")
        return df_clean

    def _remove_outliers(self, df: pd.DataFrame, column: str) -> int:
        """
        Remove outliers based on IQR method
        
        Args:
            df: DataFrame
            column: Column name for outlier handling
            
        Returns:
            Number of removed outliers
        """
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - config.data.OUTLIER_THRESHOLD * IQR
        upper_bound = Q3 + config.data.OUTLIER_THRESHOLD * IQR

        outliers_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
        outliers_count = outliers_mask.sum()

        # Remove outliers
        df.drop(df[outliers_mask].index, inplace=True)
        df.reset_index(drop=True, inplace=True)

        return outliers_count

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Erstellt neue Features durch Feature Engineering
        
        Args:
            df: Eingabe-DataFrame
            
        Returns:
            DataFrame mit neuen Features
        """
        df_features = df.copy()

        # Demografische Features
        df_features['Alter'] = config.data.CURRENT_YEAR - df_features['Geburtsjahr']
        df_features['Altersgruppe'] = pd.cut(
            df_features['Alter'],
            bins=[0, 30, 45, 60, 100],
            labels=['Jung', 'Mittel', 'Senior', 'Rentner']
        )

        # Haushalts-Features
        df_features['Haushaltsgröße'] = (
            df_features['Kinder_zu_Hause'] + df_features['Teenager_zu_Hause']
        )
        df_features['Hat_Kinder'] = (df_features['Haushaltsgröße'] > 0).astype(int)

        # Ausgaben-Features
        df_features = self._create_spending_features(df_features)

        # Kaufverhalten-Features
        df_features = self._create_purchase_features(df_features)

        # Engagement-Features
        df_features = self._create_engagement_features(df_features)

        # Loyalitäts-Features
        df_features = self._create_loyalty_features(df_features)

        # Kundenwert-Segmentierung
        df_features['Kundenwert_Quartil'] = pd.qcut(
            df_features['Gesamtausgaben'],
            q=4,
            labels=['Niedrig', 'Mittel', 'Hoch', 'Premium']
        )

        logger.info(f"Feature Engineering abgeschlossen. {len(df_features.columns) - len(df.columns)} neue Features erstellt.")
        return df_features

    def _create_spending_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Erstellt ausgabenbezogene Features"""
        # Gesamtausgaben
        df['Gesamtausgaben'] = df[config.data.SPENDING_COLUMNS].sum(axis=1)
        df['Durchschnittsausgaben'] = df['Gesamtausgaben'] / len(config.data.SPENDING_COLUMNS)

        # Premium-Produkte Features
        df['Premium_Ausgaben'] = df[config.data.PREMIUM_CATEGORIES].sum(axis=1)
        df['Premium_Anteil'] = df['Premium_Ausgaben'] / (df['Gesamtausgaben'] + 1)

        return df

    def _create_purchase_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Erstellt kaufverhaltensbezogene Features"""
        # Gesamtkäufe
        df['Gesamtkäufe'] = df[config.data.PURCHASE_CHANNELS].sum(axis=1)
        df['Durchschnittlicher_Kaufwert'] = df['Gesamtausgaben'] / (df['Gesamtkäufe'] + 1)

        # Kanal-Präferenzen
        df['Bevorzugt_Online'] = (
            df['Anzahl_Webkäufe'] > df['Anzahl_Ladeneinkäufe']
        ).astype(int)

        df['Multi_Channel'] = (
            (df['Anzahl_Webkäufe'] > 0) &
            (df['Anzahl_Ladeneinkäufe'] > 0)
        ).astype(int)

        return df

    def _create_engagement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Erstellt engagement-bezogene Features"""
        # Kampagnen-Engagement
        df['Kampagnen_Engagement'] = df[config.data.CAMPAIGN_COLUMNS].sum(axis=1)

        # Engagement-Rate
        df['Engagement_Rate'] = df['Kampagnen_Engagement'] / len(config.data.CAMPAIGN_COLUMNS)

        return df

    def _create_loyalty_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Erstellt loyalitätsbezogene Features"""
        # Kundenloyalität basierend auf Registrierungsdatum
        try:
            df['Kunde_seit_Tagen'] = (datetime.now() - df['Datum_Kunde']).dt.days
            # Negative Werte korrigieren (falls Datum in der Zukunft)
            df['Kunde_seit_Tagen'] = df['Kunde_seit_Tagen'].clip(lower=0)
            df['Loyalitäts_Score'] = df['Kunde_seit_Tagen'] / 365  # Jahre als Kunde
        except Exception as e:
            logger.warning(f"Loyalitäts-Feature Erstellung fehlgeschlagen: {e}")
            # Fallback: Standard-Werte
            df['Kunde_seit_Tagen'] = 365
            df['Loyalitäts_Score'] = 1.0

        return df

    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encodiert kategorische Variablen
        
        Args:
            df: DataFrame mit kategorischen Variablen
            
        Returns:
            DataFrame mit encodierten Variablen
        """
        df_encoded = df.copy()

        # Label Encoding für kategorische Variablen
        categorical_columns = {
            'Bildungsniveau': 'bildung',
            'Familienstand': 'familienstand'
        }

        for column, encoder_name in categorical_columns.items():
            if encoder_name not in self.encoders:
                self.encoders[encoder_name] = LabelEncoder()
                df_encoded[f'{column}_encoded'] = self.encoders[encoder_name].fit_transform(df_encoded[column])
            else:
                df_encoded[f'{column}_encoded'] = self.encoders[encoder_name].transform(df_encoded[column])

        return df_encoded

    def prepare_ml_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
        """
        Bereitet Features für Machine Learning vor
        
        Args:
            df: DataFrame mit allen Features
            
        Returns:
            Tuple aus (Feature-DataFrame, Feature-Namen-Liste)
        """
        # Stelle sicher, dass alle ML-Features vorhanden sind
        available_features = [feat for feat in config.model.ML_FEATURES if feat in df.columns]

        if len(available_features) != len(config.model.ML_FEATURES):
            missing_features = set(config.model.ML_FEATURES) - set(available_features)
            logger.warning(f"Fehlende ML-Features: {missing_features}")

        self.feature_names = available_features
        X = df[available_features].copy()

        return X, self.feature_names

    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Skaliert Features für ML-Modelle
        
        Args:
            X_train: Training-Features
            X_test: Test-Features (optional)
            
        Returns:
            Tuple aus (skalierte Training-Features, skalierte Test-Features)
        """
        if self.scaler is None:
            self.scaler = StandardScaler()

        X_train_scaled = self.scaler.fit_transform(X_train)

        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled

        return X_train_scaled, None

    def process_pipeline(self, filepath: str = None) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, list]]:
        """
        Vollständige Datenverarbeitungs-Pipeline
        
        Args:
            filepath: Pfad zur Datendatei
            
        Returns:
            Tuple aus (verarbeiteter DataFrame, ML-Features, Feature-Namen) oder None
        """
        try:
            # Daten laden
            df = self.load_data(filepath)
            if df is None:
                return None

            # Daten bereinigen
            df_clean = self.clean_data(df)

            # Features erstellen
            df_features = self.create_features(df_clean)

            # Kategorische Variablen encodieren
            df_encoded = self.encode_categorical_features(df_features)

            # ML-Features vorbereiten
            X, feature_names = self.prepare_ml_features(df_encoded)

            logger.info("Datenverarbeitungs-Pipeline erfolgreich abgeschlossen")
            return df_encoded, X, feature_names

        except Exception as e:
            logger.error(f"Fehler in der Datenverarbeitungs-Pipeline: {str(e)}")
            return None
