"""
CONFIGURATION - MARKETING CAMPAIGN OPTIMIZER
===========================================

Central config file for all parameters and constants.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import os

@dataclass
class DataConfig:
    """Data processing configuration"""
    
    # File paths
    DEFAULT_DATA_FILE: str = "Marktkampagne.csv"
    MODEL_SAVE_PATH: str = "models/"
    
    # Data cleaning
    OUTLIER_METHOD: str = "IQR"  # IQR oder Z-Score
    OUTLIER_THRESHOLD: float = 1.5
    
    # Feature Engineering
    CURRENT_YEAR: int = 2024
    PREMIUM_CATEGORIES: List[str] = ["Ausgaben_Wein", "Ausgaben_Gold"]
    
    # Spending categories
    SPENDING_COLUMNS: List[str] = [
        "Ausgaben_Wein", "Ausgaben_Obst", "Ausgaben_Fleisch",
        "Ausgaben_Fisch", "Ausgaben_Süßigkeiten", "Ausgaben_Gold"
    ]
    
    # Purchase channel categories
    PURCHASE_CHANNELS: List[str] = [
        "Anzahl_Webkäufe", "Anzahl_Katalogkäufe", "Anzahl_Ladeneinkäufe"
    ]
    
    # Campaign columns
    CAMPAIGN_COLUMNS: List[str] = [
        "Kampagne_1_Akzeptiert", "Kampagne_2_Akzeptiert", "Kampagne_3_Akzeptiert",
        "Kampagne_4_Akzeptiert", "Kampagne_5_Akzeptiert"
    ]

@dataclass
class ModelConfig:
    """ML model configuration"""
    
    # Model parameters
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    CV_FOLDS: int = 5
    
    # Random Forest Parameter
    RF_N_ESTIMATORS: int = 100
    RF_MAX_DEPTH: int = None
    RF_MIN_SAMPLES_SPLIT: int = 2
    
    # Clustering Parameter
    N_CLUSTERS: int = 4
    CLUSTER_INIT: str = "k-means++"
    CLUSTER_N_INIT: int = 10
    
    # ML feature selection
    ML_FEATURES: List[str] = [
        'Alter', 'Bildungsniveau_encoded', 'Familienstand_encoded', 'Einkommen',
        'Haushaltsgröße', 'Hat_Kinder', 'Gesamtausgaben', 'Premium_Anteil',
        'Gesamtkäufe', 'Durchschnittlicher_Kaufwert', 'Kampagnen_Engagement',
        'Loyalitäts_Score', 'Letzter_Kauf_Tage', 'Anzahl_WebBesuche_Monat', 'Kundensegment'
    ]
    
    # Features für Clustering
    CLUSTERING_FEATURES: List[str] = [
        'Alter', 'Einkommen', 'Gesamtausgaben', 'Gesamtkäufe',
        'Premium_Anteil', 'Kampagnen_Engagement', 'Loyalitäts_Score'
    ]

@dataclass
class BusinessConfig:
    """Business parameters for ROI calculations"""
    
    # Default values
    DEFAULT_CAMPAIGN_COST: float = 5.0  # Euro pro Kunde
    DEFAULT_REVENUE_PER_CONVERSION: float = 150.0  # Euro pro Conversion
    
    # ROI scenarios
    ROI_SCENARIOS: Dict[str, float] = {
        'Alle Kunden': 0.0,
        'Top 75%': 0.25,
        'Top 50%': 0.5,
        'Top 25%': 0.75,
        'Top 10%': 0.9
    }
    
    # Segment labels
    SEGMENT_LABELS: Dict[int, str] = {
        0: 'Gelegenheitskunden',
        1: 'Loyale Stammkunden',
        2: 'Premium Kunden',
        3: 'Preisbewusste Kunden'
    }

@dataclass
class UIConfig:
    """UI configuration"""
    
    # Streamlit config
    PAGE_TITLE: str = "Marketing Kampagne Optimizer"
    PAGE_ICON: str = "📊"
    LAYOUT: str = "wide"
    
    # Colors and styling
    PRIMARY_COLOR: str = "#1f77b4"
    SUCCESS_COLOR: str = "#28a745"
    WARNING_COLOR: str = "#ffc107"
    ERROR_COLOR: str = "#dc3545"
    
    # Slider ranges
    CAMPAIGN_COST_RANGE: tuple = (1.0, 20.0, 0.5)  # min, max, step
    REVENUE_RANGE: tuple = (50.0, 500.0, 10.0)
    PROBABILITY_RANGE: tuple = (0.0, 1.0, 0.05)
    
    # Export settings
    MAX_EXPORT_ROWS: int = 1000
    DATE_FORMAT: str = "%Y%m%d_%H%M%S"

# Globale Konfiguration
class Config:
    """Main configuration class"""
    
    def __init__(self):
        self.data = DataConfig()
        self.model = ModelConfig()
        self.business = BusinessConfig()
        self.ui = UIConfig()
    
    def get_model_path(self, model_name: str) -> str:
        """Get full path for model storage"""
        os.makedirs(self.data.MODEL_SAVE_PATH, exist_ok=True)
        return os.path.join(self.data.MODEL_SAVE_PATH, f"{model_name}.pkl")
    
    def validate_data_columns(self, df_columns: List[str]) -> bool:
        """Validate required columns are present"""
        required_columns = (
            self.data.SPENDING_COLUMNS + 
            self.data.PURCHASE_CHANNELS + 
            self.data.CAMPAIGN_COLUMNS +
            ['ID', 'Geburtsjahr', 'Bildungsniveau', 'Familienstand', 'Einkommen',
             'Kinder_zu_Hause', 'Teenager_zu_Hause', 'Datum_Kunde', 'Letzter_Kauf_Tage',
             'Anzahl_WebBesuche_Monat', 'Antwort_Letzte_Kampagne']
        )
        
        missing_columns = [col for col in required_columns if col not in df_columns]
        
        if missing_columns:
            print(f"Missing columns: {missing_columns}")
            return False
        
        return True

# Singleton instance
config = Config()