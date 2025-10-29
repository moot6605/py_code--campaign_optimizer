"""
MACHINE LEARNING MODELS - MARKETING CAMPAIGN OPTIMIZER
=====================================================

Module for all ML functionality:
- Model training and evaluation
- Customer segmentation with clustering
- Predictions and scoring
- Model persistence
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
# Scikit-learn Imports
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     train_test_split)
from sklearn.preprocessing import StandardScaler

from .config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLModelManager:
    """
    Manager class for all ML models
    """

    def __init__(self):
        """Initialize ML Model Manager"""
        self.models: Dict[str, Any] = {}
        self.best_model: Optional[Any] = None
        self.best_model_name: str = ""
        self.scaler: Optional[StandardScaler] = None
        self.cluster_model: Optional[KMeans] = None
        self.cluster_scaler: Optional[StandardScaler] = None

    def train_classification_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Train various classification models
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Dict with model performance metrics
        """
        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config.model.TEST_SIZE,
            random_state=config.model.RANDOM_STATE,
            stratify=y
        )

        # Handle class imbalance
        X_train_balanced, y_train_balanced = self._handle_class_imbalance(X_train, y_train)

        # Feature Scaling
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_balanced)
        X_test_scaled = self.scaler.transform(X_test)

        # Define models
        models = {
            'Logistic Regression': LogisticRegression(
                random_state=config.model.RANDOM_STATE,
                max_iter=1000
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=config.model.RF_N_ESTIMATORS,
                random_state=config.model.RANDOM_STATE,
                max_depth=config.model.RF_MAX_DEPTH,
                min_samples_split=config.model.RF_MIN_SAMPLES_SPLIT
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=config.model.RF_N_ESTIMATORS,
                random_state=config.model.RANDOM_STATE
            )
        }

        results = {}
        best_f1 = 0

        for name, model in models.items():
            logger.info(f"Training {name}...")

            # Train model
            model.fit(X_train_scaled, y_train_balanced)

            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)

            # Cross-Validation
            cv_scores = cross_val_score(
                model, X_train_scaled, y_train_balanced,
                cv=config.model.CV_FOLDS, scoring='f1'
            )

            # Store results
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'test_data': (X_test_scaled, y_test)
            }

            # Track best model
            if f1 > best_f1:
                best_f1 = f1
                self.best_model = model
                self.best_model_name = name

            logger.info(f"{name} - F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

        self.models = results
        logger.info(f"Best model: {self.best_model_name} (F1: {best_f1:.4f})")

        # Speichere ML-Run-Ergebnisse
        self._save_run_results(results, X.shape[0], X.shape[1])

        return results

    def _handle_class_imbalance(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Handle class imbalance with SMOTE
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Tuple of (balanced features, balanced labels)
        """
        logger.info("Handling class imbalance with SMOTE...")
        logger.info(f"Before SMOTE: {y_train.value_counts().to_dict()}")

        smote = SMOTE(random_state=config.model.RANDOM_STATE)
        X_balanced, y_balanced = smote.fit_resample(X_train, y_train)

        logger.info(f"After SMOTE: {pd.Series(y_balanced).value_counts().to_dict()}")

        return X_balanced, y_balanced

    def perform_clustering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform K-Means clustering for customer segmentation
        
        Args:
            df: DataFrame with customer data
            
        Returns:
            DataFrame with cluster assignments
        """
        logger.info("Performing customer segmentation with K-Means...")

        # Select features for clustering
        X_cluster = df[config.model.CLUSTERING_FEATURES].copy()

        # Standardization for clustering
        self.cluster_scaler = StandardScaler()
        X_scaled = self.cluster_scaler.fit_transform(X_cluster)

        # K-Means Clustering
        self.cluster_model = KMeans(
            n_clusters=config.model.N_CLUSTERS,
            init=config.model.CLUSTER_INIT,
            n_init=config.model.CLUSTER_N_INIT,
            random_state=config.model.RANDOM_STATE
        )

        cluster_labels = self.cluster_model.fit_predict(X_scaled)

        # Add cluster labels to DataFrame
        df_clustered = df.copy()
        df_clustered['Kundensegment'] = cluster_labels
        df_clustered['Segment_Label'] = df_clustered['Kundensegment'].map(
            config.business.SEGMENT_LABELS
        )

        # Cluster statistics
        self._analyze_clusters(df_clustered)

        logger.info("Customer segmentation completed")
        return df_clustered

    def _analyze_clusters(self, df: pd.DataFrame) -> None:
        """
        Analysiert und loggt Cluster-Charakteristika
        
        Args:
            df: DataFrame mit Cluster-Zuordnungen
        """
        cluster_stats = df.groupby('Segment_Label')[config.model.CLUSTERING_FEATURES].mean()

        logger.info("Cluster-Charakteristika:")
        for segment in cluster_stats.index:
            stats = cluster_stats.loc[segment]
            logger.info(f"{segment}: Alter={stats['Alter']:.1f}, "
                       f"Einkommen={stats['Einkommen']:.0f}, "
                       f"Ausgaben={stats['Gesamtausgaben']:.0f}")

    def predict_conversion_probability(self, X: pd.DataFrame) -> np.ndarray:
        """
        Vorhersage der Conversion-Wahrscheinlichkeiten
        
        Args:
            X: Feature-Matrix
            
        Returns:
            Array mit Conversion-Wahrscheinlichkeiten
        """
        if self.best_model is None or self.scaler is None:
            raise ValueError("Modell muss zuerst trainiert werden")

        X_scaled = self.scaler.transform(X)
        probabilities = self.best_model.predict_proba(X_scaled)[:, 1]

        return probabilities

    def _save_run_results(self, results: Dict[str, Dict], n_samples: int, n_features: int) -> None:
        """
        Speichert ML-Run-Ergebnisse in JSON und CSV
        
        Args:
            results: Dictionary mit Modell-Ergebnissen
            n_samples: Anzahl Datensaetze
            n_features: Anzahl Features
        """
        # Erstelle Ausgabe-Verzeichnis falls nicht vorhanden
        output_dir = Path('ml_runs')
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # JSON-Report: Vollstaendige Run-Details
        json_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset': {
                'n_samples': n_samples,
                'n_features': n_features
            },
            'best_model': self.best_model_name,
            'models': {}
        }

        # CSV-Daten: Zeitreihen-Tracking
        csv_rows = []

        for name, metrics in results.items():
            # JSON: Detaillierte Metriken pro Modell
            json_data['models'][name] = {
                'accuracy': float(metrics['accuracy']),
                'f1_score': float(metrics['f1_score']),
                'roc_auc': float(metrics['roc_auc']),
                'cv_mean': float(metrics['cv_mean']),
                'cv_std': float(metrics['cv_std'])
            }

            # CSV: Eine Zeile pro Modell fuer historisches Tracking
            csv_rows.append({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model': name,
                'accuracy': metrics['accuracy'],
                'f1_score': metrics['f1_score'],
                'roc_auc': metrics['roc_auc'],
                'cv_mean': metrics['cv_mean'],
                'cv_std': metrics['cv_std'],
                'is_best': name == self.best_model_name,
                'n_samples': n_samples,
                'n_features': n_features
            })

        # Speichere JSON-Report (ein File pro Run)
        json_file = output_dir / f'run_{timestamp}.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        logger.info(f"JSON-Report gespeichert: {json_file}")

        # Speichere CSV-Log (append zu einer Datei fuer alle Runs)
        csv_file = output_dir / 'ml_runs_history.csv'
        df_new = pd.DataFrame(csv_rows)

        # Append zu existierender CSV oder erstelle neue
        if csv_file.exists():
            df_existing = pd.read_csv(csv_file)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(csv_file, index=False)
        else:
            df_new.to_csv(csv_file, index=False)

        logger.info(f"CSV-History aktualisiert: {csv_file}")

    def predict_single_customer(self, customer_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Vorhersage für einen einzelnen Kunden
        
        Args:
            customer_features: Dictionary mit Kunden-Features
            
        Returns:
            Dictionary mit Vorhersage-Ergebnissen
        """
        if self.best_model is None:
            raise ValueError("Modell muss zuerst trainiert werden")

        # Feature-Array erstellen
        feature_array = np.array([[customer_features[feat] for feat in config.model.ML_FEATURES]])

        # Vorhersage
        X_scaled = self.scaler.transform(feature_array)
        probability = self.best_model.predict_proba(X_scaled)[0, 1]
        prediction = self.best_model.predict(X_scaled)[0]

        # Risiko-Level bestimmen
        if probability > 0.7:
            risk_level = "Niedrig"
            recommendation = "Kontaktieren"
        elif probability > 0.4:
            risk_level = "Mittel"
            recommendation = "Prüfen"
        else:
            risk_level = "Hoch"
            recommendation = "Nicht kontaktieren"

        return {
            'probability': probability,
            'prediction': prediction,
            'risk_level': risk_level,
            'recommendation': recommendation
        }

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Gibt Feature Importance des besten Modells zurück
        
        Returns:
            DataFrame mit Feature Importance oder None
        """
        if self.best_model is None or not hasattr(self.best_model, 'feature_importances_'):
            return None

        importance_df = pd.DataFrame({
            'Feature': config.model.ML_FEATURES,
            'Importance': self.best_model.feature_importances_
        }).sort_values('Importance', ascending=False)

        return importance_df

    def evaluate_model_performance(self) -> Dict[str, Any]:
        """
        Evaluiert die Performance aller trainierten Modelle
        
        Returns:
            Dictionary mit Performance-Metriken
        """
        if not self.models:
            raise ValueError("Keine Modelle trainiert")

        performance_data = []

        for name, result in self.models.items():
            performance_data.append({
                'Modell': name,
                'Accuracy': result['accuracy'],
                'F1-Score': result['f1_score'],
                'ROC-AUC': result['roc_auc'],
                'CV F1-Score': result['cv_mean'],
                'CV Std': result['cv_std']
            })

        performance_df = pd.DataFrame(performance_data)

        return {
            'performance_table': performance_df,
            'best_model': self.best_model_name,
            'best_f1_score': performance_df.loc[performance_df['F1-Score'].idxmax(), 'F1-Score']
        }

    def save_models(self, base_path: str = None) -> Dict[str, str]:
        """
        Speichert trainierte Modelle
        
        Args:
            base_path: Basis-Pfad für Modell-Speicherung
            
        Returns:
            Dictionary mit gespeicherten Dateipfaden
        """
        if base_path is None:
            base_path = config.data.MODEL_SAVE_PATH

        saved_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            # Bestes Modell speichern
            if self.best_model is not None:
                model_path = config.get_model_path(f"best_model_{timestamp}")
                joblib.dump(self.best_model, model_path)
                saved_files['best_model'] = model_path

            # Scaler speichern
            if self.scaler is not None:
                scaler_path = config.get_model_path(f"scaler_{timestamp}")
                joblib.dump(self.scaler, scaler_path)
                saved_files['scaler'] = scaler_path

            # Cluster-Modell speichern
            if self.cluster_model is not None:
                cluster_path = config.get_model_path(f"cluster_model_{timestamp}")
                joblib.dump(self.cluster_model, cluster_path)
                saved_files['cluster_model'] = cluster_path

            # Cluster-Scaler speichern
            if self.cluster_scaler is not None:
                cluster_scaler_path = config.get_model_path(f"cluster_scaler_{timestamp}")
                joblib.dump(self.cluster_scaler, cluster_scaler_path)
                saved_files['cluster_scaler'] = cluster_scaler_path

            logger.info(f"Modelle erfolgreich gespeichert: {list(saved_files.keys())}")

        except Exception as e:
            logger.error(f"Fehler beim Speichern der Modelle: {str(e)}")

        return saved_files

    def load_models(self, model_paths: Dict[str, str]) -> bool:
        """
        Lädt gespeicherte Modelle
        
        Args:
            model_paths: Dictionary mit Pfaden zu den Modell-Dateien
            
        Returns:
            True bei Erfolg, False bei Fehler
        """
        try:
            if 'best_model' in model_paths:
                self.best_model = joblib.load(model_paths['best_model'])
                logger.info("Bestes Modell geladen")

            if 'scaler' in model_paths:
                self.scaler = joblib.load(model_paths['scaler'])
                logger.info("Scaler geladen")

            if 'cluster_model' in model_paths:
                self.cluster_model = joblib.load(model_paths['cluster_model'])
                logger.info("Cluster-Modell geladen")

            if 'cluster_scaler' in model_paths:
                self.cluster_scaler = joblib.load(model_paths['cluster_scaler'])
                logger.info("Cluster-Scaler geladen")

            return True

        except Exception as e:
            logger.error(f"Fehler beim Laden der Modelle: {str(e)}")
            return False

    def hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series, model_name: str = 'Random Forest') -> Dict[str, Any]:
        """
        Führt Hyperparameter-Tuning durch
        
        Args:
            X: Feature-Matrix
            y: Zielvariable
            model_name: Name des Modells für Tuning
            
        Returns:
            Dictionary mit besten Parametern und Performance
        """
        logger.info(f"Starte Hyperparameter-Tuning für {model_name}...")

        # Parameter-Grid definieren
        if model_name == 'Random Forest':
            model = RandomForestClassifier(random_state=config.model.RANDOM_STATE)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        else:
            logger.warning(f"Hyperparameter-Tuning für {model_name} nicht implementiert")
            return {}

        # Grid Search
        grid_search = GridSearchCV(
            model, param_grid,
            cv=config.model.CV_FOLDS,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )

        # Training-Daten vorbereiten
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.model.TEST_SIZE,
            random_state=config.model.RANDOM_STATE, stratify=y
        )

        X_train_balanced, y_train_balanced = self._handle_class_imbalance(X_train, y_train)

        if self.scaler is None:
            self.scaler = StandardScaler()

        X_train_scaled = self.scaler.fit_transform(X_train_balanced)

        # Grid Search durchführen
        grid_search.fit(X_train_scaled, y_train_balanced)

        logger.info(f"Beste Parameter: {grid_search.best_params_}")
        logger.info(f"Beste CV-Score: {grid_search.best_score_:.4f}")

        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_model': grid_search.best_estimator_
        }
