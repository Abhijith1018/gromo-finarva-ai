# Complete implementation for Google Colab
# Run this after you've loaded your CSV file

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import joblib
from scipy import stats
from scipy.signal import savgol_filter
import shap
from sklearn.preprocessing import MinMaxScaler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LightGBMDriverRiskAnalyzer:
    def __init__(self, model_params: Dict = None):
        """
        Initialize LightGBM-based driver risk analyzer for per-second telemetry data

        Args:
            model_params: LightGBM parameters dictionary
        """
        self.model_params = model_params or {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 100,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'n_estimators': 500,
            'early_stopping_rounds': 50
        }

        self.model = None
        self.feature_importance = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.risk_thresholds = {'low': 0.3, 'medium': 0.7, 'high': 1.0}

    def load_and_preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Load and preprocess the per-second driving data

        Args:
            df: Raw driving data DataFrame

        Returns:
            Preprocessed DataFrame
        """
        df = df.copy()

        # Add timestamp column based on index (assuming per-second data)
        df['timestamp'] = pd.date_range(start='2024-01-01 09:00:00', periods=len(df), freq='1S')
        df['seconds_elapsed'] = df.index

        # Extract time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_rush_hour'] = ((df['hour'].between(7, 9)) | (df['hour'].between(17, 19))).astype(int)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

        logger.info(f"Loaded {len(df)} seconds of driving data")
        return df

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-series features from per-second data

        Args:
            df: Preprocessed DataFrame

        Returns:
            DataFrame with temporal features
        """
        df = df.copy()

        # Define window sizes for rolling features
        windows = [5, 10, 30, 60]  # 5s, 10s, 30s, 60s windows

        # Core telemetry columns for feature engineering
        core_features = ['speed', 'rpm', 'acceleration', 'throttle_position',
                        'engine_temperature', 'system_voltage', 'engine_load_value']

        for feature in core_features:
            if feature in df.columns:
                # Rolling statistics
                for window in windows:
                    if len(df) > window:
                        df[f'{feature}_mean_{window}s'] = df[feature].rolling(window=window, min_periods=1).mean()
                        df[f'{feature}_std_{window}s'] = df[feature].rolling(window=window, min_periods=1).std().fillna(0)
                        df[f'{feature}_max_{window}s'] = df[feature].rolling(window=window, min_periods=1).max()
                        df[f'{feature}_min_{window}s'] = df[feature].rolling(window=window, min_periods=1).min()

                # Lag features (previous values)
                for lag in [1, 5, 10]:
                    if len(df) > lag:
                        df[f'{feature}_lag_{lag}s'] = df[feature].shift(lag).fillna(df[feature].iloc[0])

                # Rate of change
                df[f'{feature}_rate_change'] = df[feature].diff().fillna(0)
                df[f'{feature}_rate_change_abs'] = np.abs(df[f'{feature}_rate_change'])

        # Smooth core features using Savitzky-Golay filter
        for feature in ['speed', 'acceleration', 'rpm']:
            if feature in df.columns and len(df) > 11:
                try:
                    df[f'{feature}_smooth'] = savgol_filter(df[feature], window_length=min(11, len(df)//2*2+1), polyorder=2)
                except:
                    df[f'{feature}_smooth'] = df[feature]

        # Speed-acceleration relationship
        if 'speed' in df.columns and 'acceleration' in df.columns:
            df['speed_accel_product'] = df['speed'] * np.abs(df['acceleration'])

        if 'rpm' in df.columns and 'speed' in df.columns:
            df['rpm_speed_ratio'] = np.where(df['speed'] > 5, df['rpm'] / df['speed'], 0)

        # Driving efficiency metrics
        if 'engine_load_value' in df.columns and 'speed' in df.columns:
            df['fuel_efficiency_proxy'] = df['engine_load_value'] / (df['speed'] + 1)

        if 'rpm' in df.columns and 'engine_load_value' in df.columns:
            df['engine_stress'] = (df['rpm'] / 6000) * (df['engine_load_value'] / 100)

        # Contextual features
        if 'heart_rate' in df.columns and 'body_temperature' in df.columns:
            df['biometric_stress'] = ((df['heart_rate'] - 70) / 50) + ((df['body_temperature'] - 36.5) / 1.5)
            df['high_stress'] = (df['biometric_stress'] > df['biometric_stress'].quantile(0.8)).astype(int)

        # Environmental risk factors
        if 'has_precipitation' in df.columns:
            df['weather_risk'] = df['has_precipitation'].astype(int)
            if 'is_day_time' in df.columns:
                df['night_rain_risk'] = (df['has_precipitation'] & (1 - df['is_day_time'])).astype(int)

        # GPS-based features (if available)
        if all(col in df.columns for col in ['latitude', 'longitude']):
            # Calculate GPS speed (approximate)
            df['gps_speed'] = np.sqrt(
                (df['latitude'].diff() * 111000) ** 2 +
                (df['longitude'].diff() * 111000 * np.cos(np.radians(df['latitude']))) ** 2
            ) * 3.6  # Convert to km/h
            df['gps_speed'] = df['gps_speed'].fillna(0)

            # Speed discrepancy (OBD vs GPS)
            if 'speed' in df.columns:
                df['speed_discrepancy'] = np.abs(df['speed'] - df['gps_speed'])
                df['high_speed_discrepancy'] = (df['speed_discrepancy'] > 10).astype(int)

        logger.info(f"Created temporal features. DataFrame now has {df.shape[1]} columns")
        return df

    def create_risk_labels(self, df: pd.DataFrame, method: str = 'composite') -> pd.DataFrame:
        """
        Create risk labels for training based on driving behavior patterns

        Args:
            df: DataFrame with features
            method: Method for creating labels ('composite', 'threshold', 'anomaly')

        Returns:
            DataFrame with risk labels
        """
        df = df.copy()

        if method == 'composite':
            # Multi-factor risk scoring
            risk_score = np.zeros(len(df))

            # Speed-based risk
            if 'speed' in df.columns:
                risk_score += np.where(df['speed'] > 100, 3,
                                     np.where(df['speed'] > 80, 2,
                                             np.where(df['speed'] > 60, 1, 0)))

            # Acceleration-based risk
            if 'acceleration' in df.columns:
                risk_score += np.where(np.abs(df['acceleration']) > 0.5, 3,
                                     np.where(np.abs(df['acceleration']) > 0.3, 2,
                                             np.where(np.abs(df['acceleration']) > 0.2, 1, 0)))

            # Engine stress risk
            if 'rpm' in df.columns:
                risk_score += np.where(df['rpm'] > 5000, 2,
                                     np.where(df['rpm'] > 4000, 1, 0))

            # Temperature risk
            if 'engine_temperature' in df.columns:
                risk_score += np.where(df['engine_temperature'] > 110, 2,
                                     np.where(df['engine_temperature'] > 100, 1, 0))

            # Volatility risk (if available)
            if 'speed_std_30s' in df.columns:
                speed_volatility_threshold = df['speed_std_30s'].quantile(0.8)
                risk_score += (df['speed_std_30s'] > speed_volatility_threshold).astype(int)

            # Biometric risk (if available)
            if 'high_stress' in df.columns:
                risk_score += df['high_stress'] * 2

            # Environmental risk
            if 'weather_risk' in df.columns:
                risk_score += df['weather_risk']

            # Create binary labels (high risk vs normal)
            risk_threshold = np.percentile(risk_score, 85)  # Top 15% as high risk
            df['is_high_risk'] = (risk_score >= risk_threshold).astype(int)
            df['risk_score_raw'] = risk_score

        elif method == 'threshold':
            # Simple threshold-based labeling
            high_risk_conditions = np.zeros(len(df), dtype=bool)

            if 'speed' in df.columns:
                high_risk_conditions |= (df['speed'] > 90)
            if 'acceleration' in df.columns:
                high_risk_conditions |= (np.abs(df['acceleration']) > 0.4)
            if 'rpm' in df.columns:
                high_risk_conditions |= (df['rpm'] > 4500)
            if 'engine_temperature' in df.columns:
                high_risk_conditions |= (df['engine_temperature'] > 105)

            df['is_high_risk'] = high_risk_conditions.astype(int)

        elif method == 'anomaly':
            # Use isolation forest for anomaly-based labeling
            from sklearn.ensemble import IsolationForest

            features_for_anomaly = ['speed', 'acceleration', 'rpm', 'engine_temperature', 'throttle_position']
            available_features = [f for f in features_for_anomaly if f in df.columns]

            if len(available_features) > 0:
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                anomaly_labels = iso_forest.fit_predict(df[available_features].fillna(0))
                df['is_high_risk'] = (anomaly_labels == -1).astype(int)
            else:
                df['is_high_risk'] = 0

        # Create multi-class risk levels
        if 'risk_score_raw' in df.columns:
            df['risk_level'] = pd.cut(df['risk_score_raw'],
                                    bins=[-np.inf, 2, 5, np.inf],
                                    labels=['low', 'medium', 'high'])
        else:
            df['risk_level'] = np.where(df['is_high_risk'] == 1, 'high', 'low')

        risk_distribution = df['is_high_risk'].value_counts(normalize=True)
        logger.info(f"Risk label distribution: {risk_distribution.to_dict()}")

        return df


    def prepare_features_for_training(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare features for LightGBM training

        Args:
            df: DataFrame with all features

        Returns:
            Tuple of (feature DataFrame, feature names list)
        """
        # Exclude non-feature columns
        exclude_columns = [
            'timestamp', 'is_high_risk', 'risk_level', 'risk_score_raw',
            'id_vehicle', 'id_driver', 'latitude', 'longitude'  # Keep these separate if needed for analysis
        ]

        feature_columns = [col for col in df.columns if col not in exclude_columns]

        # Handle categorical variables
        categorical_features = ['current_weather', 'observation_hour']

        for cat_feature in categorical_features:
            if cat_feature in feature_columns:
                if cat_feature not in self.label_encoders:
                    self.label_encoders[cat_feature] = LabelEncoder()
                    df[cat_feature] = self.label_encoders[cat_feature].fit_transform(df[cat_feature].astype(str))
                else:
                    df[cat_feature] = self.label_encoders[cat_feature].transform(df[cat_feature].astype(str))

        # Select final features and handle any remaining issues
        X = df[feature_columns].copy()

        # Remove any columns with all NaN or constant values
        X = X.loc[:, X.std() != 0]  # Remove constant columns
        X = X.dropna(axis=1, how='all')  # Remove all-NaN columns

        # Fill any remaining NaN values
        X = X.fillna(X.median())

        # Remove highly correlated features (>0.95 correlation)
        corr_matrix = X.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
        X = X.drop(columns=high_corr_features)

        feature_names = list(X.columns)
        self.feature_names = feature_names

        logger.info(f"Prepared {len(feature_names)} features for training")
        logger.info(f"Removed {len(high_corr_features)} highly correlated features")

        return X, feature_names

    def train_model(self, df: pd.DataFrame, validation_split: float = 0.2) -> Dict:
        """
        Train LightGBM model on the prepared data

        Args:
            df: DataFrame with features and labels
            validation_split: Fraction of data to use for validation

        Returns:
            Dictionary with training results
        """
        logger.info("Starting model training...")

        # Prepare features
        X, feature_names = self.prepare_features_for_training(df)
        y = df['is_high_risk'].values

        # Time-based split to avoid data leakage
        split_index = int(len(df) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_val = y[:split_index], y[split_index:]

        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Train model
        self.model = lgb.train(
            self.model_params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )

        # Get predictions
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)

        # Calculate metrics
        train_auc = roc_auc_score(y_train, train_pred)
        val_auc = roc_auc_score(y_val, val_pred)

        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)

        results = {
            'train_auc': train_auc,
            'val_auc': val_auc,
            'best_iteration': self.model.best_iteration,
            'feature_importance': self.feature_importance,
            'train_predictions': train_pred,
            'val_predictions': val_pred,
            'y_train': y_train,
            'y_val': y_val
        }

        logger.info(f"Training completed. Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")

        return results

    def predict_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict risk scores for new data

        Args:
            df: DataFrame with same structure as training data

        Returns:
            DataFrame with risk predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")

        # Apply same preprocessing
        df_processed = self.load_and_preprocess_data(df)
        df_processed = self.create_temporal_features(df_processed)

        # Prepare features
        X, _ = self.prepare_features_for_training(df_processed)

        # Make predictions
        risk_probabilities = self.model.predict(X)

        # Create risk categories
        risk_categories = pd.cut(risk_probabilities,
                               bins=[0, self.risk_thresholds['low'],
                                    self.risk_thresholds['medium'], 1],
                               labels=['low', 'medium', 'high'])

        # Add predictions to dataframe
        result_df = df.copy()
        result_df['risk_probability'] = risk_probabilities
        result_df['risk_category'] = risk_categories
        result_df['timestamp'] = pd.date_range(start='2024-01-01 09:00:00', periods=len(df), freq='1S')

        return result_df

    def create_visualizations(self, df: pd.DataFrame, predictions_df: pd.DataFrame = None):
        """
        Create comprehensive visualizations for the analysis
        """
        plt.style.use('default')
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Driver Risk Analysis Dashboard', fontsize=16, fontweight='bold')

        # 1. Speed and Risk over time
        if predictions_df is not None:
            time_minutes = np.arange(len(predictions_df)) / 60
            ax1 = axes[0, 0]
            ax1_twin = ax1.twinx()

            ax1.plot(time_minutes, predictions_df['speed'], 'b-', alpha=0.7, label='Speed')
            ax1_twin.plot(time_minutes, predictions_df['risk_probability'], 'r-', linewidth=2, label='Risk')
            ax1_twin.axhline(y=self.risk_thresholds['medium'], color='orange', linestyle='--', alpha=0.7)

            ax1.set_xlabel('Time (minutes)')
            ax1.set_ylabel('Speed (km/h)', color='b')
            ax1_twin.set_ylabel('Risk Probability', color='r')
            ax1.set_title('Speed vs Risk Over Time')
            ax1.legend(loc='upper left')
            ax1_twin.legend(loc='upper right')

        # 2. Feature Importance
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(10)
            axes[0, 1].barh(range(len(top_features)), top_features['importance'])
            axes[0, 1].set_yticks(range(len(top_features)))
            axes[0, 1].set_yticklabels(top_features['feature'])
            axes[0, 1].set_xlabel('Importance')
            axes[0, 1].set_title('Top 10 Risk Factors')

        # 3. Speed Distribution
        axes[1, 0].hist(df['speed'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].axvline(df['speed'].mean(), color='red', linestyle='--', label=f'Mean: {df["speed"].mean():.1f}')
        axes[1, 0].set_xlabel('Speed (km/h)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Speed Distribution')
        axes[1, 0].legend()

        # 4. Engine Metrics
        if 'rpm' in df.columns and 'engine_temperature' in df.columns:
            scatter = axes[1, 1].scatter(df['rpm'], df['engine_temperature'],
                                       c=df['speed'], cmap='viridis', alpha=0.6)
            axes[1, 1].set_xlabel('RPM')
            axes[1, 1].set_ylabel('Engine Temperature (°C)')
            axes[1, 1].set_title('Engine Performance Map')
            plt.colorbar(scatter, ax=axes[1, 1], label='Speed (km/h)')

        # 5. Risk Distribution
        if predictions_df is not None:
            axes[2, 0].hist(predictions_df['risk_probability'], bins=30, alpha=0.7, color='coral', edgecolor='black')
            axes[2, 0].axvline(predictions_df['risk_probability'].mean(), color='red', linestyle='--',
                             label=f'Mean: {predictions_df["risk_probability"].mean():.3f}')
            axes[2, 0].set_xlabel('Risk Probability')
            axes[2, 0].set_ylabel('Frequency')
            axes[2, 0].set_title('Risk Score Distribution')
            axes[2, 0].legend()

        # 6. Acceleration Analysis
        if 'acceleration' in df.columns:
            axes[2, 1].plot(df['acceleration'], alpha=0.7, color='green')
            axes[2, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[2, 1].axhline(y=0.3, color='orange', linestyle='--', alpha=0.7, label='Aggressive Threshold')
            axes[2, 1].axhline(y=-0.3, color='red', linestyle='--', alpha=0.7, label='Hard Braking')
            axes[2, 1].set_xlabel('Time (seconds)')
            axes[2, 1].set_ylabel('Acceleration (m/s²)')
            axes[2, 1].set_title('Acceleration Pattern')
            axes[2, 1].legend()

        plt.tight_layout()
        plt.show()

    def generate_report(self, df: pd.DataFrame, predictions_df: pd.DataFrame = None, training_results: Dict = None):
        """
        Generate detailed analysis report
        """

        print("="*80)
        print("LIGHTGBM DRIVER RISK ANALYSIS REPORT")
        print("="*80)

        # Data Overview
        print(f"\n📊 DATA OVERVIEW:")
        print(f"Total Duration: {len(df)} seconds ({len(df)/60:.1f} minutes)")
        print(f"Data Points Analyzed: {len(df):,}")
        print(f"Features Used for Training: {len(self.feature_names) if self.feature_names else 'N/A'}")

        if predictions_df is not None:
            # Risk Analysis
            print(f"\n⚠️  RISK ANALYSIS:")
            avg_risk = predictions_df['risk_probability'].mean()
            max_risk = predictions_df['risk_probability'].max()
            high_risk_pct = (predictions_df['risk_probability'] > self.risk_thresholds['medium']).mean() * 100

            print(f"Average Risk Score: {avg_risk:.3f}")
            print(f"Maximum Risk Score: {max_risk:.3f}")
            print(f"High Risk Time: {high_risk_pct:.1f}% of journey")

            # Risk distribution
            risk_dist = predictions_df['risk_category'].value_counts(normalize=True) * 100
            print(f"\n📈 RISK DISTRIBUTION:")
            for category, percentage in risk_dist.items():
                print(f"{category.title()} Risk: {percentage:.1f}% of journey")

        # Driving Patterns
        print(f"\n🚗 DRIVING PATTERNS:")
        print(f"Average Speed: {df['speed'].mean():.1f} km/h")
        print(f"Maximum Speed: {df['speed'].max():.1f} km/h")
        print(f"Speed Variability (StdDev): {df['speed'].std():.1f} km/h")

        if 'acceleration' in df.columns:
            print(f"Average Acceleration: {df['acceleration'].mean():.3f} m/s²")
            print(f"Max Acceleration: {df['acceleration'].max():.3f} m/s²")
            print(f"Max Deceleration: {df['acceleration'].min():.3f} m/s²")

        # Engine Performance
        if 'rpm' in df.columns:
            print(f"\n🔧 ENGINE PERFORMANCE:")
            print(f"Average RPM: {df['rpm'].mean():.0f}")
            print(f"Maximum RPM: {df['rpm'].max():.0f}")

        if 'engine_temperature' in df.columns:
            print(f"Average Engine Temperature: {df['engine_temperature'].mean():.1f}°C")
            print(f"Maximum Engine Temperature: {df['engine_temperature'].max():.1f}°C")

        if 'engine_load_value' in df.columns:
            print(f"Average Engine Load: {df['engine_load_value'].mean():.1f}%")

        # Model Performance
        if training_results:
            print(f"\n🤖 MODEL PERFORMANCE:")
            print(f"Training AUC: {training_results.get('train_auc', 'N/A'):.4f}")
            print(f"Validation AUC: {training_results.get('val_auc', 'N/A'):.4f}")
            print(f"Best Iteration: {training_results.get('best_iteration', 'N/A')}")

        # Top Risk Factors
        if self.feature_importance is not None:
            print(f"\n🎯 TOP RISK FACTORS:")
            top_5_features = self.feature_importance.head(5)
            for i, (_, row) in enumerate(top_5_features.iterrows(), 1):
                print(f"{i}. {row['feature']}: {row['importance']:.0f}")

        # Generate Recommendations
        print(f"\n💡 RECOMMENDATIONS:")

        if predictions_df is not None:
            if high_risk_pct > 20:
                print("• ⚠️  High-risk behavior detected for significant portion of journey")
                print("  - Consider defensive driving training")
                print("  - Review aggressive acceleration and braking patterns")

            if max_risk > 0.8:
                print("• 🚨 Critical risk events detected")
                print("  - Immediate review of driving behavior needed")
                print("  - Consider driver coaching or intervention")

        if df['speed'].max() > 100:
            print("• 🏎️  Excessive speeding detected")
            print("  - Speed awareness training recommended")

        if 'engine_temperature' in df.columns and df['engine_temperature'].max() > 110:
            print("• 🌡️  Engine overheating detected")
            print("  - Vehicle maintenance check required")

        if 'acceleration' in df.columns and df['acceleration'].std() > 0.3:
            print("• 🔄 High acceleration variability")
            print("  - Focus on smoother driving techniques")

        print("="*80)

    def save_model(self, filepath: str):
        """Save the trained model and preprocessors"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'label_encoders': self.label_encoders,
            'risk_thresholds': self.risk_thresholds,
            'model_params': self.model_params
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a pre-trained model"""
        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data['feature_importance']
        self.label_encoders = model_data['label_encoders']
        self.risk_thresholds = model_data['risk_thresholds']
        self.model_params = model_data['model_params']

        logger.info(f"Model loaded from {filepath}")

    def analyze_risk_patterns(self, predictions_df: pd.DataFrame) -> Dict:
        """
        Analyze risk patterns and identify key insights

        Args:
            predictions_df: DataFrame with predictions

        Returns:
            Dictionary with pattern analysis results
        """
        if 'risk_probability' not in predictions_df.columns:
            raise ValueError("predictions_df must contain 'risk_probability' column")

        patterns = {}

        # Time-based patterns
        if 'timestamp' in predictions_df.columns:
            predictions_df['hour'] = predictions_df['timestamp'].dt.hour
            hourly_risk = predictions_df.groupby('hour')['risk_probability'].mean()
            patterns['peak_risk_hour'] = hourly_risk.idxmax()
            patterns['lowest_risk_hour'] = hourly_risk.idxmin()
            patterns['hourly_risk_variation'] = hourly_risk.std()

        # Speed-risk correlation
        if 'speed' in predictions_df.columns:
            speed_risk_corr = predictions_df['speed'].corr(predictions_df['risk_probability'])
            patterns['speed_risk_correlation'] = speed_risk_corr

            # Find speed ranges with highest risk
            predictions_df['speed_bin'] = pd.cut(predictions_df['speed'], bins=10)
            speed_risk_by_bin = predictions_df.groupby('speed_bin')['risk_probability'].mean()
            patterns['riskiest_speed_range'] = speed_risk_by_bin.idxmax()

        # Risk event clustering
        high_risk_threshold = self.risk_thresholds['medium']
        risk_events = predictions_df[predictions_df['risk_probability'] > high_risk_threshold]

        if len(risk_events) > 0:
            patterns['total_risk_events'] = len(risk_events)
            patterns['risk_event_duration_avg'] = self._calculate_event_duration(risk_events)
            patterns['max_consecutive_risk_seconds'] = self._find_max_consecutive_risk(predictions_df, high_risk_threshold)
        else:
            patterns['total_risk_events'] = 0
            patterns['risk_event_duration_avg'] = 0
            patterns['max_consecutive_risk_seconds'] = 0

        # Overall risk metrics
        patterns['overall_risk_score'] = predictions_df['risk_probability'].mean()
        patterns['risk_volatility'] = predictions_df['risk_probability'].std()
        patterns['high_risk_percentage'] = (predictions_df['risk_probability'] > high_risk_threshold).mean() * 100

        return patterns

    def _calculate_event_duration(self, risk_events_df: pd.DataFrame) -> float:
        """Calculate average duration of risk events"""
        if len(risk_events_df) == 0:
            return 0

        # Group consecutive events
        risk_events_df = risk_events_df.sort_index()
        diff = risk_events_df.index.to_series().diff()
        groups = (diff != 1).cumsum()

        event_durations = risk_events_df.groupby(groups).size()
        return event_durations.mean()

    def _find_max_consecutive_risk(self, df: pd.DataFrame, threshold: float) -> int:
        """Find maximum consecutive seconds of high risk"""
        high_risk_mask = df['risk_probability'] > threshold

        if not high_risk_mask.any():
            return 0

        # Find consecutive True values
        groups = (high_risk_mask != high_risk_mask.shift()).cumsum()
        consecutive_counts = high_risk_mask.groupby(groups).sum()

        return consecutive_counts.max() if len(consecutive_counts) > 0 else 0

    def create_interactive_dashboard(self, df: pd.DataFrame, predictions_df: pd.DataFrame = None):
        """
        Create interactive Plotly dashboard

        Args:
            df: Original data
            predictions_df: Predictions dataframe
        """
        if predictions_df is None:
            print("No predictions available. Run predict_risk() first.")
            return

        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Speed & Risk Over Time', 'Risk Distribution',
                          'Speed Distribution', 'Engine Performance Map',
                          'Acceleration Pattern', 'Risk Events Timeline'),
            specs=[[{"secondary_y": True}, {}],
                   [{}, {"type": "scatter"}],
                   [{}, {}]]
        )

        # Time series data
        time_minutes = np.arange(len(predictions_df)) / 60

        # 1. Speed and Risk over time
        fig.add_trace(
            go.Scatter(x=time_minutes, y=predictions_df['speed'],
                      name='Speed (km/h)', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time_minutes, y=predictions_df['risk_probability'],
                      name='Risk Probability', line=dict(color='red'), yaxis='y2'),
            row=1, col=1
        )

        # 2. Risk Distribution
        fig.add_trace(
            go.Histogram(x=predictions_df['risk_probability'],
                        name='Risk Distribution', nbinsx=30),
            row=1, col=2
        )

        # 3. Speed Distribution
        fig.add_trace(
            go.Histogram(x=df['speed'], name='Speed Distribution', nbinsx=30),
            row=2, col=1
        )

        # 4. Engine Performance Map (if available)
        if 'rpm' in df.columns and 'engine_temperature' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['rpm'], y=df['engine_temperature'],
                          mode='markers', name='Engine Performance',
                          marker=dict(color=df['speed'], colorscale='Viridis',
                                    colorbar=dict(title="Speed (km/h)"))),
                row=2, col=2
            )

        # 5. Acceleration Pattern
        if 'acceleration' in df.columns:
            fig.add_trace(
                go.Scatter(x=list(range(len(df))), y=df['acceleration'],
                          name='Acceleration', line=dict(color='green')),
                row=3, col=1
            )

        # 6. Risk Events Timeline
        high_risk_events = predictions_df[predictions_df['risk_probability'] > self.risk_thresholds['medium']]
        if len(high_risk_events) > 0:
            fig.add_trace(
                go.Scatter(x=high_risk_events.index / 60,
                          y=high_risk_events['risk_probability'],
                          mode='markers', name='High Risk Events',
                          marker=dict(color='red', size=8)),
                row=3, col=2
            )

        # Update layout
        fig.update_layout(
            height=900,
            title_text="Interactive Driver Risk Analysis Dashboard",
            showlegend=True
        )

        # Update axis labels
        fig.update_xaxes(title_text="Time (minutes)", row=1, col=1)
        fig.update_yaxes(title_text="Speed (km/h)", row=1, col=1)
        fig.update_yaxes(title_text="Risk Probability", secondary_y=True, row=1, col=1)

        fig.update_xaxes(title_text="Risk Probability", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)

        fig.update_xaxes(title_text="Speed (km/h)", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)

        if 'rpm' in df.columns and 'engine_temperature' in df.columns:
            fig.update_xaxes(title_text="RPM", row=2, col=2)
            fig.update_yaxes(title_text="Engine Temperature (°C)", row=2, col=2)

        if 'acceleration' in df.columns:
            fig.update_xaxes(title_text="Time (seconds)", row=3, col=1)
            fig.update_yaxes(title_text="Acceleration (m/s²)", row=3, col=1)

        fig.update_xaxes(title_text="Time (minutes)", row=3, col=2)
        fig.update_yaxes(title_text="Risk Probability", row=3, col=2)

        fig.show()

    def export_results(self, predictions_df: pd.DataFrame, filepath: str, format: str = 'csv'):
        """
        Export analysis results to file

        Args:
            predictions_df: DataFrame with predictions
            filepath: Output file path
            format: Export format ('csv', 'excel', 'json')
        """
        if format.lower() == 'csv':
            predictions_df.to_csv(filepath, index=False)
        elif format.lower() == 'excel':
            with pd.ExcelWriter(filepath) as writer:
                predictions_df.to_excel(writer, sheet_name='Predictions', index=False)
                if self.feature_importance is not None:
                    self.feature_importance.to_excel(writer, sheet_name='Feature_Importance', index=False)
        elif format.lower() == 'json':
            predictions_df.to_json(filepath, orient='records', date_format='iso')
        else:
            raise ValueError("Format must be 'csv', 'excel', or 'json'")

        logger.info(f"Results exported to {filepath}")
    def calculate_shap_contributions(self, df: pd.DataFrame) -> pd.DataFrame:
            """
            Calculate SHAP values to understand feature contributions to risk scores

            Args:
                df: DataFrame with features (same structure as training data)

            Returns:
                DataFrame with SHAP contributions for each feature at each time point
            """
            if self.model is None:
                raise ValueError("Model not trained yet. Call train_model() first.")

            print("🔍 Calculating SHAP contributions...")

            # Prepare features same way as training
            df_processed = self.load_and_preprocess_data(df)
            df_processed = self.create_temporal_features(df_processed)
            X, _ = self.prepare_features_for_training(df_processed)

            # Create SHAP explainer
            explainer = shap.TreeExplainer(self.model)

            # Calculate SHAP values (this might take a few minutes for large datasets)
            print("⏳ Computing SHAP values (this may take a moment)...")
            shap_values = explainer.shap_values(X)

            # Convert to DataFrame with proper column names
            shap_df = pd.DataFrame(shap_values, columns=self.feature_names)

            # Add base prediction and total prediction
            shap_df['base_prediction'] = explainer.expected_value
            shap_df['total_shap_contribution'] = shap_values.sum(axis=1)
            shap_df['final_risk_score'] = shap_df['base_prediction'] + shap_df['total_shap_contribution']

            # Add timestamp for reference
            shap_df['timestamp'] = pd.date_range(start='2024-01-01 09:00:00', periods=len(df), freq='1S')
            shap_df['seconds_elapsed'] = range(len(df))

            print(f"✅ SHAP analysis complete. Shape: {shap_df.shape}")
            return shap_df

    def analyze_feature_risk_contributions(self, shap_df: pd.DataFrame, top_n: int = 10) -> Dict:
            """
            Analyze which features contribute most to risk across the entire journey

            Args:
                shap_df: DataFrame with SHAP values
                top_n: Number of top contributing features to analyze

            Returns:
                Dictionary with contribution analysis
            """
            # Exclude metadata columns
            feature_cols = [col for col in shap_df.columns if col not in ['base_prediction', 'total_shap_contribution', 'final_risk_score', 'timestamp', 'seconds_elapsed']]

            analysis = {}

            # 1. Overall feature importance (average absolute SHAP values)
            avg_abs_contributions = shap_df[feature_cols].abs().mean().sort_values(ascending=False)
            analysis['top_risk_contributors'] = avg_abs_contributions.head(top_n).to_dict()

            # 2. Positive vs negative contributions
            avg_contributions = shap_df[feature_cols].mean().sort_values(ascending=False)
            analysis['most_risk_increasing'] = avg_contributions.head(top_n).to_dict()
            analysis['most_risk_decreasing'] = avg_contributions.tail(top_n).to_dict()

            # 3. Feature volatility (how much each feature's contribution varies)
            contribution_volatility = shap_df[feature_cols].std().sort_values(ascending=False)
            analysis['most_volatile_contributors'] = contribution_volatility.head(top_n).to_dict()

            # 4. Maximum single contributions
            max_contributions = shap_df[feature_cols].max().sort_values(ascending=False)
            analysis['max_single_contributions'] = max_contributions.head(top_n).to_dict()

            # 5. Summary statistics
            analysis['summary'] = {
                'base_risk_level': shap_df['base_prediction'].iloc[0],
                'avg_total_contribution': shap_df['total_shap_contribution'].mean(),
                'max_risk_spike': shap_df['final_risk_score'].max(),
                'min_risk_point': shap_df['final_risk_score'].min(),
                'risk_volatility': shap_df['final_risk_score'].std()
            }

            return analysis

    def create_rule_based_feature_scoring(self, df: pd.DataFrame) -> pd.DataFrame:
            """
            Create interpretable rule-based scoring for individual features

            Args:
                df: Original DataFrame

            Returns:
                DataFrame with rule-based feature risk scores
            """
            df_processed = self.load_and_preprocess_data(df)
            df_processed = self.create_temporal_features(df_processed)

            # Define risk thresholds for each feature type
            risk_rules = {
                'speed': {
                    'low': (0, 60),      # 0-60 km/h
                    'medium': (60, 90),  # 60-90 km/h
                    'high': (90, float('inf'))  # >90 km/h
                },
                'acceleration': {
                    'low': (-0.2, 0.2),      # Normal acceleration
                    'medium': (-0.4, 0.4),   # Moderate
                    'high': (float('-inf'), float('inf'))  # Aggressive
                },
                'rpm': {
                    'low': (0, 3000),
                    'medium': (3000, 4500),
                    'high': (4500, float('inf'))
                },
                'engine_temperature': {
                    'low': (0, 100),
                    'medium': (100, 110),
                    'high': (110, float('inf'))
                },
                'throttle_position': {
                    'low': (0, 30),
                    'medium': (30, 70),
                    'high': (70, 100)
                }
            }

            # Create rule-based scores
            rule_scores = pd.DataFrame(index=df_processed.index)
            rule_scores['timestamp'] = pd.date_range(start='2024-01-01 09:00:00', periods=len(df), freq='1S')

            for feature, thresholds in risk_rules.items():
                if feature in df_processed.columns:
                    values = df_processed[feature]

                    # Create risk scores (0=low, 1=medium, 2=high)
                    scores = np.zeros(len(values))
                    scores[(values >= thresholds['medium'][0]) & (values < thresholds['medium'][1])] = 1
                    scores[values >= thresholds['high'][0]] = 2
                    if feature == 'acceleration':  # Special case for acceleration (absolute value)
                        abs_values = np.abs(values)
                        scores = np.zeros(len(values))
                        scores[(abs_values >= 0.2) & (abs_values < 0.4)] = 1
                        scores[abs_values >= 0.4] = 2

                    rule_scores[f'{feature}_risk_score'] = scores
                    rule_scores[f'{feature}_value'] = values

                    # Add risk level labels
                    risk_labels = ['low', 'medium', 'high']
                    rule_scores[f'{feature}_risk_level'] = [risk_labels[int(score)] for score in scores]

            # Calculate composite rule-based risk score
            risk_columns = [col for col in rule_scores.columns if col.endswith('_risk_score')]
            if risk_columns:
                rule_scores['composite_rule_risk'] = rule_scores[risk_columns].mean(axis=1)
                rule_scores['composite_rule_level'] = pd.cut(
                    rule_scores['composite_rule_risk'],
                    bins=[0, 0.5, 1.5, 2],
                    labels=['low', 'medium', 'high']
                )

            return rule_scores

    def get_moment_by_moment_breakdown(self, shap_df: pd.DataFrame, rule_df: pd.DataFrame,
                                          high_risk_threshold: float = 0.7) -> pd.DataFrame:
            """
            Get detailed breakdown for each high-risk moment

            Args:
                shap_df: SHAP contributions DataFrame
                rule_df: Rule-based scores DataFrame
                high_risk_threshold: Threshold for considering a moment "high risk"

            Returns:
                DataFrame with detailed breakdowns for high-risk moments
            """
            # Find high-risk moments
            high_risk_moments = shap_df[shap_df['final_risk_score'] > high_risk_threshold].copy()

            if len(high_risk_moments) == 0:
                print(f"No moments found with risk score > {high_risk_threshold}")
                return pd.DataFrame()

            # Get feature columns (excluding metadata)
            feature_cols = [col for col in shap_df.columns if col not in ['base_prediction', 'total_shap_contribution', 'final_risk_score', 'timestamp', 'seconds_elapsed']]

            breakdown_data = []

            for idx, row in high_risk_moments.iterrows():
                moment_data = {
                    'timestamp': row['timestamp'],
                    'seconds_elapsed': row['seconds_elapsed'],
                    'total_risk_score': row['final_risk_score'],
                    'base_risk': row['base_prediction']
                }

                # Get top 5 SHAP contributors for this moment
                moment_contributions = row[feature_cols].sort_values(key=abs, ascending=False).head(5)

                for i, (feature, contribution) in enumerate(moment_contributions.items(), 1):
                    moment_data[f'top_{i}_feature'] = feature
                    moment_data[f'top_{i}_shap_contribution'] = contribution

                    # Add rule-based context if available
                    if f'{feature.split("_")[0]}_risk_level' in rule_df.columns:
                        rule_level = rule_df.loc[idx, f'{feature.split("_")[0]}_risk_level']
                        moment_data[f'top_{i}_rule_level'] = rule_level

                breakdown_data.append(moment_data)

            return pd.DataFrame(breakdown_data)

    def print_risk_summary(self, shap_analysis: Dict, moment_breakdown: pd.DataFrame):
            """
            Print interpretable summary of risk analysis

            Args:
                shap_analysis: Output from analyze_feature_risk_contributions
                moment_breakdown: Output from get_moment_by_moment_breakdown
            """
            print("="*80)
            print("DETAILED FEATURE RISK CONTRIBUTION ANALYSIS")
            print("="*80)

            print("\n🎯 TOP RISK-CONTRIBUTING BEHAVIORS:")
            for i, (feature, importance) in enumerate(shap_analysis['top_risk_contributors'].items(), 1):
                print(f"{i:2d}. {feature:30s}: {importance:+.4f}")

            print("\n📈 MOST RISK-INCREASING BEHAVIORS:")
            for i, (feature, contribution) in enumerate(shap_analysis['most_risk_increasing'].items(), 1):
                if contribution > 0:
                    print(f"{i:2d}. {feature:30s}: +{contribution:.4f}")

            print("\n📉 MOST RISK-DECREASING BEHAVIORS:")
            for i, (feature, contribution) in enumerate(list(shap_analysis['most_risk_decreasing'].items())[:5], 1):
                if contribution < 0:
                    print(f"{i:2d}. {feature:30s}: {contribution:.4f}")

            print(f"\n🚨 HIGH-RISK MOMENTS ANALYSIS:")
            print(f"Total high-risk moments identified: {len(moment_breakdown)}")

            if len(moment_breakdown) > 0:
                print(f"Peak risk score: {moment_breakdown['total_risk_score'].max():.3f}")
                print(f"Average risk during high-risk moments: {moment_breakdown['total_risk_score'].mean():.3f}")

                print(f"\n📊 TOP CONTRIBUTORS DURING HIGH-RISK MOMENTS:")
                # Count frequency of top contributors
                top_contributors = {}
                for i in range(1, 6):
                    col = f'top_{i}_feature'
                    if col in moment_breakdown.columns:
                        for feature in moment_breakdown[col].dropna():
                            top_contributors[feature] = top_contributors.get(feature, 0) + 1

                for feature, count in sorted(top_contributors.items(), key=lambda x: x[1], reverse=True)[:5]:
                    pct = (count / len(moment_breakdown)) * 100
                    print(f"   {feature:30s}: {count:3d} times ({pct:5.1f}%)")

            print("\n📈 SUMMARY STATISTICS:")
            summary = shap_analysis['summary']
            print(f"Base risk level: {summary['base_risk_level']:.3f}")
            print(f"Average feature contribution: {summary['avg_total_contribution']:+.3f}")
            print(f"Maximum risk spike: {summary['max_risk_spike']:.3f}")
            print(f"Risk volatility: {summary['risk_volatility']:.3f}")



# Usage Example and Main Analysis Function
def run_complete_analysis(df: pd.DataFrame, save_model: bool = True, model_path: str = 'driver_risk_model.pkl'):
    """
    Run complete driver risk analysis pipeline

    Args:
        df: Raw driving data DataFrame
        save_model: Whether to save the trained model
        model_path: Path to save/load model

    Returns:
        Tuple of (analyzer, predictions_df, training_results)
    """
    print("🚀 Starting Complete Driver Risk Analysis...")
    print("="*60)

    # Initialize analyzer
    analyzer = LightGBMDriverRiskAnalyzer()

    # Step 1: Data preprocessing
    print("\n📋 Step 1: Data Preprocessing...")
    df_processed = analyzer.load_and_preprocess_data(df)

    # Step 2: Feature engineering
    print("\n🔧 Step 2: Feature Engineering...")
    df_features = analyzer.create_temporal_features(df_processed)

    # Step 3: Create risk labels
    print("\n🎯 Step 3: Creating Risk Labels...")
    df_labeled = analyzer.create_risk_labels(df_features, method='composite')

    # Step 4: Train model
    print("\n🤖 Step 4: Training LightGBM Model...")
    training_results = analyzer.train_model(df_labeled)

    # Step 5: Make predictions
    print("\n🔮 Step 5: Making Risk Predictions...")
    predictions_df = analyzer.predict_risk(df)

    # Step 6: Analyze patterns
    print("\n📊 Step 6: Analyzing Risk Patterns...")
    risk_patterns = analyzer.analyze_risk_patterns(predictions_df)

    # Step 7: Generate visualizations
    print("\n📈 Step 7: Creating Visualizations...")
    analyzer.create_visualizations(df, predictions_df)

    # Step 8: Generate report
    print("\n📝 Step 8: Generating Analysis Report...")
    analyzer.generate_report(df, predictions_df, training_results)

    # Step 9: Create interactive dashboard
    print("\n🎛️  Step 9: Creating Interactive Dashboard...")
    analyzer.create_interactive_dashboard(df, predictions_df)

    # Step 10: Save model if requested
    if save_model:
        print(f"\n💾 Step 10: Saving Model to {model_path}...")
        analyzer.save_model(model_path)

    print("\n✅ Analysis Complete!")
    print("\n📋 Risk Pattern Summary:")
    for key, value in risk_patterns.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    return analyzer, predictions_df, training_results

def run_complete_feature_analysis(analyzer, df):
    """
    Run complete feature-level risk analysis

    Args:
        analyzer: Trained LightGBMDriverRiskAnalyzer instance
        df: Original DataFrame

    Returns:
        Tuple of (shap_df, rule_df, analysis_dict, breakdown_df)
    """
    print("🔍 Starting comprehensive feature risk analysis...")

    # 1. Calculate SHAP contributions
    shap_df = analyzer.calculate_shap_contributions(df)

    # 2. Analyze SHAP contributions
    shap_analysis = analyzer.analyze_feature_risk_contributions(shap_df)

    # 3. Create rule-based scoring
    rule_df = analyzer.create_rule_based_feature_scoring(df)

    # 4. Get moment-by-moment breakdown
    moment_breakdown = analyzer.get_moment_by_moment_breakdown(shap_df, rule_df)

    # 5. Print summary
    analyzer.print_risk_summary(shap_analysis, moment_breakdown)

    return shap_df, rule_df, shap_analysis, moment_breakdown

# Quick start function for Colab users
def quick_analysis(csv_file_path: str):
    """
    Quick analysis function for Colab users

    Args:
        csv_file_path: Path to CSV file

    Returns:
        Analysis results
    """


    print(f"📁 Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"📊 Data columns: {list(df.columns)}")

    # Run analysis
    return run_complete_analysis(df)


# Example usage in Google Colab:

# After uploading your CSV file to Colab:

#print("Method 1: Quick analysis")
#analyzer, predictions, results = quick_analysis('your_driving_data.csv')

print("Method 2: Step-by-step analysis")
analyzer, predictions, results = run_complete_analysis(df)

print("Method SHAP: Feature analysis")
shap_df, rule_df, shap_analysis, moment_breakdown = run_complete_feature_analysis(analyzer, df)

#print("Method 3: Custom analysis")
#analyzer = LightGBMDriverRiskAnalyzer()
#df_processed = analyzer.load_and_preprocess_data(df)
#df_features = analyzer.create_temporal_features(df_processed)
#df_labeled = analyzer.create_risk_labels(df_features)
#training_results = analyzer.train_model(df_labeled)
#predictions = analyzer.predict_risk(df)
#analyzer.generate_report(df, predictions, training_results)

# Export results
#analyzer.export_results(predictions, 'driver_risk_analysis_results.csv')
