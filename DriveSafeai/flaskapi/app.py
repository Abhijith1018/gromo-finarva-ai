from flask import Flask, request, jsonify, render_template_string, send_file
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import IsolationForest
from dotenv import load_dotenv
from scipy.signal import savgol_filter
import logging
from typing import Dict, List, Tuple ,Any
import warnings
import pickle
import os
from openai import OpenAI
import sqlite3
import io
import tempfile
from datetime import datetime
import json
import sys
import traceback
import folium
from folium.plugins import HeatMap, MarkerCluster, Fullscreen
import tempfile
import base64
from werkzeug.utils import secure_filename

warnings.filterwarnings('ignore')
load_dotenv()
OpenAI.api_key = os.getenv("OPENAI_API_KEY")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Map visualization imports
import folium
from folium.plugins import HeatMap, MarkerCluster

warnings.filterwarnings('ignore')
load_dotenv()
OpenAI.api_key = os.getenv("OPENAI_API_KEY")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/maps', exist_ok=True)


class MapVisualizer:
    """
    Map visualization component integrated with DriveSafeAI system
    """

    def __init__(self):
        """Initialize the map visualizer"""
        self.risk_color_map = {
            'low': 'green',
            'medium': 'orange',
            'high': 'red'
        }
        self.event_icons = {
            'hard_braking': 'stop',
            'sharp_turn': 'arrow-right',
            'speeding': 'flash',
            'high_risk': 'exclamation-triangle',
            'normal': 'info-sign'
        }

    def create_driving_route_map(self, df_with_predictions, save_path="static/maps/route_map.html"):
        """Create a map showing the complete driving route with risk analysis"""
        # Validate required columns
        required_cols = ['latitude', 'longitude', 'risk_score', 'risk_category']
        missing_cols = [col for col in required_cols if col not in df_with_predictions.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Clean data - remove rows with invalid GPS coordinates
        df_clean = df_with_predictions.dropna(subset=['latitude', 'longitude'])
        df_clean = df_clean[(df_clean['latitude'] != 0) & (df_clean['longitude'] != 0)]

        if len(df_clean) == 0:
            raise ValueError("No valid GPS coordinates found in the data")

        # Calculate map center
        center_lat = df_clean['latitude'].mean()
        center_lon = df_clean['longitude'].mean()

        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=13,
            tiles='OpenStreetMap'
        )

        # Add route line (connect GPS points in chronological order)
        if len(df_clean) > 1:
            route_coords = df_clean[['latitude', 'longitude']].values.tolist()

            # Color-code route segments by risk level
            for i in range(len(route_coords) - 1):
                start_point = route_coords[i]
                end_point = route_coords[i + 1]
                risk_level = df_clean.iloc[i]['risk_category']

                folium.PolyLine(
                    locations=[start_point, end_point],
                    color=self.risk_color_map.get(risk_level, 'blue'),
                    weight=4,
                    opacity=0.7
                ).add_to(m)

        # Add start and end markers
        start_point = df_clean.iloc[0]
        end_point = df_clean.iloc[-1]

        folium.Marker(
            location=[start_point['latitude'], start_point['longitude']],
            popup="Trip Start",
            icon=folium.Icon(color='green', icon='play', prefix='fa')
        ).add_to(m)

        folium.Marker(
            location=[end_point['latitude'], end_point['longitude']],
            popup="Trip End",
            icon=folium.Icon(color='black', icon='stop', prefix='fa')
        ).add_to(m)

        # Add risk event markers for high-risk points
        high_risk_points = df_clean[df_clean['risk_score'] >= 6.0]

        for idx, point in high_risk_points.iterrows():
            popup_content = self._create_risk_popup(point)

            folium.Marker(
                location=[point['latitude'], point['longitude']],
                popup=folium.Popup(popup_content, max_width=300),
                tooltip=f"Risk Score: {point['risk_score']:.2f}",
                icon=folium.Icon(
                    color=self.risk_color_map.get(point['risk_category'], 'blue'),
                    icon='exclamation-triangle',
                    prefix='fa'
                )
            ).add_to(m)

        # Add legend and trip statistics
        self._add_legend(m)
        self._add_trip_stats(m, df_clean)

        # Save map
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        m.save(save_path)

        return m, save_path

    def create_risk_heatmap(self, df_with_predictions, save_path="static/maps/heatmap.html"):
        """Create a heatmap showing risk concentration areas"""
        df_clean = df_with_predictions.dropna(subset=['latitude', 'longitude', 'risk_score'])

        if len(df_clean) == 0:
            raise ValueError("No valid GPS coordinates found in the data")

        # Create base map
        center_lat = df_clean['latitude'].mean()
        center_lon = df_clean['longitude'].mean()

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles='OpenStreetMap'
        )

        # Prepare heatmap data - weight by risk score
        heat_data = []
        for idx, row in df_clean.iterrows():
            # Use risk score as weight (higher risk = more intense heat)
            weight = max(0.1, row['risk_score'] / 10.0)  # Normalize to 0-1 range
            heat_data.append([row['latitude'], row['longitude'], weight])

        # Add heatmap
        HeatMap(
            heat_data,
            min_opacity=0.3,
            max_zoom=18,
            radius=15,
            blur=10,
            gradient={0.2: 'green', 0.5: 'yellow', 0.8: 'orange', 1.0: 'red'}
        ).add_to(m)

        # Save map
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        m.save(save_path)

        return m, save_path

    def _create_risk_popup(self, point):
        """Create detailed popup content for risk events"""
        popup_content = f"""
        <div style="font-family: Arial; font-size: 12px;">
        <b>🚨 Risk Event</b><br>
        <b>Risk Score:</b> {point['risk_score']:.2f}/10<br>
        <b>Risk Level:</b> {point['risk_category']}<br>
        <b>Drive Score:</b> {point.get('drive_score', 'N/A')}<br>
        """

        if 'timestamp' in point:
            popup_content += f"<b>Time:</b> {point['timestamp']}<br>"
        if 'speed' in point:
            popup_content += f"<b>Speed:</b> {point['speed']:.1f} km/h<br>"
        if 'acceleration' in point:
            popup_content += f"<b>Acceleration:</b> {point['acceleration']:.2f} m/s²<br>"
        if 'rpm' in point:
            popup_content += f"<b>RPM:</b> {point['rpm']:.0f}<br>"
        if 'engine_temperature' in point:
            popup_content += f"<b>Engine Temp:</b> {point['engine_temperature']:.1f}°C<br>"

        popup_content += "</div>"
        return popup_content

    def _add_legend(self, map_obj):
        """Add legend to the map"""
        legend_html = '''
        <div style="position: fixed;
                    bottom: 50px; left: 50px; width: 150px; height: 120px;
                    background-color: white; border:2px solid grey; z-index:9999;
                    font-size:12px; padding: 10px">
        <p><b>🚗 DriveSafeAI Risk Levels</b></p>
        <p><i class="fa fa-circle" style="color:red"></i> High Risk (6-10)</p>
        <p><i class="fa fa-circle" style="color:orange"></i> Medium Risk (3-6)</p>
        <p><i class="fa fa-circle" style="color:green"></i> Low Risk (0-3)</p>
        <p><i class="fa fa-play" style="color:green"></i> Trip Start</p>
        <p><i class="fa fa-stop" style="color:black"></i> Trip End</p>
        </div>
        '''
        map_obj.get_root().html.add_child(folium.Element(legend_html))

    def _add_trip_stats(self, map_obj, df):
        """Add trip statistics to the map"""
        avg_risk = df['risk_score'].mean()
        max_risk = df['risk_score'].max()
        avg_drive_score = df.get('drive_score', pd.Series([0])).mean()

        stats_html = f'''
        <div style="position: fixed;
                    top: 10px; right: 10px; width: 200px; height: 100px;
                    background-color: white; border:2px solid grey; z-index:9999;
                    font-size:12px; padding: 10px">
        <p><b>📊 Trip Statistics</b></p>
        <p>Average Risk: {avg_risk:.2f}/10</p>
        <p>Max Risk: {max_risk:.2f}/10</p>
        <p>Drive Score: {avg_drive_score:.1f}/100</p>
        <p>Duration: {len(df)} seconds</p>
        </div>
        '''
        map_obj.get_root().html.add_child(folium.Element(stats_html))


class LightGBMDriverRiskAnalyzer:
    def __init__(self, model_params: Dict = None):
        """
        Initialize LightGBM-based driver risk analyzer for per-second telemetry data
        """
        self.model_params = model_params or {
            'objective': 'regression',
            'metric': 'rmse',
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

        # Store exact training features and statistics
        self.training_features = None
        self.feature_statistics = {}
        self.feature_names = None

        self.risk_thresholds = {'low': 3, 'medium': 6, 'high': 10}
        self.is_trained = False

    def load_and_preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Load and preprocess the per-second driving data"""
        df = df.copy()

        # Add timestamp column based on index (assuming per-second data)
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.date_range(start='2024-01-01 09:00:00', periods=len(df), freq='1S')
        if 'seconds_elapsed' not in df.columns:
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
        """Create time-series features from per-second data - ROBUST VERSION"""
        df = df.copy()

        # Define window sizes for rolling features
        windows = [5, 10, 30, 60]  # 5s, 10s, 30s, 60s windows

        # Core telemetry columns for feature engineering
        core_features = ['speed', 'rpm', 'acceleration', 'throttle_position',
                         'engine_temperature', 'system_voltage', 'engine_load_value']

        # ROBUST FEATURE CREATION: Only create features for available columns
        available_core_features = [f for f in core_features if f in df.columns]
        logger.info(f"Available core features: {available_core_features}")

        for feature in available_core_features:
            try:
                # Rolling statistics - only if we have enough data
                for window in windows:
                    if len(df) >= window:
                        df[f'{feature}_mean_{window}s'] = df[feature].rolling(window=window, min_periods=1).mean()
                        df[f'{feature}_std_{window}s'] = df[feature].rolling(window=window, min_periods=1).std().fillna(
                            0)
                        df[f'{feature}_max_{window}s'] = df[feature].rolling(window=window, min_periods=1).max()
                        df[f'{feature}_min_{window}s'] = df[feature].rolling(window=window, min_periods=1).min()

                # Lag features (previous values) - only if we have enough data
                for lag in [1, 5, 10]:
                    if len(df) > lag:
                        df[f'{feature}_lag_{lag}s'] = df[feature].shift(lag).fillna(df[feature].iloc[0])

                # Rate of change
                df[f'{feature}_rate_change'] = df[feature].diff().fillna(0)
                df[f'{feature}_rate_change_abs'] = np.abs(df[f'{feature}_rate_change'])

            except Exception as e:
                logger.warning(f"Error creating features for {feature}: {e}")
                continue

        # Smooth core features using Savitzky-Golay filter
        smooth_features = ['speed', 'acceleration', 'rpm']
        for feature in smooth_features:
            if feature in df.columns and len(df) > 11:
                try:
                    window_length = min(11, len(df) // 2 * 2 + 1)
                    if window_length >= 3:
                        df[f'{feature}_smooth'] = savgol_filter(df[feature], window_length=window_length, polyorder=2)
                    else:
                        df[f'{feature}_smooth'] = df[feature]
                except Exception as e:
                    logger.warning(f"Could not smooth {feature}: {e}")
                    df[f'{feature}_smooth'] = df[feature]

        # Composite features - only create if base features exist
        if 'speed' in df.columns and 'acceleration' in df.columns:
            df['speed_accel_product'] = df['speed'] * np.abs(df['acceleration'])

        if 'rpm' in df.columns and 'speed' in df.columns:
            df['rpm_speed_ratio'] = np.where(df['speed'] > 5, df['rpm'] / df['speed'], 0)

        # Driving efficiency metrics
        if 'engine_load_value' in df.columns and 'speed' in df.columns:
            df['fuel_efficiency_proxy'] = df['engine_load_value'] / (df['speed'] + 1)

        if 'rpm' in df.columns and 'engine_load_value' in df.columns:
            df['engine_stress'] = (df['rpm'] / 6000) * (df['engine_load_value'] / 100)

        # Biometric features - only if available
        if 'heart_rate' in df.columns and 'body_temperature' in df.columns:
            df['biometric_stress'] = ((df['heart_rate'] - 70) / 50) + ((df['body_temperature'] - 36.5) / 1.5)
            df['high_stress'] = (df['biometric_stress'] > df['biometric_stress'].quantile(0.8)).astype(int)
        elif 'heart_rate' in df.columns:
            df['biometric_stress'] = (df['heart_rate'] - 70) / 50
            df['high_stress'] = (df['heart_rate'] > df['heart_rate'].quantile(0.8)).astype(int)

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
        """Create risk labels for training based on driving behavior patterns"""
        df = df.copy()

        if method == 'composite':
            # Multi-factor risk scoring (0-10 scale)
            risk_score = np.zeros(len(df))

            # Speed-based risk (0-3 points)
            if 'speed' in df.columns:
                risk_score += np.where(df['speed'] > 100, 3,
                                       np.where(df['speed'] > 80, 2,
                                                np.where(df['speed'] > 60, 1, 0)))

            # Acceleration-based risk (0-3 points)
            if 'acceleration' in df.columns:
                risk_score += np.where(np.abs(df['acceleration']) > 0.5, 3,
                                       np.where(np.abs(df['acceleration']) > 0.3, 2,
                                                np.where(np.abs(df['acceleration']) > 0.2, 1, 0)))

            # Engine stress risk (0-2 points)
            if 'rpm' in df.columns:
                risk_score += np.where(df['rpm'] > 5000, 2,
                                       np.where(df['rpm'] > 4000, 1, 0))

            # Temperature risk (0-2 points)
            if 'engine_temperature' in df.columns:
                risk_score += np.where(df['engine_temperature'] > 110, 2,
                                       np.where(df['engine_temperature'] > 100, 1, 0))

            # Volatility risk (0-1 point)
            if 'speed_std_30s' in df.columns:
                speed_volatility_threshold = df['speed_std_30s'].quantile(0.8)
                risk_score += (df['speed_std_30s'] > speed_volatility_threshold).astype(int)

            # Biometric risk (0-2 points)
            if 'high_stress' in df.columns:
                risk_score += df['high_stress'] * 2

            # Cap at 10 and store raw score
            df['risk_score_raw'] = np.clip(risk_score, 0, 10)

        # Create binary high-risk flag using proper thresholds
        df['is_high_risk'] = (df['risk_score_raw'] >= self.risk_thresholds['medium']).astype(int)

        # Create multi-class risk levels using proper thresholds
        df['risk_level'] = pd.cut(df['risk_score_raw'],
                                  bins=[-np.inf, self.risk_thresholds['low'],
                                        self.risk_thresholds['medium'], np.inf],
                                  labels=['low', 'medium', 'high'])

        risk_distribution = df['is_high_risk'].value_counts(normalize=True)
        logger.info(f"Risk label distribution: {risk_distribution.to_dict()}")
        logger.info(f"Risk score statistics: mean={df['risk_score_raw'].mean():.2f}, "
                    f"std={df['risk_score_raw'].std():.2f}, max={df['risk_score_raw'].max():.2f}")

        return df

    def prepare_features_for_training(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[
        pd.DataFrame, List[str]]:
        """Prepare features for LightGBM training/prediction with ROBUST feature alignment"""
        # Exclude non-feature columns
        exclude_columns = [
            'timestamp', 'is_high_risk', 'risk_level', 'risk_score_raw',
            'id_vehicle', 'id_driver', 'latitude', 'longitude'
        ]

        feature_columns = [col for col in df.columns if col not in exclude_columns]

        # Handle categorical variables (none in this case, but keeping structure)
        categorical_features = []

        for cat_feature in categorical_features:
            if cat_feature in feature_columns:
                if cat_feature not in self.label_encoders:
                    if is_training:
                        self.label_encoders[cat_feature] = LabelEncoder()
                        df[cat_feature] = self.label_encoders[cat_feature].fit_transform(df[cat_feature].astype(str))
                    else:
                        logger.warning(f"Label encoder for {cat_feature} not found during prediction")
                        continue
                else:
                    try:
                        df[cat_feature] = self.label_encoders[cat_feature].transform(df[cat_feature].astype(str))
                    except ValueError as e:
                        logger.error(f"Error encoding {cat_feature}: {e}")
                        raise

        # Select initial features
        X = df[feature_columns].copy()

        if is_training:
            # TRAINING MODE: Create and store training features
            logger.info(f"🏋️ TRAINING MODE: Processing {X.shape[1]} initial features")

            # Remove any columns with all NaN or constant values
            initial_cols = X.shape[1]
            X = X.loc[:, X.std() != 0]
            X = X.dropna(axis=1, how='all')
            logger.info(f"   Removed {initial_cols - X.shape[1]} constant/empty columns")

            # Fill any remaining NaN values
            X = X.fillna(X.median())

            # Remove highly correlated features
            if X.shape[1] > 1:
                corr_matrix = X.corr().abs()
                upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                high_corr_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
                X = X.drop(columns=high_corr_features)
                logger.info(f"   Removed {len(high_corr_features)} highly correlated features")

            # STORE TRAINING FEATURES AND STATISTICS
            self.training_features = list(X.columns)
            self.feature_names = self.training_features
            self.feature_statistics = {
                'means': X.mean().to_dict(),
                'medians': X.median().to_dict(),
                'stds': X.std().to_dict(),
                'mins': X.min().to_dict(),
                'maxs': X.max().to_dict()
            }

            logger.info(f"✅ TRAINING: Final feature set has {len(self.training_features)} features")

        else:
            # PREDICTION MODE: Align with training features
            logger.info(f"🔮 PREDICTION MODE: Processing {X.shape[1]} initial features")

            if self.training_features is None:
                raise ValueError("Model not trained yet. training_features not available.")

            # Fill NaN values first
            X = X.fillna(X.median())

            # Ensure we have exactly the training features
            missing_features = set(self.training_features) - set(X.columns)
            extra_features = set(X.columns) - set(self.training_features)

            logger.info(f"🔍 Feature alignment analysis:")
            logger.info(f"   Training features: {len(self.training_features)}")
            logger.info(f"   Current features: {len(X.columns)}")
            logger.info(f"   Missing features: {len(missing_features)}")
            logger.info(f"   Extra features: {len(extra_features)}")

            # Add missing features with default values
            for feature in missing_features:
                if feature in self.feature_statistics['medians']:
                    default_value = self.feature_statistics['medians'][feature]
                    logger.info(f"   ➕ Adding missing feature '{feature}' with median: {default_value:.4f}")
                    X[feature] = default_value
                else:
                    logger.warning(f"   ⚠️ Adding missing feature '{feature}' with value 0")
                    X[feature] = 0

            # Remove extra features
            if extra_features:
                logger.info(f"   ➖ Removing {len(extra_features)} extra features")
                X = X.drop(columns=list(extra_features))

            # Ensure correct column order
            X = X[self.training_features]

            logger.info(f"✅ PREDICTION: Successfully aligned to {len(X.columns)} features")

        feature_names = list(X.columns)
        return X, feature_names

    def train_model(self, df: pd.DataFrame, validation_split: float = 0.2) -> Dict:
        """Train LightGBM model on the prepared data"""
        logger.info("🚀 Starting model training...")

        # Prepare features with training flag
        X, feature_names = self.prepare_features_for_training(df, is_training=True)
        y = df['risk_score_raw'].values

        # Time-based split to avoid data leakage
        split_index = int(len(df) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_val = y[:split_index], y[split_index:]

        logger.info(f"📊 Training set: {len(X_train)} samples, Validation set: {len(X_val)} samples")

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

        # Mark as trained
        self.is_trained = True

        # Get predictions
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)

        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        train_r2 = r2_score(y_train, train_pred)
        val_r2 = r2_score(y_val, val_pred)

        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)

        results = {
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'best_iteration': self.model.best_iteration,
            'feature_importance': self.feature_importance,
            'training_features': self.training_features
        }

        logger.info(f"✅ Training completed successfully!")
        logger.info(f"📈 Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}")
        logger.info(f"📈 Train R²: {train_r2:.4f}, Val R²: {val_r2:.4f}")
        logger.info(f"🎯 Model trained on {len(self.training_features)} features")

        return results

    def predict_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict risk scores for new data with ROBUST feature handling"""
        if not self.is_trained:
            raise ValueError("❌ Model not trained yet. Call train_model() first or load a trained model.")

        logger.info(f"🔮 Starting prediction on {len(df)} data points...")

        # Apply same preprocessing pipeline
        df_processed = self.load_and_preprocess_data(df)
        df_processed = self.create_temporal_features(df_processed)

        # Prepare features with prediction flag - this handles feature alignment
        X, actual_features = self.prepare_features_for_training(df_processed, is_training=False)

        logger.info(f"✅ Feature alignment successful. Using {len(actual_features)} features for prediction.")

        # Get risk scores
        risk_scores = self.model.predict(X)

        # Ensure scores are within valid range
        risk_scores = np.clip(risk_scores, 0, 10)

        # Create risk categories using proper thresholds
        risk_categories = pd.cut(risk_scores,
                                 bins=[0, self.risk_thresholds['low'],
                                       self.risk_thresholds['medium'],
                                       self.risk_thresholds['high']],
                                 labels=['low', 'medium', 'high'])

        # Create result DataFrame
        result_df = df.copy()
        result_df['risk_score'] = risk_scores
        result_df['risk_category'] = risk_categories
        result_df['driveScore'] = 100 - 10 * risk_scores

        # Add timestamp if not present
        if 'timestamp' not in result_df.columns:
            result_df['timestamp'] = pd.date_range(start='2024-01-01 09:00:00', periods=len(df), freq='1S')

        logger.info(f"🎯 Prediction completed successfully!")
        logger.info(f"📊 Average risk score: {risk_scores.mean():.2f}")
        logger.info(f"📊 Risk distribution: {pd.Series(risk_categories).value_counts().to_dict()}")

        return result_df

    def train_and_save(self, combined_df: pd.DataFrame, model_save_path: str, validation_split: float = 0.2):
        """Complete training workflow: preprocess -> create features -> create labels -> train -> save"""
        logger.info("🔄 Starting complete training workflow...")

        # Full preprocessing pipeline
        df_processed = self.load_and_preprocess_data(combined_df)
        df_processed = self.create_temporal_features(df_processed)
        df_processed = self.create_risk_labels(df_processed)

        # Train model
        results = self.train_model(df_processed, validation_split)

        logger.info("✅ Complete training workflow finished!")
        return results


# Initialize the analyzer globally
analyzer = None
HARDCODED_API_KEY = "https://www.google.com/search?client=safari&rls=en&q=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&ie=UTF-8&oe=UTF-8"
client = OpenAI(
    api_key=HARDCODED_API_KEY)
def initialize_model():
    """Initialize and train the model on startup"""
    global analyzer

    try:
        # Load the datasets
        df = pd.read_csv("C:/Users/Abhijith/OneDrive/Desktop/DF1.csv")
        df2 = pd.read_csv("C:/Users/Abhijith/OneDrive/Desktop/DF2.csv")
        df3 = pd.read_csv("C:/Users/Abhijith/OneDrive/Desktop/DF3.csv")
        df4 = pd.read_csv("C:/Users/Abhijith/OneDrive/Desktop/DF4.csv")
        df5 = pd.read_csv("C:/Users/Abhijith/OneDrive/Desktop/DF5.csv")

        # Combine datasets
        combined_df = pd.concat([df, df2, df3, df4], ignore_index=True)

        # Initialize analyzer
        analyzer = LightGBMDriverRiskAnalyzer()

        # Train the model
        training_results = analyzer.train_and_save(combined_df, "my_driver_risk_model", validation_split=0.2)

        logger.info("✅ Model trained and ready for predictions!")
        logger.info(
            f"📊 Training Results - Val RMSE: {training_results['val_rmse']:.4f}, Val R²: {training_results['val_r2']:.4f}")

    except Exception as e:
        logger.error(f"❌ Error during model initialization: {e}")
        raise

def convert_to_serializable(obj):
    """Convert pandas/numpy data types to JSON serializable types"""
    import numpy as np
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):  # numpy scalars
        return obj.item()
    return obj


def calculate_trip_metrics(trip_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate comprehensive trip metrics with safety checks."""
    metrics = {
        "duration": int(len(trip_df)),
        "avg_speed": float(trip_df['speed'].mean()) if 'speed' in trip_df else 0.0,
        "max_speed": float(trip_df['speed'].max()) if 'speed' in trip_df else 0.0,
        "avg_risk": float(trip_df['risk_score'].mean()) if 'risk_score' in trip_df else 0.0,
        "max_risk": float(trip_df['risk_score'].max()) if 'risk_score' in trip_df else 0.0,
        "avg_drive_score": float(trip_df['drive_score'].mean()) if 'drive_score' in trip_df else 0.0,
        "max_rpm": int(trip_df['rpm'].max()) if 'rpm' in trip_df else 0,
        "min_visibility": float(trip_df['visibility'].min()) if 'visibility' in trip_df else 100.0,
        "precipitation_time": int((trip_df['has_precipitation'] == 1).sum()) if 'has_precipitation' in trip_df else 0,
        "harsh_braking_events": int(len(trip_df[trip_df['acceleration'] < -3])) if 'acceleration' in trip_df else 0,
        "rapid_acceleration_events": int(len(trip_df[trip_df['acceleration'] > 2])) if 'acceleration' in trip_df else 0,
        "high_throttle_events": int(len(trip_df[trip_df['throttle_position'] > 80])) if 'throttle_position' in trip_df else 0,
    }
    return metrics


def test_openai_connection():
    """Test OpenAI API connection with proper error handling"""
    print("🔍 Testing OpenAI API connection...")

    # Check API key from environment only
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        print("💡 Set it with: export OPENAI_API_KEY='your-key-here'")
        return False

    if not api_key.startswith('sk-'):
        print(f"❌ Invalid API key format. Should start with 'sk-'")
        return False

    print(f"✅ Found API key (length: {len(api_key)})")

    try:
        # Initialize client with environment variable only
        client = OpenAI()  # Will automatically use OPENAI_API_KEY env var
        print("✅ OpenAI client initialized")

        # Test API call
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'API working!'"}],
            max_tokens=10
        )

        result = response.choices[0].message.content
        print(f"✅ OpenAI API working! Response: {result}")
        return True

    except Exception as e:
        print(f"❌ OpenAI API error: {e}")
        if "authentication" in str(e).lower():
            print("💡 Check your API key is valid and has billing enabled")
        return False


# Updated version of the feedback function with better debugging
def llm_generate_feedback_and_tags_debug(trip_df: pd.DataFrame) -> Dict[str, Any]:
    """Generate AI-powered feedback with detailed debugging."""
    print("🚀 Starting AI feedback generation...")

    # Check API key first
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ No OpenAI API key found")
        return {"error": "No API key", "ai_generated": False}

    print(f"✅ API key found (length: {len(api_key)})")

    try:
        # Initialize client
        client = OpenAI(api_key=api_key)
        print("✅ OpenAI client initialized")

        # Calculate comprehensive metrics
        metrics = calculate_trip_metrics(trip_df)
        print(f"✅ Metrics calculated: {metrics}")

        # Get sample events for context
        sample_columns = ['speed', 'rpm', 'acceleration', 'throttle_position']
        available_columns = [col for col in sample_columns if col in trip_df.columns]
        print(f"📊 Available columns: {available_columns}")

        if available_columns:
            sample_events = trip_df[available_columns].head(3).to_dict(orient='records')
        else:
            sample_events = []

        # Create prompt
        prompt = f"""
Analyze this driving trip data and provide gamified feedback.

TRIP METRICS:
- Duration: {metrics['duration']} seconds
- Speed: Avg {metrics['avg_speed']:.1f} km/h, Max {metrics['max_speed']:.1f} km/h
- Risk Score: Avg {metrics['avg_risk']:.2f}, Max {metrics['max_risk']:.2f}
- Drive Score: {metrics['avg_drive_score']:.2f}
- Harsh braking: {metrics['harsh_braking_events']} events
- Rapid acceleration: {metrics['rapid_acceleration_events']} events

Respond with ONLY valid JSON:
{{"incidents": ["safety concerns"], "stars": 4, "badges": ["badges"], "feedback": "message", "recommendations": ["tips"]}}
"""

        print("📝 Prompt created, making API call...")

        # Make API call
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Use cheaper model first
            messages=[
                {"role": "system", "content": "You are a driving coach. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=400
        )

        print("✅ API call successful!")

        content = response.choices[0].message.content.strip()
        print(f"📄 Raw response: {content}")

        # Clean up formatting
        if content.startswith('```json'):
            content = content[7:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()

        parsed_response = json.loads(content)
        parsed_response['ai_generated'] = True

        print("✅ Successfully generated AI feedback!")
        return parsed_response

    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        print(f"❌ Raw content: {content if 'content' in locals() else 'No content'}")
        return {"error": "JSON parse failed", "ai_generated": False}
    except Exception as e:
        print(f"❌ AI feedback failed: {e}")
        print(f"❌ Error type: {type(e).__name__}")
        return {"error": f"AI error: {str(e)}", "ai_generated": False}


# Alternative function using environment variable check
def llm_generate_feedback_and_tags_with_fallback(trip_df: pd.DataFrame) -> Dict[str, Any]:
    """Generate AI-powered feedback with automatic fallback to rule-based system."""

    # Check if OpenAI API key is properly configured
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.startswith("sk-proj-") == False:
        print("⚠️ OpenAI API key not properly configured, using fallback feedback")
        return generate_fallback_feedback(trip_df)

    try:
        # Initialize client with API key
        client = OpenAI(api_key=api_key)

        # Calculate comprehensive metrics
        metrics = calculate_trip_metrics(trip_df)

        # Get sample events for context
        sample_columns = ['speed', 'rpm', 'acceleration', 'throttle_position']
        available_columns = [col for col in sample_columns if col in trip_df.columns]

        if available_columns:
            sample_events = trip_df[available_columns].head(3).to_dict(orient='records')
        else:
            sample_events = []

        # Create prompt
        prompt = f"""
Analyze this driving trip data and provide gamified feedback as a driving coach.

TRIP METRICS:
- Duration: {metrics['duration']} seconds
- Speed: Avg {metrics['avg_speed']:.1f} km/h, Max {metrics['max_speed']:.1f} km/h
- Risk Score: Avg {metrics['avg_risk']:.2f}, Max {metrics['max_risk']:.2f}
- Drive Score: {metrics['avg_drive_score']:.2f}
- Harsh braking: {metrics['harsh_braking_events']} events
- Rapid acceleration: {metrics['rapid_acceleration_events']} events
- High throttle: {metrics['high_throttle_events']} events

Respond with ONLY valid JSON in this exact format:
{{"incidents": ["list of safety concerns"], "stars": 4, "badges": ["achievement badges"], "feedback": "encouraging message with emojis", "recommendations": ["specific tips"]}}
"""

        # Make API call
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a driving coach. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=400
        )

        content = response.choices[0].message.content.strip()

        # Clean up formatting
        if content.startswith('```json'):
            content = content[7:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()

        parsed_response = json.loads(content)
        parsed_response['ai_generated'] = True

        print("✅ Successfully generated AI feedback")
        return parsed_response

    except Exception as e:
        print(f"❌ AI feedback failed: {e}")
        print("🔄 Falling back to rule-based feedback")
        fallback = generate_fallback_feedback(trip_df)
        fallback['ai_generated'] = False
        return fallback


def generate_fallback_feedback(trip_df: pd.DataFrame) -> Dict[str, Any]:
    """Generate basic feedback when AI fails."""
    metrics = calculate_trip_metrics(trip_df)

    # Simple rule-based feedback
    incidents = []
    if metrics['harsh_braking_events'] > 3:
        incidents.append("Frequent harsh braking detected")
    if metrics['rapid_acceleration_events'] > 2:
        incidents.append("Multiple rapid acceleration events")
    if metrics['max_speed'] > 120:
        incidents.append("Excessive speeding detected")

    # Simple star rating
    stars = 5
    if metrics['avg_risk'] > 0.7:
        stars -= 2
    elif metrics['avg_risk'] > 0.4:
        stars -= 1

    if metrics['harsh_braking_events'] > 2:
        stars -= 1

    stars = max(1, min(5, stars))

    badges = ["Safe Driver"] if stars >= 4 else ["Improving Driver"]

    return {
        "incidents": incidents,
        "stars": stars,
        "badges": badges,
        "feedback": f"Trip completed with {stars}/5 stars! {'Great job maintaining safe driving habits! 🌟' if stars >= 4 else 'Focus on smoother driving for better scores. 🚗'}",
        "recommendations": ["Practice gradual acceleration", "Maintain safe following distance", "Monitor speed limits"]
    }


def process_trip_with_feedback(trip_df: pd.DataFrame) -> Dict[str, Any]:
    """Process a single trip with AI feedback and fallback handling."""
    trip_id = trip_df['trip_id'].iloc[0] if 'trip_id' in trip_df else 'unknown'

    # Try AI generation first
    result = llm_generate_feedback_and_tags_debug(trip_df)

    # Use fallback if AI fails
    if "error" in result:
        logger.warning(f"AI generation failed for trip {trip_id}, using fallback")
        result = generate_fallback_feedback(trip_df)

    # Add metadata - FIX SERIALIZATION HERE
    result.update({
        "trip_id": convert_to_serializable(trip_id),
        "processed_at": datetime.now().isoformat(),
        "duration_minutes": convert_to_serializable(len(trip_df) / 60),
    })

    return result

class MapVisualizer:
    """Map visualization component"""

    def __init__(self):
        self.risk_color_map = {
            'low': 'green',
            'medium': 'orange',
            'high': 'red'
        }
        self.event_icons = {
            'hard_braking': 'stop',
            'sharp_turn': 'arrow-right',
            'speeding': 'flash',
            'high_risk': 'exclamation-triangle',
            'normal': 'info-sign'
        }

    def load_and_process_csv(self, csv_path):
        """Load CSV and add risk analysis columns"""
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded CSV with {len(df)} rows and columns: {list(df.columns)}")

            # Check for required columns
            if 'latitude' not in df.columns or 'longitude' not in df.columns:
                raise ValueError("CSV must contain 'latitude' and 'longitude' columns")

            # Clean data
            df_clean = df.dropna(subset=['latitude', 'longitude'])
            df_clean = df_clean[(df_clean['latitude'] != 0) & (df_clean['longitude'] != 0)]
            df_clean = df_clean[(df_clean['latitude'].between(-90, 90)) &
                                (df_clean['longitude'].between(-180, 180))]

            if len(df_clean) == 0:
                raise ValueError("No valid GPS coordinates found in CSV")

            # Add risk analysis if not present
            if 'risk_score' not in df_clean.columns:
                df_clean = self._generate_risk_scores(df_clean)

            if 'risk_category' not in df_clean.columns:
                df_clean['risk_category'] = df_clean['risk_score'].apply(self._categorize_risk)

            logger.info(f"Processed data: {len(df_clean)} valid GPS points")
            return df_clean

        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise

    def _generate_risk_scores(self, df):
        """Generate synthetic risk scores based on available data"""
        df = df.copy()  # Avoid modifying original dataframe

        # Use speed if available, otherwise use GPS variations
        if 'speed' in df.columns:
            # Higher speeds = higher risk
            speed_range = df['speed'].max() - df['speed'].min()
            if speed_range > 0:
                normalized_speed = (df['speed'] - df['speed'].min()) / speed_range
                df['risk_score'] = 2 + (normalized_speed * 6)  # Scale to 2-8 range
            else:
                df['risk_score'] = 5.0  # Default medium risk
        else:
            # Use GPS coordinate variations as proxy for risk
            df['lat_diff'] = df['latitude'].diff().abs().fillna(0)
            df['lon_diff'] = df['longitude'].diff().abs().fillna(0)
            df['movement'] = np.sqrt(df['lat_diff'] ** 2 + df['lon_diff'] ** 2)

            # Normalize movement to risk score
            movement_range = df['movement'].max() - df['movement'].min()
            if movement_range > 0:
                movement_normalized = (df['movement'] - df['movement'].min()) / movement_range
                df['risk_score'] = 1 + (movement_normalized * 7)  # Scale to 1-8 range
            else:
                df['risk_score'] = 4.0  # Default medium risk

            # Add some randomness for more interesting visualization
            df['risk_score'] += np.random.normal(0, 0.5, len(df))
            df['risk_score'] = np.clip(df['risk_score'], 0, 10)

        return df

    def _categorize_risk(self, risk_score):
        """Categorize risk scores into low/medium/high"""
        if risk_score < 3:
            return 'low'
        elif risk_score < 6:
            return 'medium'
        else:
            return 'high'

    def create_driving_route_map(self, df, save_path="static/maps/route_map.html"):
        """Create interactive route map"""
        # Calculate map center
        center_lat = df['latitude'].mean()
        center_lon = df['longitude'].mean()

        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=13,
            tiles='OpenStreetMap'
        )

        # Add route line with risk-based coloring
        if len(df) > 1:
            route_coords = df[['latitude', 'longitude']].values.tolist()

            for i in range(len(route_coords) - 1):
                start_point = route_coords[i]
                end_point = route_coords[i + 1]
                risk_level = df.iloc[i]['risk_category']

                folium.PolyLine(
                    locations=[start_point, end_point],
                    color=self.risk_color_map.get(risk_level, 'blue'),
                    weight=4,
                    opacity=0.7,
                    popup=f"Risk: {risk_level} ({df.iloc[i]['risk_score']:.1f})"
                ).add_to(m)

        # Add start and end markers
        start_point = df.iloc[0]
        end_point = df.iloc[-1]

        folium.Marker(
            location=[start_point['latitude'], start_point['longitude']],
            popup="🚗 Trip Start",
            icon=folium.Icon(color='green', icon='play', prefix='fa')
        ).add_to(m)

        folium.Marker(
            location=[end_point['latitude'], end_point['longitude']],
            popup="🏁 Trip End",
            icon=folium.Icon(color='black', icon='stop', prefix='fa')
        ).add_to(m)

        # Add clustered markers for high-risk points
        marker_cluster = MarkerCluster().add_to(m)
        high_risk_points = df[df['risk_score'] >= 6.0]

        for idx, point in high_risk_points.iterrows():
            popup_content = self._create_detailed_popup(point)

            folium.Marker(
                location=[point['latitude'], point['longitude']],
                popup=folium.Popup(popup_content, max_width=300),
                tooltip=f"⚠️ Risk: {point['risk_score']:.1f}",
                icon=folium.Icon(
                    color='red',
                    icon='exclamation-triangle',
                    prefix='fa'
                )
            ).add_to(marker_cluster)

        # Add interactive features
        self._add_legend(m)
        self._add_trip_statistics(m, df)
        self._add_controls(m)

        # Save map
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        m.save(save_path)

        return m, save_path

    def create_risk_heatmap(self, df, save_path="static/maps/heatmap.html"):
        """Create risk heatmap"""
        center_lat = df['latitude'].mean()
        center_lon = df['longitude'].mean()

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles='OpenStreetMap'
        )

        # Prepare heatmap data
        heat_data = []
        for idx, row in df.iterrows():
            weight = max(0.1, row['risk_score'] / 10.0)
            heat_data.append([row['latitude'], row['longitude'], weight])

        # Add heatmap
        HeatMap(
            heat_data,
            min_opacity=0.3,
            max_zoom=18,
            radius=15,
            blur=10,
            gradient={0.2: 'green', 0.5: 'yellow', 0.8: 'orange', 1.0: 'red'}
        ).add_to(m)

        # Add controls and legend
        self._add_controls(m)
        self._add_heatmap_legend(m)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        m.save(save_path)
        return m, save_path

    def _create_detailed_popup(self, point):
        """Create detailed popup for markers"""
        popup_content = f"""
        <div style="font-family: Arial; font-size: 12px; max-width: 250px;">
        <b>🚨 Risk Event</b><br>
        <b>Risk Score:</b> {point['risk_score']:.2f}/10<br>
        <b>Risk Level:</b> {point['risk_category']}<br>
        <b>Location:</b> {point['latitude']:.4f}, {point['longitude']:.4f}<br>
        """

        # Add additional fields if available
        additional_fields = {
            'speed': 'Speed (km/h)',
            'acceleration': 'Acceleration (m/s²)',
            'rpm': 'RPM',
            'engine_temperature': 'Engine Temp (°C)',
            'timestamp': 'Time'
        }

        for col, label in additional_fields.items():
            if col in point.index and pd.notna(point[col]):
                if col == 'timestamp':
                    popup_content += f"<b>{label}:</b> {str(point[col])}<br>"
                else:
                    popup_content += f"<b>{label}:</b> {point[col]:.2f}<br>"

        popup_content += "</div>"
        return popup_content

    def _add_legend(self, map_obj):
        """Add legend to map"""
        legend_html = '''
        <div style="position: fixed;
                    bottom: 50px; left: 50px; width: 180px; height: 140px;
                    background-color: white; border:2px solid grey; z-index:9999;
                    font-size:12px; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <p><b>🚗 DriveSafeAI Risk Levels</b></p>
        <p><i class="fa fa-circle" style="color:red"></i> High Risk (6-10)</p>
        <p><i class="fa fa-circle" style="color:orange"></i> Medium Risk (3-6)</p>
        <p><i class="fa fa-circle" style="color:green"></i> Low Risk (0-3)</p>
        <p><i class="fa fa-play" style="color:green"></i> Trip Start</p>
        <p><i class="fa fa-stop" style="color:black"></i> Trip End</p>
        </div>
        '''
        map_obj.get_root().html.add_child(folium.Element(legend_html))

    def _add_heatmap_legend(self, map_obj):
        """Add heatmap legend"""
        legend_html = '''
        <div style="position: fixed;
                    bottom: 50px; left: 50px; width: 180px; height: 100px;
                    background-color: white; border:2px solid grey; z-index:9999;
                    font-size:12px; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <p><b>🔥 Risk Heatmap</b></p>
        <div style="background: linear-gradient(to right, green, yellow, orange, red); height: 10px; margin: 5px 0;"></div>
        <div style="display: flex; justify-content: space-between; font-size: 10px;">
            <span>Low</span><span>High</span>
        </div>
        </div>
        '''
        map_obj.get_root().html.add_child(folium.Element(legend_html))

    def _add_trip_statistics(self, map_obj, df):
        """Add trip statistics"""
        avg_risk = df['risk_score'].mean()
        max_risk = df['risk_score'].max()
        high_risk_count = len(df[df['risk_score'] >= 6])

        # Calculate distance if possible
        total_distance = 0
        if len(df) > 1:
            for i in range(len(df) - 1):
                lat1, lon1 = df.iloc[i][['latitude', 'longitude']]
                lat2, lon2 = df.iloc[i + 1][['latitude', 'longitude']]
                # Simple distance calculation (not perfectly accurate but good enough)
                distance = np.sqrt((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2) * 111  # Rough km conversion
                total_distance += distance

        stats_html = f'''
        <div style="position: fixed;
                    top: 10px; right: 10px; width: 220px; height: 140px;
                    background-color: white; border:2px solid grey; z-index:9999;
                    font-size:12px; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <p><b>📊 Trip Statistics</b></p>
        <p>📍 Total Points: {len(df)}</p>
        <p>📈 Average Risk: {avg_risk:.2f}/10</p>
        <p>🔴 Max Risk: {max_risk:.2f}/10</p>
        <p>⚠️ High Risk Events: {high_risk_count}</p>
        <p>📏 Est. Distance: {total_distance:.1f} km</p>
        </div>
        '''
        map_obj.get_root().html.add_child(folium.Element(stats_html))

    def _add_controls(self, map_obj):
        """Add map controls"""
        # Add fullscreen button
        Fullscreen().add_to(map_obj)


# Flask Routes
@app.route('/')
def home():
    """Home page with API information"""
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>DriveSafeAI Map API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; }
            .button { display: inline-block; padding: 10px 20px; background-color: #007bff; color: white; text-decoration: none; border-radius: 5px; margin: 5px; }
            .button:hover { background-color: #0056b3; }
            .info { background-color: #e3f2fd; padding: 15px; border-radius: 5px; margin: 20px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚗 DriveSafeAI Map Visualization API</h1>

            <div class="info">
                <h3>📋 Available Endpoints:</h3>
                <ul>
                    <li><strong>/generate-map</strong> - Generate route map from CSV</li>
                    <li><strong>/generate-heatmap</strong> - Generate risk heatmap from CSV</li>
                    <li><strong>/view-map</strong> - View the generated route map</li>
                    <li><strong>/view-heatmap</strong> - View the generated heatmap</li>
                    <li><strong>/api/stats</strong> - Get CSV statistics as JSON</li>
                </ul>
            </div>

            <div style="text-align: center;">
                <a href="/generate-map" class="button">🗺️ Generate Route Map</a>
                <a href="/generate-heatmap" class="button">🔥 Generate Heatmap</a>
                <a href="/api/stats" class="button">📊 View Stats</a>
            </div>

            <div class="info">
                <h3>📝 Requirements:</h3>
                <p>Your CSV file (DF2.csv) must contain 'latitude' and 'longitude' columns.</p>
                <p>Optional columns: speed, acceleration, rpm, engine_temperature, timestamp</p>
            </div>
        </div>
    </body>
    </html>
    ''')


@app.route('/generate-map')
def generate_map():
    """Generate route map from CSV"""
    try:
        visualizer = MapVisualizer()
        df = visualizer.load_and_process_csv('C:/Users/Abhijith/OneDrive/Desktop/DF2.csv')
        map_obj, path = visualizer.create_driving_route_map(df)

        return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head><title>Map Generated</title></head>
        <body style="font-family: Arial; text-align: center; margin: 50px;">
            <h2>✅ Route Map Generated Successfully!</h2>
            <p>Processed {{ point_count }} GPS points</p>
            <a href="/view-map" style="padding: 10px 20px; background-color: #28a745; color: white; text-decoration: none; border-radius: 5px;">🗺️ View Interactive Map</a>
            <br><br>
            <a href="/" style="color: #007bff;">← Back to Home</a>
        </body>
        </html>
        ''', point_count=len(df))

    except Exception as e:
        return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head><title>Error</title></head>
        <body style="font-family: Arial; text-align: center; margin: 50px;">
            <h2>❌ Error Generating Map</h2>
            <p style="color: red;">{{ error }}</p>
            <a href="/" style="color: #007bff;">← Back to Home</a>
        </body>
        </html>
        ''', error=str(e))


@app.route('/generate-heatmap')
def generate_heatmap():
    """Generate risk heatmap from CSV"""
    try:
        visualizer = MapVisualizer()
        df = visualizer.load_and_process_csv('C:/Users/Abhijith/OneDrive/Desktop/DF2.csv')
        map_obj, path = visualizer.create_risk_heatmap(df)

        return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head><title>Heatmap Generated</title></head>
        <body style="font-family: Arial; text-align: center; margin: 50px;">
            <h2>✅ Risk Heatmap Generated Successfully!</h2>
            <p>Processed {{ point_count }} GPS points</p>
            <a href="/view-heatmap" style="padding: 10px 20px; background-color: #dc3545; color: white; text-decoration: none; border-radius: 5px;">🔥 View Heatmap</a>
            <br><br>
            <a href="/" style="color: #007bff;">← Back to Home</a>
        </body>
        </html>
        ''', point_count=len(df))

    except Exception as e:
        return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head><title>Error</title></head>
        <body style="font-family: Arial; text-align: center; margin: 50px;">
            <h2>❌ Error Generating Heatmap</h2>
            <p style="color: red;">{{ error }}</p>
            <a href="/" style="color: #007bff;">← Back to Home</a>
        </body>
        </html>
        ''', error=str(e))


@app.route('/view-map')
def view_map():
    """View the generated route map"""
    try:
        return send_file('static/maps/route_map.html')
    except FileNotFoundError:
        return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head><title>Map Not Found</title></head>
        <body style="font-family: Arial; text-align: center; margin: 50px;">
            <h2>⚠️ Map Not Found</h2>
            <p>Please generate a map first.</p>
            <a href="/generate-map" style="padding: 10px 20px; background-color: #007bff; color: white; text-decoration: none; border-radius: 5px;">Generate Map</a>
            <br><br>
            <a href="/" style="color: #007bff;">← Back to Home</a>
        </body>
        </html>
        ''')


@app.route('/view-heatmap')
def view_heatmap():
    """View the generated heatmap"""
    try:
        return send_file('static/maps/heatmap.html')
    except FileNotFoundError:
        return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head><title>Heatmap Not Found</title></head>
        <body style="font-family: Arial; text-align: center; margin: 50px;">
            <h2>⚠️ Heatmap Not Found</h2>
            <p>Please generate a heatmap first.</p>
            <a href="/generate-heatmap" style="padding: 10px 20px; background-color: #dc3545; color: white; text-decoration: none; border-radius: 5px;">Generate Heatmap</a>
            <br><br>
            <a href="/" style="color: #007bff;">← Back to Home</a>
        </body>
        </html>
        ''')


@app.route('/api/stats')
def get_stats():
    """Get CSV statistics as JSON"""
    try:
        visualizer = MapVisualizer()
        df = visualizer.load_and_process_csv('C:/Users/Abhijith/OneDrive/Desktop/DF2.csv')

        stats = {
            'total_points': len(df),
            'avg_risk_score': float(df['risk_score'].mean()),
            'max_risk_score': float(df['risk_score'].max()),
            'min_risk_score': float(df['risk_score'].min()),
            'high_risk_count': int(len(df[df['risk_score'] >= 6])),
            'medium_risk_count': int(len(df[(df['risk_score'] >= 3) & (df['risk_score'] < 6)])),
            'low_risk_count': int(len(df[df['risk_score'] < 3])),
            'lat_range': [float(df['latitude'].min()), float(df['latitude'].max())],
            'lon_range': [float(df['longitude'].min()), float(df['longitude'].max())],
            'available_columns': list(df.columns)
        }

        return jsonify(stats)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint"""
    if analyzer and analyzer.is_trained:
        return jsonify({
            "status": "healthy",
            "message": "Driver Risk Analysis API with LLM Feedback is running",
            "model_trained": True,
            "features_count": len(analyzer.training_features) if analyzer.training_features else 0,
            "openai_configured": bool(os.getenv("OPENAI_API_KEY"))
        })
    else:
        return jsonify({
            "status": "initializing",
            "message": "Model is still training...",
            "model_trained": False
        }), 503


@app.route("/predict", methods=["POST"])
def predict_drive_score():
    """Predict drive score for input data (JSON or CSV file)"""
    try:
        if not analyzer or not analyzer.is_trained:
            return jsonify({
                "error": "Model not ready yet. Please wait for training to complete."
            }), 503

        # Check if request contains a file (CSV upload)
        if 'file' in request.files:
            file = request.files['file']

            # Check if file was actually uploaded
            if file.filename == '':
                return jsonify({
                    "error": "No file selected. Please upload a CSV file."
                }), 400

            # Check file extension
            if not file.filename.lower().endswith('.csv'):
                return jsonify({
                    "error": "Invalid file format. Please upload a CSV file."
                }), 400

            try:
                # Read CSV file directly from memory
                file_content = file.read().decode('utf-8')
                df_input = pd.read_csv(io.StringIO(file_content))

                # Validate that CSV is not empty
                if df_input.empty:
                    return jsonify({
                        "error": "CSV file is empty. Please upload a file with data."
                    }), 400

            except Exception as e:
                return jsonify({
                    "error": f"Error reading CSV file: {str(e)}"
                }), 400

        # Check if request contains JSON data
        elif request.is_json:
            data = request.get_json()

            if not data:
                return jsonify({
                    "error": "No data provided. Please send JSON data in the request body or upload a CSV file."
                }), 400

            # Handle both single record and multiple records
            if isinstance(data, dict):
                # Single record
                df_input = pd.DataFrame([data])
            elif isinstance(data, list):
                # Multiple records
                df_input = pd.DataFrame(data)
            else:
                return jsonify({
                    "error": "Invalid data format. Expected JSON object or array."
                }), 400

        else:
            return jsonify({
                "error": "No data provided. Please send JSON data or upload a CSV file."
            }), 400

        # Make prediction
        predictions = analyzer.predict_risk(df_input)

        # Format response
        if len(predictions) == 1:
            # Single prediction
            result = {
                "drive_score": round(float(predictions['drive_score'].iloc[0]), 2),
                "risk_score": round(float(predictions['risk_score'].iloc[0]), 2),
                "risk_category": str(predictions['risk_category'].iloc[0]),
                "input_method": "csv_file" if 'file' in request.files else "json"
            }
        else:
            # Multiple predictions
            result = {
                "prediction": {
                    "drive_score": round(float(predictions['drive_score'].mean()), 2),
                    "trip_id": round(float(predictions['trip_id'].mean())) if 'trip_id' in predictions else 0,
                    "driver_id": round(float(predictions['id_driver'].mean())) if 'id_driver' in predictions else 0,
                    "distance_travelled": round(
                        float(predictions['distance_travelled'].sum())) if 'distance_travelled' in predictions else 0,
                }
            }

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({
            "error": f"Prediction failed: {str(e)}"
        }), 500


@app.route("/generate_feedback", methods=["POST"])
def generate_feedback_api():
    """Generate AI-powered feedback for trip data"""
    try:
        if not analyzer or not analyzer.is_trained:
            return jsonify({
                "error": "Model not ready yet. Please wait for training to complete."
            }), 503

        # Handle file upload or JSON data
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400

            if not file.filename.lower().endswith('.csv'):
                return jsonify({"error": "Invalid file format. Please upload a CSV file."}), 400

            try:
                file_content = file.read().decode('utf-8')
                df = pd.read_csv(io.StringIO(file_content))
            except Exception as e:
                return jsonify({"error": f"Error reading CSV file: {str(e)}"}), 400

        elif request.is_json:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No data provided"}), 400
            df = pd.DataFrame(data)
        else:
            return jsonify({"error": "No data provided"}), 400

        if df.empty:
            return jsonify({"error": "Empty dataset"}), 400

        # First get predictions from the analyzer
        predictions = analyzer.predict_risk(df)

        # Merge predictions back with original data for feedback generation
        df_with_predictions = df.copy()
        if 'drive_score' not in df_with_predictions.columns:
            df_with_predictions['drive_score'] = predictions['drive_score']
        if 'risk_score' not in df_with_predictions.columns:
            df_with_predictions['risk_score'] = predictions['risk_score']


        feedbacks = []
        for trip_id, group in df_with_predictions.groupby('trip_id'):
            feedback = process_trip_with_feedback(group)
            feedbacks.append(feedback)

        return jsonify({
            "feedbacks": feedbacks,
            "total_trips": len(feedbacks),
            "processed_at": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Feedback API error: {e}")
        return jsonify({"error": f"Processing error: {str(e)}"}), 500


@app.route("/predict_with_feedback", methods=["POST"])
def predict_with_feedback():
    """Combined endpoint: predict drive scores AND generate AI feedback - fetches data from CSV"""
    try:
        if not analyzer or not analyzer.is_trained:
            return jsonify({
                "error": "Model not ready yet. Please wait for training to complete."
            }), 503

        # Read data from CSV file instead of SQL
        csv_file_path = "df2.csv"  # Update path if needed

        try:
            # Load CSV into DataFrame
            df_input = pd.read_csv(r"C:\Users\Abhijith\OneDrive\Desktop\DF2.csv")

            if df_input.empty:
                return jsonify({
                    "error": "CSV file is empty or contains no data"
                }), 400

            # Step 1: Get predictions
            predictions = analyzer.predict_risk(df_input)

            # Step 2: Merge predictions with original data
            df_with_predictions = df_input.copy()
            df_with_predictions['driveScore'] = predictions['driveScore']
            df_with_predictions['risk_score'] = predictions['risk_score']
            df_with_predictions['risk_category'] = predictions['risk_category']

            # Step 3: Generate AI feedback
            if 'trip_id' not in df_with_predictions.columns:
                df_with_predictions['trip_id'] = range(1, len(df_with_predictions) + 1)

            feedbacks = []
            trip_summaries = []

            for trip_id, group in df_with_predictions.groupby('trip_id'):
                # Use the updated function that handles OpenAI properly
                feedback = process_trip_with_feedback_updated(group)
                feedbacks.append(feedback)

                # Get prediction summary for this trip - FIX THE SERIALIZATION HERE
                trip_summary = {
                    "trip_id": convert_to_serializable(trip_id),
                    "driveScore": convert_to_serializable(round(float(group['driveScore'].mean()), 2)),
                    "risk_score": convert_to_serializable(round(float(group['risk_score'].mean()), 2)),
                    "risk_category": str(group['risk_category'].iloc[0]),
                    "records_count": convert_to_serializable(len(group))
                }
                trip_summaries.append(trip_summary)

            # Overall summary - FIX THE SERIALIZATION HERE TOO
            overall_summary = {
                "average_drive_score": convert_to_serializable(round(float(predictions['driveScore'].mean()), 2)),
                "total_trips": convert_to_serializable(len(feedbacks)),
                "total_records": convert_to_serializable(len(df_input)),
                "ai_feedback_success_rate": convert_to_serializable(
                    sum(1 for f in feedbacks if f.get('ai_generated', False)) / len(feedbacks) * 100
                ),
                "input_method": "csv_file"
            }

            return jsonify({
                "AI_feedback": feedbacks,
                "trip_summaries": trip_summaries,
                "overall_summary": overall_summary,
                "processed_at": datetime.now().isoformat()
            })

        except FileNotFoundError:
            return jsonify({
                "error": "DF2.csv file not found. Please ensure the file exists in the correct path."
            }), 404
        except pd.errors.EmptyDataError:
            return jsonify({
                "error": "CSV file is empty or corrupted"
            }), 400
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            return jsonify({
                "error": f"Failed to read CSV file: {str(e)}"
            }), 500

    except Exception as e:
        logger.error(f"Combined prediction error: {e}")
        return jsonify({"error": f"Processing error: {str(e)}"}), 500

def process_trip_with_feedback_updated(trip_df: pd.DataFrame) -> Dict[str, Any]:
    """Process a single trip with AI feedback and fallback handling - UPDATED VERSION."""
    trip_id = trip_df['trip_id'].iloc[0] if 'trip_id' in trip_df else 'unknown'

    # Try AI generation first using the updated function
    result = llm_generate_feedback_and_tags_with_fallback(trip_df)

    # Add metadata - FIX SERIALIZATION HERE
    result.update({
        "trip_id": convert_to_serializable(trip_id),
        "processed_at": datetime.now().isoformat(),
        "duration_minutes": convert_to_serializable(len(trip_df) / 60),
        "feedback_source": "openai" if result.get('ai_generated', False) else "fallback"
    })

    return result


@app.route("/predictfsql", methods=["POST"])
def predict_from_sql():
    """Fetch data from SQL and predict directly without CSV conversion"""
    try:
        if not analyzer or not analyzer.is_trained:
            return jsonify({
                "error": "Model not ready yet. Please wait for training to complete."
            }), 503

        # Get query parameters from request
        data = request.get_json()

        if not data or 'query' not in data:
            return jsonify({
                "error": "Please provide a SQL query in JSON format: {'query': 'SELECT * FROM table'}"
            }), 400

        sql_query = data['query']

        # Connect to database (adjust connection parameters as needed)
        conn = sqlite3.connect('your_database.db')  # Replace with your DB

        try:
            # Execute query and get data as DataFrame
            df_from_sql = pd.read_sql_query(sql_query, conn)

            if df_from_sql.empty:
                return jsonify({
                    "error": "Query returned no data"
                }), 400

            # Make predictions directly on the DataFrame
            predictions = analyzer.predict_risk(df_from_sql)

            # Format response
            if len(predictions) == 1:
                result = {
                    "driveScore": round(float(predictions['driveScore'].iloc[0]), 2),
                    "risk_score": round(float(predictions['risk_score'].iloc[0]), 2),
                    "risk_category": str(predictions['risk_category'].iloc[0]),
                    "input_method": "sql_query",
                    "records_processed": len(df_from_sql)
                }
            else:
                result = {
                    "predictions": [
                        {
                            "driveScore": round(float(row['driveScore']), 2),
                            "risk_score": round(float(row['risk_score']), 2),
                            "risk_category": str(row['risk_category'])
                        }
                        for _, row in predictions.iterrows()
                    ],
                    "summary": {
                        "average_drive_score": round(float(predictions['driveScore'].mean()), 2),
                        "average_risk_score": round(float(predictions['risk_score'].mean()), 2),
                        "total_records": len(predictions)
                    },
                    "input_method": "sql_query"
                }

            return jsonify(result)

        finally:
            conn.close()

    except Exception as e:
        logger.error(f"Error during SQL prediction: {e}")
        return jsonify({
            "error": f"Prediction failed: {str(e)}"
        }), 500


@app.route("/model_info", methods=["GET"])
def model_info():
    """Get information about the trained model"""
    try:
        if not analyzer or not analyzer.is_trained:
            return jsonify({
                "error": "Model not trained yet."
            }), 503

        # Get feature importance
        feature_importance = None
        if analyzer.feature_importance is not None:
            feature_importance = analyzer.feature_importance.head(10).to_dict('records')

        return jsonify({
            "model_status": "trained",
            "training_features_count": len(analyzer.training_features),
            "risk_thresholds": analyzer.risk_thresholds,
            "top_features": feature_importance,
            "model_params": analyzer.model_params,
            "openai_configured": bool(os.getenv("OPENAI_API_KEY"))
        })

    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({
            "error": f"Failed to get model info: {str(e)}"
        }), 500


@app.route("/debug-file", methods=["POST"])
def debug_file_processing():
    """Debug route to see exactly how Flask processes files"""
    debug_info = {
        "request_files_keys": list(request.files.keys()),
        "request_files_length": len(request.files),
        "content_type": request.content_type,
        "has_file_key": 'file' in request.files
    }

    # If we have files, get details about each one
    if request.files:
        debug_info["file_details"] = {}
        for key, file_obj in request.files.items():
            debug_info["file_details"][key] = {
                "filename": file_obj.filename,
                "content_type": file_obj.content_type,
                "content_length": len(file_obj.read()),
                "has_content": bool(file_obj.filename)
            }
            file_obj.seek(0)

    # Also check if there's a specific 'file' key
    if 'file' in request.files:
        file = request.files['file']
        debug_info["specific_file_info"] = {
            "filename": file.filename,
            "is_empty_filename": file.filename == '',
            "content_type": file.content_type,
            "first_100_chars": file.read(100).decode('utf-8', errors='ignore')
        }
        file.seek(0)

    return jsonify(debug_info)


@app.route("/debug_openai", methods=["GET"])
def debug_openai_comprehensive():
    """Comprehensive OpenAI debugging endpoint"""
    debug_info = {
        "timestamp": datetime.now().isoformat(),
        "environment_check": {},
        "library_check": {},
        "api_key_check": {},
        "connection_test": {},
        "recommendations": []
    }

    try:
        # 1. Environment Check
        debug_info["environment_check"] = {
            "python_version": sys.version,
            "os_type": os.name,
            "current_directory": os.getcwd()
        }

        # 2. Library Check
        try:
            import openai
            debug_info["library_check"] = {
                "openai_imported": True,
                "openai_version": getattr(openai, '__version__', 'unknown'),
                "openai_module_path": openai.__file__ if hasattr(openai, '__file__') else 'unknown'
            }
        except ImportError as e:
            debug_info["library_check"] = {
                "openai_imported": False,
                "import_error": str(e)
            }
            debug_info["recommendations"].append("Install OpenAI library: pip install openai>=1.0.0")

        # 3. API Key Check
        api_key = os.getenv("OPENAI_API_KEY")
        debug_info["api_key_check"] = {
            "api_key_exists": bool(api_key),
            "api_key_length": len(api_key) if api_key else 0,
            "starts_with_sk": api_key.startswith('sk-') if api_key else False,
            "has_proj_format": api_key.startswith('sk-proj-') if api_key else False,
            "first_10_chars": api_key[:10] if api_key else None,
            "last_4_chars": api_key[-4:] if api_key else None
        }

        if not api_key:
            debug_info["recommendations"].append("Set OPENAI_API_KEY environment variable")
        elif not api_key.startswith('sk-'):
            debug_info["recommendations"].append("API key should start with 'sk-'")
        elif len(api_key) < 40:
            debug_info["recommendations"].append("API key seems too short")

        # 4. Connection Test
        if api_key and api_key.startswith('sk-'):
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)

                # Test simple API call
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=5
                )

                debug_info["connection_test"] = {
                    "connection_successful": True,
                    "response_received": True,
                    "response_content": response.choices[0].message.content,
                    "model_used": "gpt-3.5-turbo"
                }

            except Exception as e:
                debug_info["connection_test"] = {
                    "connection_successful": False,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "full_traceback": traceback.format_exc()
                }

                # Add specific recommendations based on error type
                error_str = str(e).lower()
                if "authentication" in error_str or "invalid api key" in error_str:
                    debug_info["recommendations"].append("Invalid API key - check your OpenAI API key")
                elif "quota" in error_str or "billing" in error_str:
                    debug_info["recommendations"].append("OpenAI account billing issue - check your OpenAI account")
                elif "rate limit" in error_str:
                    debug_info["recommendations"].append("Rate limit exceeded - wait and try again")
                elif "network" in error_str or "connection" in error_str:
                    debug_info["recommendations"].append("Network connection issue - check internet connectivity")
        else:
            debug_info["connection_test"] = {
                "connection_successful": False,
                "error_message": "Skipped due to invalid API key"
            }

    except Exception as e:
        debug_info["debug_error"] = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc()
        }

    return jsonify(debug_info)

# Initialize model on startup
print("🚀 Initializing Driver Risk Analysis Model with LLM Feedback...")
initialize_model()

if __name__ == "__main__":
    print("🚀 Starting Driver Risk Analysis API with LLM Feedback...")
    print("📊 Endpoints:")
    print("   GET  / - Health check")
    print("   POST /predict - Get drive score predictions")
    print("   POST /generate_feedback - Generate AI feedback for trip data")
    print("   POST /predict_with_feedback - Combined predictions + AI feedback")
    print("   POST /predictfsql - Predict from SQL query")
    print("   GET  /model_info - Model information")
    print("   POST /debug-file - Debug file uploads")
    app.run(host="0.0.0.0", port=5000, debug=True)

