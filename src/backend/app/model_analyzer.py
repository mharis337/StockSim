# model_analyzer.py

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import logging
import talib
import pandas as pd

logger = logging.getLogger(__name__)

class ModelAnalyzer:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.input_shape = self.model.input_shape
        self.scaler = MinMaxScaler()
        logger.info(f"Model loaded with input shape: {self.input_shape}")

    def get_required_features(self):
        """Get the number of features required by the model"""
        if len(self.input_shape) != 3:
            raise ValueError("Model input shape must be (batch_size, sequence_length, features)")
        return self.input_shape[-1]

    def calculate_technical_indicators(self, df):
        """Calculate all technical indicators for the dataframe"""
        try:
            # Ensure numeric type for all inputs
            close = df['Close'].astype(float).values
            high = df['High'].astype(float).values
            low = df['Low'].astype(float).values
            volume = df['Volume'].astype(float).values
            
            # Moving Averages
            df['SMA_20'] = talib.SMA(close, timeperiod=20)
            df['EMA_20'] = talib.EMA(close, timeperiod=20)
            df['MA5'] = talib.SMA(close, timeperiod=5)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close)
            df['MACD'] = macd
            df['MACD_Signal'] = macd_signal
            df['MACD_Hist'] = macd_hist
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close)
            df['BB_Upper'] = bb_upper
            df['BB_Middle'] = bb_middle
            df['BB_Lower'] = bb_lower
            
            # Volatility Indicators
            df['ATR'] = talib.ATR(high, low, close)
            df['STDDEV'] = talib.STDDEV(close, timeperiod=20)
            df['SAR'] = talib.SAR(high, low)
            
            # Momentum Indicators
            df['RSI'] = talib.RSI(close)
            df['ROC'] = talib.ROC(close)
            df['WILLR'] = talib.WILLR(high, low, close)
            df['STOCH_K'], df['STOCH_D'] = talib.STOCH(high, low, close)
            
            # Volume Indicators
            df['OBV'] = talib.OBV(close, volume)
            df['AD'] = talib.AD(high, low, close, volume)  # Fixed: Using correct function name
            df['MFI'] = talib.MFI(high, low, close, volume)
            
            # Trend Indicators
            df['ADX'] = talib.ADX(high, low, close)
            df['PLUS_DI'] = talib.PLUS_DI(high, low, close)
            df['MINUS_DI'] = talib.MINUS_DI(high, low, close)
            
            # Custom Indicators
            typical_price = (high + low + close) / 3
            df['PIVOT'] = typical_price
            df['R1'] = (2 * df['PIVOT']) - low
            df['S1'] = (2 * df['PIVOT']) - high
            
            # Replace infinite values with NaN
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # Forward fill NaN values first
            df.fillna(method='ffill', inplace=True)
            
            # Then backward fill any remaining NaNs at the beginning
            df.fillna(method='bfill', inplace=True)
            
            # Create additional derived features if needed to reach required feature count
            df['Price_MA5_Ratio'] = close / df['MA5']
            df['Price_MA20_Ratio'] = close / df['SMA_20']
            df['BB_Position'] = (close - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
            df['Trend_Strength'] = df['ADX'] * np.where(df['PLUS_DI'] > df['MINUS_DI'], 1, -1)
            
            logger.info("Technical indicators calculated successfully")
            logger.info(f"Available features: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            raise

    def prepare_features(self, df, selected_features, sequence_length=10):
        """Prepare features for model input with dynamic feature selection"""
        try:
            required_features = self.get_required_features()
            logger.info(f"Model requires {required_features} features")

            # Calculate all technical indicators first
            df = self.calculate_technical_indicators(df)
            
            # Get all available features
            available_features = list(df.columns)
            logger.info(f"Available features after calculation: {available_features}")

            # If no features selected, use all available ones
            if not selected_features:
                selected_features = available_features

            # Validate we have enough features
            if len(selected_features) < required_features:
                remaining_features = [f for f in available_features if f not in selected_features]
                additional_features = remaining_features[:required_features - len(selected_features)]
                selected_features.extend(additional_features)
                logger.info(f"Auto-selected additional features: {additional_features}")
            elif len(selected_features) > required_features:
                logger.warning(f"Too many features selected, using first {required_features}")
                selected_features = selected_features[:required_features]

            # Extract and scale features
            feature_data = df[selected_features].values
            scaled_data = self.scaler.fit_transform(feature_data)

            # Create sequences
            X = []
            timestamps = []
            for i in range(sequence_length, len(scaled_data)):
                X.append(scaled_data[i-sequence_length:i])
                timestamps.append(df.index[i])

            X = np.array(X)
            logger.info(f"Prepared sequences with shape: {X.shape}")

            return X, timestamps, selected_features

        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise

    def analyze_stock(self, df, selected_features):
        """Perform complete stock analysis with the model"""
        try:
            # Prepare sequences
            X, timestamps, used_features = self.prepare_features(df, selected_features)
            
            # Make predictions
            predictions = self.model.predict(X)
            if len(predictions.shape) > 1:
                predictions = predictions.flatten()
            
            # Calculate signals
            signals = []
            prices = df['Close'].values[-len(predictions):]
            
            for i in range(len(predictions)):
                current_price = prices[i]
                predicted_price = predictions[i]
                predicted_change = (predicted_price - current_price) / current_price * 100
                
                signal = "hold"
                if predicted_change > 1:  # 1% threshold
                    signal = "buy"
                elif predicted_change < -1:
                    signal = "sell"
                
                confidence = min(abs(predicted_change) * 20, 100)  # Scale confidence
                
                signals.append({
                    "date": timestamps[i].strftime("%Y-%m-%d"),
                    "price": float(current_price),
                    "predicted_price": float(predicted_price),
                    "predicted_change": float(predicted_change),
                    "signal": signal,
                    "confidence": float(confidence),
                    "target_price": float(predicted_price),
                    "stop_loss": float(current_price * (0.98 if signal == "buy" else 1.02))
                })
            
            return signals, used_features

        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            raise