import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import logging
import talib
import pandas as pd
from model_cache import model_cache

logger = logging.getLogger(__name__)

class ModelAnalyzer:
    def __init__(self, model_path):
        self.model = model_cache.get_model(model_path)
        if not self.model:
            raise ValueError(f"Failed to load model from {model_path}")
        self.model_info = model_cache.get_model_info(model_path)
        self.input_shape = self.model_info["input_shape"]
        self.scaler = MinMaxScaler()
        logger.info(f"Model loaded with input shape: {self.input_shape}")

    def get_required_features(self):
        if len(self.input_shape) != 3:
            raise ValueError("Model input shape must be (batch_size, sequence_length, features)")
        return self.input_shape[-1]

    def calculate_technical_indicators(self, df):
        try:
            close = df['Close'].astype(float).values
            high = df['High'].astype(float).values
            low = df['Low'].astype(float).values
            volume = df['Volume'].astype(float).values

            df['SMA_20'] = talib.SMA(close, timeperiod=20)
            df['EMA_20'] = talib.EMA(close, timeperiod=20)
            df['MA5'] = talib.SMA(close, timeperiod=5)

            macd, macd_signal, macd_hist = talib.MACD(close)
            df['MACD'] = macd
            df['MACD_Signal'] = macd_signal
            df['MACD_Hist'] = macd_hist

            bb_upper, bb_middle, bb_lower = talib.BBANDS(close)
            df['BB_Upper'] = bb_upper
            df['BB_Middle'] = bb_middle
            df['BB_Lower'] = bb_lower

            df['ATR'] = talib.ATR(high, low, close)
            df['STDDEV'] = talib.STDDEV(close, timeperiod=20)
            df['SAR'] = talib.SAR(high, low)

            df['RSI'] = talib.RSI(close)
            df['ROC'] = talib.ROC(close)
            df['WILLR'] = talib.WILLR(high, low, close)
            df['STOCH_K'], df['STOCH_D'] = talib.STOCH(high, low, close)

            df['OBV'] = talib.OBV(close, volume)
            df['AD'] = talib.AD(high, low, close, volume)
            df['MFI'] = talib.MFI(high, low, close, volume)

            df['ADX'] = talib.ADX(high, low, close)
            df['PLUS_DI'] = talib.PLUS_DI(high, low, close)
            df['MINUS_DI'] = talib.MINUS_DI(high, low, close)

            typical_price = (high + low + close) / 3
            df['PIVOT'] = typical_price
            df['R1'] = (2 * df['PIVOT']) - low
            df['S1'] = (2 * df['PIVOT']) - high

            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.fillna(method='ffill', inplace=True)
            df.fillna(method='bfill', inplace=True)

            return df

        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            raise

    def prepare_features(self, df: pd.DataFrame, selected_features=None):
        try:
            required_features = self.get_required_features()
            logger.info(f"Model requires {required_features} features")

            df = self.calculate_technical_indicators(df)
            available_features = list(df.columns)

            if selected_features is None or not selected_features:
                selected_features = available_features

            if len(selected_features) < required_features:
                remaining_features = [f for f in available_features if f not in selected_features]
                additional_features = remaining_features[:required_features - len(selected_features)]
                selected_features.extend(additional_features)
            elif len(selected_features) > required_features:
                selected_features = selected_features[:required_features]

            feature_data = df[selected_features].values
            scaled_data = self.scaler.fit_transform(feature_data)

            sequence_length = 10
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

    def clean_numeric_value(self, value):
        if pd.isna(value) or np.isnan(value) or np.isinf(value):
            return None
        return float(value)

    def generate_technical_signals(self, df):
        try:
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14, min_periods=1).mean()
            avg_loss = loss.rolling(window=14, min_periods=1).mean()
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))

            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

            sma = df['Close'].rolling(window=20).mean()
            std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = sma + (std * 2)
            df['BB_Lower'] = sma - (std * 2)

            df['RSI_Buy_Signal'] = ((df['RSI'] < 30) & df['RSI'].notna()).astype(int)
            df['RSI_Sell_Signal'] = ((df['RSI'] > 70) & df['RSI'].notna()).astype(int)
            df['MACD_Buy_Signal'] = ((df['MACD'] > df['MACD_Signal']) & df['MACD'].notna() & df['MACD_Signal'].notna()).astype(int)
            df['MACD_Sell_Signal'] = ((df['MACD'] < df['MACD_Signal']) & df['MACD'].notna() & df['MACD_Signal'].notna()).astype(int)

            df['Technical_Buy_Signal'] = (
                (df['RSI'] < 30) & 
                (df['MACD'] > df['MACD_Signal']) & 
                df['RSI'].notna() & 
                df['MACD'].notna() & 
                df['MACD_Signal'].notna()
            ).astype(int)
            
            df['Technical_Sell_Signal'] = (
                (df['RSI'] > 70) & 
                (df['MACD'] < df['MACD_Signal']) & 
                df['RSI'].notna() & 
                df['MACD'].notna() & 
                df['MACD_Signal'].notna()
            ).astype(int)

            df['BB_Buy_Signal'] = (
                (df['Close'] < df['BB_Lower']) & 
                df['BB_Lower'].notna()
            ).astype(int)
            
            df['BB_Sell_Signal'] = (
                (df['Close'] > df['BB_Upper']) & 
                df['BB_Upper'].notna()
            ).astype(int)

            df = df.replace([np.inf, -np.inf], np.nan)
            for col in df.select_dtypes(include=['float64']).columns:
                df[col] = df[col].where(df[col].notna(), None)

            return df

        except Exception as e:
            logger.error(f"Error generating technical signals: {str(e)}")
            raise

    def analyze_stock(self, df, selected_features):
        try:
            X, timestamps, used_features = self.prepare_features(df, selected_features)
            predictions = self.model.predict(X)
            if len(predictions.shape) > 1:
                predictions = predictions.flatten()

            df = self.generate_technical_signals(df)
            
            signals = []
            prices = df['Close'].values[-len(predictions):]
            
            for i in range(len(predictions)):
                current_price = self.clean_numeric_value(prices[i])
                if current_price is None:
                    continue

                predicted_change = self.clean_numeric_value(predictions[i])
                if predicted_change is None:
                    predicted_change = 0.0
                
                ml_signal = "hold"
                if predicted_change > 1:
                    ml_signal = "buy"
                elif predicted_change < -1:
                    ml_signal = "sell"

                idx = -len(predictions) + i
                tech_buy = bool(df['Technical_Buy_Signal'].iloc[idx] or df['BB_Buy_Signal'].iloc[idx])
                tech_sell = bool(df['Technical_Sell_Signal'].iloc[idx] or df['BB_Sell_Signal'].iloc[idx])

                final_signal = "hold"
                if ml_signal == "buy" and tech_buy:
                    final_signal = "buy"
                    confidence = min(abs(predicted_change) * 20 + 20, 100)
                elif ml_signal == "sell" and tech_sell:
                    final_signal = "sell"
                    confidence = min(abs(predicted_change) * 20 + 20, 100)
                else:
                    final_signal = ml_signal
                    confidence = min(abs(predicted_change) * 20, 100)

                technical_context = {
                    "rsi": self.clean_numeric_value(df['RSI'].iloc[idx]),
                    "macd": self.clean_numeric_value(df['MACD'].iloc[idx]),
                    "macd_signal": self.clean_numeric_value(df['MACD_Signal'].iloc[idx]),
                    "bb_upper": self.clean_numeric_value(df['BB_Upper'].iloc[idx]),
                    "bb_lower": self.clean_numeric_value(df['BB_Lower'].iloc[idx])
                }

                atr = self.clean_numeric_value(df.get('ATR', pd.Series([0])).iloc[idx]) or (current_price * 0.02)
                stop_loss = current_price * (1 - (atr / current_price) * 2) if final_signal == "buy" else current_price * (1 + (atr / current_price) * 2)

                signals.append({
                    "date": timestamps[i].strftime("%Y-%m-%d"),
                    "price": current_price,
                    "predicted_change": predicted_change,
                    "signal": final_signal,
                    "confidence": float(confidence),
                    "stop_loss": float(stop_loss),
                    "technical_indicators": technical_context
                })

            return signals, used_features

        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            raise

    def get_model_metrics(self):
        try:
            metrics = {
                "architecture": {
                    "input_shape": self.input_shape,
                    "layers": [
                        {
                            "name": layer.name,
                            "type": layer.__class__.__name__,
                            "config": layer.get_config()
                        }
                        for layer in self.model.layers
                    ]
                },
                "training_config": self.model.optimizer.get_config() if hasattr(self.model, 'optimizer') else None
            }
            return metrics
        except Exception as e:
            logger.error(f"Error getting model metrics: {str(e)}")
            raise

    def validate_predictions(self, predictions, timestamps):
        try:
            valid_mask = np.isfinite(predictions)
            if not np.all(valid_mask):
                logger.warning(f"Found {np.sum(~valid_mask)} invalid predictions")
                predictions = predictions[valid_mask]
                timestamps = [t for t, m in zip(timestamps, valid_mask) if m]

            clip_threshold = 20.0
            clipped_predictions = np.clip(predictions, -clip_threshold, clip_threshold)
            if not np.array_equal(predictions, clipped_predictions):
                logger.warning("Some predictions were clipped to reasonable range")
                predictions = clipped_predictions

            return predictions, timestamps
        except Exception as e:
            logger.error(f"Error validating predictions: {str(e)}")
            raise

    def calculate_risk_metrics(self, df, prediction, current_price):
        try:
            returns = np.diff(np.log(df['Close'].values[-20:]))
            volatility = np.std(returns) * np.sqrt(252) * 100

            trend_strength = df['ADX'].iloc[-1]

            pivot = df['PIVOT'].iloc[-1]
            support = df['S1'].iloc[-1]
            resistance = df['R1'].iloc[-1]

            volatility_score = min(volatility * 5, 100)
            trend_score = min(trend_strength, 100)
            prediction_magnitude = abs(prediction)
            
            risk_score = (volatility_score * 0.4 + 
                         (100 - trend_score) * 0.3 + 
                         prediction_magnitude * 5 * 0.3)
            risk_score = min(max(risk_score, 0), 100)

            return {
                "volatility": float(volatility),
                "trend_strength": float(trend_strength),
                "support_level": float(support),
                "resistance_level": float(resistance),
                "risk_score": float(risk_score)
            }
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            raise

    def get_market_context(self, df):
        try:
            last_prices = df['Close'].values[-20:]
            recent_change = ((last_prices[-1] / last_prices[0]) - 1) * 100

            sma20 = df['SMA_20'].iloc[-1]
            current_price = df['Close'].iloc[-1]
            trend = "bullish" if current_price > sma20 else "bearish"

            rsi = df['RSI'].iloc[-1]
            momentum = "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral"

            recent_volume = df['Volume'].values[-5:].mean()
            volume_20d_avg = df['Volume'].values[-20:].mean()
            volume_trend = "high" if recent_volume > volume_20d_avg * 1.2 else \
                          "low" if recent_volume < volume_20d_avg * 0.8 else "normal"

            return {
                "market_trend": trend,
                "momentum": momentum,
                "volume_trend": volume_trend,
                "recent_change": float(recent_change),
                "rsi": float(rsi)
            }
        except Exception as e:
            logger.error(f"Error analyzing market context: {str(e)}")
            raise

    def enrich_signals(self, signals, df):
        try:
            for signal in signals:
                current_price = signal["price"]
                predicted_change = signal["predicted_change"]
                
                risk_metrics = self.calculate_risk_metrics(df, predicted_change, current_price)
                signal.update(risk_metrics)
                
                market_context = self.get_market_context(df)
                signal.update(market_context)
                
                if abs(predicted_change) > 2 and risk_metrics["risk_score"] < 70:
                    position_size = 1.0 if risk_metrics["risk_score"] < 50 else 0.5
                    signal["recommended_position_size"] = position_size
                    
                    if predicted_change > 0:
                        signal["take_profit_1"] = current_price * (1 + predicted_change/3/100)
                        signal["take_profit_2"] = current_price * (1 + predicted_change/2/100)
                        signal["take_profit_3"] = current_price * (1 + predicted_change/100)
                    else:
                        signal["take_profit_1"] = current_price * (1 + predicted_change/3/100)
                        signal["take_profit_2"] = current_price * (1 + predicted_change/2/100)
                        signal["take_profit_3"] = current_price * (1 + predicted_change/100)
                
            return signals
        except Exception as e:
            logger.error(f"Error enriching signals: {str(e)}")
            raise

    def backtest_prediction_accuracy(self, df, lookback_days=30):
        try:
            accuracy_metrics = {
                "correct_direction": 0,
                "total_predictions": 0,
                "avg_error": 0.0,
                "profitable_trades": 0
            }
            
            if len(df) <= lookback_days:
                raise ValueError(f"Insufficient data for {lookback_days} day backtest")

            X, timestamps, used_features = self.prepare_features(df)
            
            test_indices = range(len(X) - lookback_days, len(X))
            total_trades = 0
            
            for i in test_indices:
                if i < 0:
                    continue
                    
                prediction = self.model.predict(X[i:i+1])[0][0]
                
                current_price = df['Close'].iloc[i]
                next_price = df['Close'].iloc[i + 1] if i + 1 < len(df) else df['Close'].iloc[i]
                actual_change = ((next_price - current_price) / current_price) * 100
                
                if (prediction > 0 and actual_change > 0) or (prediction < 0 and actual_change < 0):
                    accuracy_metrics["correct_direction"] += 1
                    accuracy_metrics["profitable_trades"] += 1
                
                prediction_error = abs(prediction - actual_change)
                accuracy_metrics["avg_error"] += prediction_error
                accuracy_metrics["total_predictions"] += 1
                total_trades += 1
                logger.debug(f"Prediction error for trade {total_trades}: {prediction_error}%")

            if accuracy_metrics["total_predictions"] > 0:
                accuracy_metrics["direction_accuracy"] = (
                    accuracy_metrics["correct_direction"] / 
                    accuracy_metrics["total_predictions"] * 100
                )
                accuracy_metrics["avg_error"] = (
                    accuracy_metrics["avg_error"] / 
                    accuracy_metrics["total_predictions"]
                )
                accuracy_metrics["profitability"] = (
                    accuracy_metrics["profitable_trades"] / 
                    accuracy_metrics["total_predictions"] * 100
                )
            
            accuracy_metrics["total_trades"] = total_trades
            accuracy_metrics["lookback_period"] = {
                "start": df.index[-lookback_days].strftime("%Y-%m-%d"),
                "end": df.index[-1].strftime("%Y-%m-%d")
            }
            accuracy_metrics["features_used"] = used_features
            
            return accuracy_metrics
        except Exception as e:
            logger.error(f"Error in backtesting: {str(e)}")
            raise
