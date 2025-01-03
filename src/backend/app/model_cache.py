import tensorflow as tf
from datetime import datetime, timedelta
import logging
from typing import Dict, Optional, Tuple, List, Union

class ModelCacheManager:
    def __init__(self, cache_timeout: int = 300):
        self._cache: Dict[str, Tuple[tf.keras.Model, datetime]] = {}
        self._cache_timeout = timedelta(seconds=cache_timeout)
        self._model_info_cache: Dict[str, dict] = {}

    def get_model(self, model_path: str) -> Optional[tf.keras.Model]:
        now = datetime.now()
        if model_path in self._cache:
            model, timestamp = self._cache[model_path]
            if now - timestamp < self._cache_timeout:
                return model
        try:
            model = tf.keras.models.load_model(model_path)
            self._cache[model_path] = (model, now)
            return model
        except Exception as e:
            return None

    def get_model_info(self, model_path: str) -> Optional[dict]:
        if model_path in self._model_info_cache:
            return self._model_info_cache[model_path]
        model = self.get_model(model_path)
        if not model:
            return None
        input_shape = model.input_shape
        required_features = input_shape[-1] if len(input_shape) >= 3 else 1
        info = {
            "input_shape": input_shape,
            "required_features": required_features,
            "model_type": model.__class__.__name__,
            "layer_config": []
        }
        for layer in model.layers:
            layer_info = {
                "name": layer.name,
                "type": layer.__class__.__name__,
                "input_shape": self._get_layer_input_shape(layer),
                "output_shape": self._get_layer_output_shape(layer)
            }
            info["layer_config"].append(layer_info)
        self._model_info_cache[model_path] = info
        return info

    def _get_shape_as_list(self, shape: Union[tf.TensorShape, Tuple[int, ...]]) -> List[int]:
        if isinstance(shape, tf.TensorShape):
            return shape.as_list()
        elif isinstance(shape, tuple):
            return list(shape)
        else:
            return "Unavailable"

    def _get_layer_input_shape(self, layer) -> Union[List[int], str]:
        if hasattr(layer, 'input_shape') and layer.input_shape is not None:
            return self._get_shape_as_list(layer.input_shape)
        elif hasattr(layer, 'input') and hasattr(layer.input, 'shape') and layer.input.shape is not None:
            return self._get_shape_as_list(layer.input.shape)
        else:
            return "Unavailable"

    def _get_layer_output_shape(self, layer) -> Union[List[int], str]:
        if hasattr(layer, 'output_shape') and layer.output_shape is not None:
            return self._get_shape_as_list(layer.output_shape)
        elif hasattr(layer, 'output') and hasattr(layer.output, 'shape') and layer.output.shape is not None:
            return self._get_shape_as_list(layer.output.shape)
        else:
            return "Unavailable"

    def clear_cache(self):
        self._cache.clear()
        self._model_info_cache.clear()

model_cache = ModelCacheManager()
