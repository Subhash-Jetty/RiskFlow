import json
import datetime
from pathlib import Path
import pandas as pd
from src import config

class FeatureRegistry:
    def __init__(self, registry_path=None):
        self.registry_path = registry_path if registry_path else config.FEATURE_REGISTRY_FILE
        self.features = {}
        if Path(self.registry_path).exists():
            with open(self.registry_path, 'r') as f:
                self.features = json.load(f)

    def register_feature(self, name, dtype, source, computation, is_lagged=False, version='1.0'):
        self.features[name] = {
            'name': name,
            'dtype': dtype,
            'source': source,
            'computation': computation,
            'is_lagged': is_lagged,
            'version': version,
            'registered_at': datetime.datetime.now().isoformat()
        }

    def get_feature(self, name):
        return self.features.get(name)

    def list_features(self):
        return list(self.features.values())

    def validate_features(self, feature_names):
        valid = [f for f in feature_names if f in self.features]
        missing = [f for f in feature_names if f not in self.features]
        return valid, missing

    def save(self):
        Path(self.registry_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, 'w') as f:
            json.dump(self.features, f, indent=4)

    def to_dataframe(self):
        return pd.DataFrame(self.list_features())

def build_default_registry():
    registry = FeatureRegistry()
    for f in config.NUMERIC_RAW:
        registry.register_feature(f, 'numeric', 'raw', 'direct from data', False)
    for f in config.BINARY_RAW:
        registry.register_feature(f, 'binary', 'raw', 'direct from data', False)
    for f in config.CATEGORICAL_RAW:
        registry.register_feature(f, 'categorical', 'raw', 'label encoded', False)
    for f in config.DERIVED_FEATURES:
        is_lagged = 'rolling' in f or 'lag' in f
        computation = f"{f} computation"
        registry.register_feature(f, 'numeric', 'engineered', computation, is_lagged)
    registry.save()
    return registry

def run_feature_store(feature_names):
    registry = build_default_registry()
    valid, missing = registry.validate_features(feature_names)
    return {
        'registry': registry,
        'valid': valid,
        'missing': missing
    }
