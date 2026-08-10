from bullsense.features.external_hft_features import register_hft_features


def bootstrap_registry():
    """
    Import all external HFT feature modules once and return the live registry.
    """
    return register_hft_features()
