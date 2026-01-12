class ConfigKeyError(KeyError):
    """Error raised when a required config key is missing."""

    def __init__(self, key_path):
        self.key_path = " -> ".join(key_path)
        super().__init__(f"Missing required configuration key: {self.key_path}")
