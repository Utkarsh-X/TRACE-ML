"""Application-level error hierarchy."""


class TraceMLError(Exception):
    """Base error for TRACE-ML."""


class ConfigError(TraceMLError):
    """Raised for configuration problems."""


class DependencyError(TraceMLError):
    """Raised when required dependencies are missing."""


class StorageError(TraceMLError):
    """Raised for persistence and query failures."""


class RecognitionError(TraceMLError):
    """Raised for model loading and recognition failures."""


class CameraError(TraceMLError):
    """Raised for webcam and frame pipeline failures."""
