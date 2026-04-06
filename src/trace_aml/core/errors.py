"""Application-level error hierarchy."""


class TraceAMLError(Exception):
    """Base error for TRACE-AML."""


class ConfigError(TraceAMLError):
    """Raised for configuration problems."""


class DependencyError(TraceAMLError):
    """Raised when required dependencies are missing."""


class StorageError(TraceAMLError):
    """Raised for persistence and query failures."""


class RecognitionError(TraceAMLError):
    """Raised for model loading and recognition failures."""


class CameraError(TraceAMLError):
    """Raised for webcam and frame pipeline failures."""
