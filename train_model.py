"""Legacy compatibility stub.

TRACE-ML v3 uses the new package runtime:
    trace-ml train rebuild
"""

from trace_ml.cli import app


if __name__ == "__main__":
    app(["train", "rebuild"])
