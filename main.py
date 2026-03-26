"""Legacy entrypoint compatibility.

TRACE-ML v3 official entrypoint:
    trace-ml --help
"""

from trace_ml.cli import app


if __name__ == "__main__":
    app()
