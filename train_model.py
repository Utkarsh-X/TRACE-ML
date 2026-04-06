"""Legacy compatibility stub.

TRACE-AML v3 uses the new package runtime:
    trace-aml train rebuild
"""

from trace_aml.cli import app


if __name__ == "__main__":
    app(["train", "rebuild"])
