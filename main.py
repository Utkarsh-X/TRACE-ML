"""Legacy entrypoint compatibility.

TRACE-AML v3 official entrypoint:
    trace-aml --help
"""

from trace_aml.cli import app


if __name__ == "__main__":
    app()
