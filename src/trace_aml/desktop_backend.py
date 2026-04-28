"""Desktop backend entrypoint for packaged TRACE-AML builds.

This delegates to the main CLI so packaged mode preserves the same startup
behavior as development mode, including `.env` loading and TRACE_DATA_ROOT
resolution before config defaults are evaluated.
"""

from trace_aml.cli import app


if __name__ == "__main__":
    app()
