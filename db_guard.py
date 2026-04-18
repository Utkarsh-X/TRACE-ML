"""
Guard: check for running service before touching the DB.
Add this to the top of any diagnostic/reset script.
"""
import socket
import sys


def abort_if_service_running(port: int = 8080) -> None:
    """Exit immediately if the TRACE-AML service appears to be running."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        result = s.connect_ex(("127.0.0.1", port))
    if result == 0:
        print(
            f"\n[ABORT] TRACE-AML service is running on port {port}.\n"
            "Stop the service first (kill the terminal running start.ps1) "
            "before running database scripts.\n"
            "Running two LanceDB connections simultaneously can corrupt the database.\n"
        )
        sys.exit(1)
