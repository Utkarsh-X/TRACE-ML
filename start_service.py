#!/usr/bin/env python3
"""Start TRACE-AML service with FastAPI and Uvicorn."""

import sys
from pathlib import Path

try:
    import uvicorn
    from trace_aml.core.config import load_settings
    from trace_aml.core.logger import configure_logger
    from trace_aml.core.health import run_health_checks
    from trace_aml.service.app import create_service_app
    from trace_aml.store.vector_store import VectorStore
    from trace_aml.recognizers.arcface import ArcFaceRecognizer
    from trace_aml.pipeline.session import RecognitionSession
    from trace_aml.core.streaming import InMemoryEventStreamPublisher
    from loguru import logger
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Ensure all dependencies are installed: pip install -e .")
    sys.exit(1)


def main():
    print("╭────────────────────────────────────────────────╮")
    print("│ TRACE-AML Service Startup                      │")
    print("│ FastAPI + Uvicorn                              │")
    print("╰────────────────────────────────────────────────╯\n")
    
    # Load settings
    settings = load_settings()
    configure_logger(settings)
    
    # Run health checks
    logger.info("Running health checks...")
    checks = run_health_checks(settings)
    failures = [c for c in checks if c.status != "OK"]
    if failures:
        logger.error(f"Health checks failed with {len(failures)} issues:")
        for check in failures:
            logger.error(f"  - {check.name}: {check.detail}")
        return 1
    logger.info(f"✓ Health checks passed ({len(checks)} checks)")
    
    # Initialize storage
    logger.info("Initializing storage...")
    store = VectorStore(settings)
    logger.info("✓ VectorStore initialized")
    
    # Initialize recognition session (for camera control)
    logger.info("Initializing recognition session...")
    recognizer = ArcFaceRecognizer(settings)
    stream_publisher = InMemoryEventStreamPublisher()
    session = RecognitionSession(
        settings=settings,
        store=store,
        recognizer=recognizer,
        stream_publisher=stream_publisher,
    )
    logger.info("✓ RecognitionSession initialized")
    
    # Create FastAPI app
    logger.info("Creating FastAPI app...")
    app = create_service_app(
        settings=settings,
        store=store,
        stream_publisher=stream_publisher,
        session=session,
    )
    logger.info("✓ FastAPI app created")
    
    # Start Uvicorn
    logger.info(f"\n🚀 Starting Uvicorn on http://0.0.0.0:8080\n")
    logger.info("📘 Access the UI at: http://localhost:8080/ui/live_ops/index.html")
    logger.info("📘 API docs: http://localhost:8080/docs\n")
    
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8080,
            log_level="info",
            access_log=True,
        )
    except KeyboardInterrupt:
        logger.info("\n\n✓ Service shutdown gracefully")
        return 0
    except Exception as e:
        logger.error(f"Service error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
