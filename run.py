"""
run.py
======
Convenience launcher for the IS 6403:1981 Bearing Capacity Web Estimator.

Usage
-----
    python run.py
    python run.py --host 0.0.0.0 --port 8080 --reload

Opens the browser automatically at http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from threading import Timer


def open_browser(url: str) -> None:
    """Open the default browser after a short delay."""
    webbrowser.open(url)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IS 6403:1981 Bearing Capacity Web Estimator"
    )
    parser.add_argument("--host",   default="127.0.0.1", help="Host address")
    parser.add_argument("--port",   default=8000, type=int, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    print(f"\n  IS 6403:1981 Bearing Capacity Estimator")
    print(f"  -----------------------------------------")
    print(f"  Server : {url}")
    print(f"  API docs : {url}/docs")
    print(f"  Press CTRL+C to stop.\n")

    # Open browser after 1.5 s
    Timer(1.5, open_browser, [url]).start()

    # pyrefly: ignore [missing-import]
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
