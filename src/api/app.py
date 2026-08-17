"""FastAPI application factory and local executable entry point."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.api.routes import router
from src.config import load_yaml
from src.logging_config import configure_logging
from src.pipeline import VehicleAnalysisPipeline

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = PROJECT_ROOT / "src" / "web"


def create_app(
    pipeline: VehicleAnalysisPipeline | None = None,
    configuration: dict[str, Any] | None = None,
) -> FastAPI:
    """Create the local web application without starting external services."""
    config = configuration or load_yaml(PROJECT_ROOT / "configs" / "default.yaml")
    api_config = config.get("api", {})

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        configure_logging()
        if application.state.pipeline is None:
            LOGGER.info("Loading configured vehicle detector")
            application.state.pipeline = VehicleAnalysisPipeline.from_config(config, PROJECT_ROOT)
        yield

    app = FastAPI(
        title="Drone Vehicle Inspection",
        version="0.1.0",
        description="Synchronous vehicle detection for aerial images.",
        lifespan=lifespan,
    )
    app.state.pipeline = pipeline
    app.state.max_upload_bytes = int(api_config.get("max_upload_mb", 12)) * 1024 * 1024
    app.state.max_pixels = int(api_config.get("max_pixels", 20_000_000))
    app.mount("/static", StaticFiles(directory=WEB_ROOT / "static"), name="static")
    templates = Jinja2Templates(directory=WEB_ROOT / "templates")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def render_home(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, "index.html")

    app.include_router(router)
    return app


app = create_app()


def main() -> None:
    """Run the application with the optional APP_PORT environment override."""
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=int(os.getenv("APP_PORT", "8000")))


if __name__ == "__main__":
    main()
