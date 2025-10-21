"""Firebase initialization helpers for the hypothesis agent."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Optional

import firebase_admin
from firebase_admin import credentials, firestore

from hypothesis_agent.config import AppSettings


@dataclass(slots=True)
class FirebaseHandle:
    """Thin wrapper around a Firebase app and Firestore client."""

    app: firebase_admin.App
    client: firestore.Client
    collection: str

    async def dispose(self) -> None:
        """Dispose of the Firebase app instance."""

        def _delete_app() -> None:
            try:
                firebase_admin.delete_app(self.app)
            except ValueError:
                # App already deleted or never initialised; nothing to do.
                pass

        await asyncio.to_thread(_delete_app)


def initialize_firebase(settings: AppSettings) -> FirebaseHandle:
    """Initialise Firebase Admin SDK and return a handle for Firestore access."""

    options: Dict[str, Any] = {}
    if settings.firebase_project_id:
        options["projectId"] = settings.firebase_project_id

    credential: Optional[credentials.Base] = None
    if settings.firebase_credentials_path:
        credential = credentials.Certificate(settings.firebase_credentials_path)
    else:
        try:
            credential = credentials.ApplicationDefault()
        except Exception:
            credential = None

    app_name = settings.firebase_app_name
    try:
        app = firebase_admin.get_app(app_name) if app_name else firebase_admin.get_app()
    except ValueError:
        app = firebase_admin.initialize_app(
            credential=credential,
            options=options or None,
            name=app_name if app_name else "[DEFAULT]",
        )

    client = firestore.client(app=app)
    return FirebaseHandle(app=app, client=client, collection=settings.firebase_collection)
