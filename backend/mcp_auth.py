from __future__ import annotations

import json
import os
import secrets
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, MutableMapping, Optional, Tuple


Header = Tuple[bytes, bytes]
Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


class ApiKeyStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def ensure_api_key(self) -> str:
        if self.path.exists():
            return self._read_existing_key()

        key = secrets.token_urlsafe(32)
        self._write_key(key)
        return key

    def write_api_key_for_tests(self, key: str) -> None:
        key = key.strip()
        if not key:
            raise ValueError("api key for tests cannot be empty")
        self._write_key(key)

    def _read_existing_key(self) -> str:
        key = self.path.read_text(encoding="utf-8").strip()
        if not key:
            raise ValueError("api key file is empty")
        self.path.chmod(0o600)
        return key

    def _write_key(self, key: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_WRONLY | os.O_TRUNC
        fd = os.open(self.path, flags, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as key_file:
                fd = -1
                key_file.write(f"{key}\n")
        finally:
            if fd >= 0:
                os.close(fd)
        os.chmod(self.path, 0o600)


def extract_api_key(headers: Iterable[Header]) -> Optional[str]:
    for name, value in headers:
        header_name = name.decode("latin-1").lower()
        header_value = value.decode("latin-1").strip()

        if header_name == "authorization":
            scheme, _, token = header_value.partition(" ")
            if scheme.lower() == "bearer" and token.strip():
                return token.strip()

        if header_name == "x-api-key" and header_value:
            return header_value

    return None


class ApiKeyAuthMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        *,
        store: ApiKeyStore,
        protected_prefix: str = "/mcp",
    ) -> None:
        self.app = app
        self.store = store
        self.protected_prefix = protected_prefix.rstrip("/") or "/"

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = scope.get("path", "")
        if scope.get("type") != "http" or not self._is_protected_path(path):
            await self.app(scope, receive, send)
            return

        expected_key = self.store.ensure_api_key()
        provided_key = extract_api_key(scope.get("headers", []))
        if not provided_key or not self._keys_match(provided_key, expected_key):
            await self._send_unauthorized(send)
            return

        await self.app(scope, receive, send)

    def _is_protected_path(self, path: str) -> bool:
        if self.protected_prefix == "/":
            return path.startswith("/")
        return path == self.protected_prefix or path.startswith(
            f"{self.protected_prefix}/"
        )

    def _keys_match(self, provided_key: str, expected_key: str) -> bool:
        try:
            provided_bytes = provided_key.encode("ascii")
            expected_bytes = expected_key.encode("ascii")
        except UnicodeEncodeError:
            return False
        return secrets.compare_digest(provided_bytes, expected_bytes)

    async def _send_unauthorized(self, send: Send) -> None:
        body = json.dumps({"detail": "Unauthorized"}).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
