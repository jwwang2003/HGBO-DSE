from backend.mcp_auth import ApiKeyAuthMiddleware, ApiKeyStore, extract_api_key


def key_mode(path):
    return path.stat().st_mode & 0o777


async def ok_app(scope, receive, send):
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"text/plain; charset=utf-8")],
        }
    )
    await send({"type": "http.response.body", "body": b"ok"})


def test_api_key_store_generates_and_reuses_key(tmp_path):
    key_path = tmp_path / "api_key"
    store = ApiKeyStore(key_path)

    first = store.ensure_api_key()
    second = store.ensure_api_key()

    assert first
    assert first == second
    assert key_path.read_text(encoding="utf-8").strip() == first
    assert key_mode(key_path) == 0o600


def test_api_key_store_reuses_existing_non_empty_key(tmp_path):
    key_path = tmp_path / "api_key"
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text("existing-secret\n", encoding="utf-8")

    assert ApiKeyStore(key_path).ensure_api_key() == "existing-secret"


def test_api_key_store_tightens_existing_key_permissions(tmp_path):
    key_path = tmp_path / "api_key"
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text("existing-secret\n", encoding="utf-8")
    key_path.chmod(0o644)

    assert ApiKeyStore(key_path).ensure_api_key() == "existing-secret"
    assert key_mode(key_path) == 0o600


def test_api_key_store_rejects_empty_existing_key(tmp_path):
    key_path = tmp_path / "api_key"
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text("   \n", encoding="utf-8")

    try:
        ApiKeyStore(key_path).ensure_api_key()
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("expected ValueError for empty key file")


def test_write_api_key_for_tests_persists_key(tmp_path):
    key_path = tmp_path / "api_key"

    ApiKeyStore(key_path).write_api_key_for_tests("test-secret")

    assert key_path.read_text(encoding="utf-8").strip() == "test-secret"
    assert key_mode(key_path) == 0o600


def test_extract_api_key_accepts_bearer_and_x_api_key_headers():
    assert extract_api_key([(b"authorization", b"Bearer abc123")]) == "abc123"
    assert extract_api_key([(b"x-api-key", b"xyz789")]) == "xyz789"


def test_api_key_middleware_rejects_invalid_key(tmp_path):
    import asyncio

    store = ApiKeyStore(tmp_path / "api_key")
    store.write_api_key_for_tests("secret")
    sent = []

    middleware = ApiKeyAuthMiddleware(ok_app, store=store, protected_prefix="/mcp")
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "headers": [(b"authorization", b"Bearer wrong")],
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    asyncio.run(middleware(scope, receive, send))

    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 401


def test_api_key_middleware_rejects_non_ascii_authorization_key(tmp_path):
    import asyncio

    store = ApiKeyStore(tmp_path / "api_key")
    store.write_api_key_for_tests("secret")
    sent = []

    middleware = ApiKeyAuthMiddleware(ok_app, store=store, protected_prefix="/mcp")
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "headers": [(b"authorization", "Bearer secrét".encode("utf-8"))],
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    asyncio.run(middleware(scope, receive, send))

    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 401


def test_api_key_middleware_rejects_non_ascii_x_api_key(tmp_path):
    import asyncio

    store = ApiKeyStore(tmp_path / "api_key")
    store.write_api_key_for_tests("secret")
    sent = []

    middleware = ApiKeyAuthMiddleware(ok_app, store=store, protected_prefix="/mcp")
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "headers": [(b"x-api-key", "secrét".encode("utf-8"))],
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    asyncio.run(middleware(scope, receive, send))

    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 401


def test_api_key_middleware_rejects_missing_key(tmp_path):
    import asyncio

    store = ApiKeyStore(tmp_path / "api_key")
    store.write_api_key_for_tests("secret")
    sent = []

    middleware = ApiKeyAuthMiddleware(ok_app, store=store, protected_prefix="/mcp")
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp/messages",
        "headers": [],
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    asyncio.run(middleware(scope, receive, send))

    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 401
    assert sent[1]["type"] == "http.response.body"
    assert b"Unauthorized" in sent[1]["body"]


def test_api_key_middleware_passes_valid_key(tmp_path):
    import asyncio

    store = ApiKeyStore(tmp_path / "api_key")
    store.write_api_key_for_tests("secret")
    sent = []

    middleware = ApiKeyAuthMiddleware(ok_app, store=store, protected_prefix="/mcp")
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "headers": [(b"authorization", b"Bearer secret")],
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    asyncio.run(middleware(scope, receive, send))

    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 200


def test_api_key_middleware_passes_unrelated_paths_without_key(tmp_path):
    import asyncio

    store = ApiKeyStore(tmp_path / "api_key")
    store.write_api_key_for_tests("secret")
    sent = []

    middleware = ApiKeyAuthMiddleware(ok_app, store=store, protected_prefix="/mcp")
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/health",
        "headers": [],
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    asyncio.run(middleware(scope, receive, send))

    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 200


def test_api_key_middleware_does_not_protect_prefix_siblings(tmp_path):
    import asyncio

    store = ApiKeyStore(tmp_path / "api_key")
    store.write_api_key_for_tests("secret")

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    for path in ("/mcp-public", "/mcping"):
        sent = []
        middleware = ApiKeyAuthMiddleware(ok_app, store=store, protected_prefix="/mcp")
        scope = {
            "type": "http",
            "method": "GET",
            "path": path,
            "headers": [],
        }

        async def send(message):
            sent.append(message)

        asyncio.run(middleware(scope, receive, send))

        assert sent[0]["type"] == "http.response.start"
        assert sent[0]["status"] == 200
