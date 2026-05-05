import functools
import uuid
import threading

# Try to use contextvars (Python 3.7+); fall back to threading.local
try:
    import contextvars
    _CTX_VAR = contextvars.ContextVar("context_id")

    def set_context_id(cid: str = None) -> str:
        """
        Generate (or reuse) and store a context UUID in the current context.
        """
        cid = cid or str(uuid.uuid4())
        _CTX_VAR.set(cid)
        return cid

    def get_context_id() -> str:
        """
        Retrieve the context UUID, creating one if needed.
        """
        try:
            return _CTX_VAR.get()
        except LookupError:
            return set_context_id()

    def clear_context_id() -> None:
        """
        Clear the stored context UUID.
        """
        # setting to None so get_context_id will regenerate next time
        _CTX_VAR.set(None)

except ImportError:
    _TLS = threading.local()

    def set_context_id(cid: str = None) -> str:
        cid = cid or str(uuid.uuid4())
        _TLS.context_id = cid
        return cid

    def get_context_id() -> str:
        if not hasattr(_TLS, "context_id") or _TLS.context_id is None:
            return set_context_id()
        return _TLS.context_id

    def clear_context_id() -> None:
        _TLS.context_id = None


def with_context(target):
    if isinstance(target, type):
        original_init = target.__init__

        @functools.wraps(original_init)
        def wrapped_init(self, *args, **kwargs):
            set_context_id()
            try:
                original_init(self, *args, **kwargs)
            finally:
                clear_context_id()

        target.__init__ = wrapped_init
        return target

    @functools.wraps(target)
    def wrapper(*args, **kwargs):
        set_context_id()        # new UUID at start
        try:
            return target(*args, **kwargs)
        finally:
            clear_context_id()  # reset afterwards
    return wrapper
