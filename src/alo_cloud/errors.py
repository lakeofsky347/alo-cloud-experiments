"""Execution-contract failures; no error grants permission to retry a request."""


class PilotStopped(RuntimeError):
    """Preserved adapter exception: stop execution and retain the existing state."""


class ProtocolError(ValueError):
    """An action or public response violates its frozen schema."""


class BaselineUnavailable(RuntimeError):
    """A required native framework or service contract is unavailable."""


class IntegrationNotConfigured(NotImplementedError):
    """A production boundary has deliberately not been connected in this scaffold."""
