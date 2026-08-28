class InvOptLabError(Exception):
    """Base exception for invoptlab."""


class ValidationError(InvOptLabError):
    """Raised when data or model validation fails."""


class CapabilityError(InvOptLabError):
    """Raised when a method requires a capability the problem does not provide."""


class SolverError(InvOptLabError):
    """Raised when a forward or inverse solver fails."""

