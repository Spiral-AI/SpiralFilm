class MaxRetriesExceededError(Exception):
    """Raised when the maximum number of retries is exceeded."""

    pass


class ContentFilterError(Exception):
    """Raised when the API response has a finish_reason of 'content_filter'."""

    pass
