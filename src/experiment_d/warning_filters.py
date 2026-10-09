"""Hide the two warnings that every model load prints and that carry no information."""
import logging
import warnings

_HARMLESS_TERMS = ("torch_dtype", "MatMul8bitLt")


class _HarmlessWarningFilter(logging.Filter):
    """Filter warnings emitted through the logging module (e.g. transformers logger)."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(term in msg for term in _HARMLESS_TERMS)


def silence_harmless_warnings():
    # (?s) allows '.' to match newlines across multiline warning messages
    for term in _HARMLESS_TERMS:
        warnings.filterwarnings("ignore", message=rf"(?s).*{term}.*")

    log_filter = _HarmlessWarningFilter()
    loggers = [logging.getLogger()]
    try:
        from transformers.utils import logging as transformers_logging
        loggers.append(transformers_logging.get_logger())
    except ImportError:
        pass
    for logger in loggers:
        logger.addFilter(log_filter)
        for handler in logger.handlers:
            handler.addFilter(log_filter)
