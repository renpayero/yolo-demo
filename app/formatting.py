"""Formatos de texto compartidos (video, alertas, API). Sin dependencias."""


def format_duration(seconds: float) -> str:
    """12 → 0:12 · 754 → 12:34 · 3754 → 1:02:34"""
    total = max(0, int(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"
