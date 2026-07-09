"""Public package exports.

`StateCodebook` is intentionally cheap to import for probe scripts; `CodeWAM`
pulls in the external `fastwam` package, so load it lazily.
"""

from codewam.codebook import StateCodebook


def __getattr__(name):
    if name == "CodeWAM":
        from codewam.model import CodeWAM

        return CodeWAM
    raise AttributeError(name)

__all__ = ["CodeWAM", "StateCodebook"]
