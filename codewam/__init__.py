"""Public package exports.

`StateCodebook` and `CodeWAM` are legacy regression exports. `CodeWAM` pulls in
the external `fastwam` package, so it remains lazy until the independent model
package replaces this compatibility alias.
"""

from codewam.codebook import StateCodebook


def __getattr__(name):
    if name == "CodeWAM":
        from codewam.model import CodeWAM

        return CodeWAM
    raise AttributeError(name)

__all__ = ["CodeWAM", "StateCodebook"]
