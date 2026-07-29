"""Public package exports.

`StateCodebook` and `CodeWAM` are legacy regression exports. `CodeWAM` pulls in
the external `fastwam` package, so it remains lazy. The independent v1 exports
do not import FastWAM.
"""

from codewam.codebook import StateCodebook
from codewam.models import CodeWAMConfig, CodeWAMV1, build_codewam_v1


def __getattr__(name):
    if name == "CodeWAM":
        from codewam.model import CodeWAM

        return CodeWAM
    raise AttributeError(name)

__all__ = [
    "CodeWAM",
    "CodeWAMConfig",
    "CodeWAMV1",
    "StateCodebook",
    "build_codewam_v1",
]
