"""Bootstrap MLTK onto sys.path so that ``import mltk`` works whether or
not MLTK is installed as a package. We vendor it as a git submodule under
``modules/MLTK``; absolute imports inside MLTK itself need its repo root on
``sys.path``.

Importing anything from the ``modules`` package implicitly runs this — no
caller needs to repeat the path-fix incantation."""
import os
import sys

_MLTK_ROOT = os.path.join(os.path.dirname(__file__), "MLTK")
if os.path.isdir(_MLTK_ROOT) and _MLTK_ROOT not in sys.path:
    sys.path.insert(0, _MLTK_ROOT)
