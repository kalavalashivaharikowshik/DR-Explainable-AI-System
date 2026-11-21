"""Filesystem anchors so the app works no matter which directory it is started from."""

import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
