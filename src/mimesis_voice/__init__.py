"""Mimesis v2 — one voice engine, N voices, discriminator in the loop.

Public surface is intentionally small: the CLI (`mimesis`) and the MCP server
(`mimesis_voice.server`) are the entry points. Everything else is a library
module those two compose. See docs/DESIGN.md for the architecture.
"""
from __future__ import annotations

__version__ = "2.0.0"
__all__ = ["__version__"]
