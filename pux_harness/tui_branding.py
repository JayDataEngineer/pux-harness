"""Org-specific branding for the dcode TUI landing page.

Provides custom ASCII art banners and subheaders that replace the default
"Deep Agents" splash when ``pux tui`` launches dcode.

The font style matches the original dcode banner: 6-row box-drawing glyphs
using the ``█╔╗╚╝═║`` block set, with proportional letter widths (8 chars).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Custom ASCII art banner
# Font: same 6-row box-drawing style as the original dcode banner
# Glyphs are 8-char wide, 6-row tall; built from P-U-X with 1-space separator
# ---------------------------------------------------------------------------

_P_PUX: list[str] = [
    "██████╗ ",  # P row 1
    "██╔══██╗",  # P row 2
    "██████╔╝",  # P row 3
    "██╔═══╝ ",  # P row 4
    "██║     ",  # P row 5
    "╚═╝     ",  # P row 6
]

_U_PUX: list[str] = [
    "██╗  ██╗",  # U row 1
    "██║  ██║",  # U row 2
    "██║  ██║",  # U row 3
    "██║  ██║",  # U row 4
    " █████╔╝",  # U row 5
    " ╚════╝ ",  # U row 6
]

_X_PUX: list[str] = [
    "██╗  ██╗",  # X row 1
    "╚██╗██╔╝",  # X row 2
    " ╚███╔╝ ",  # X row 3
    " ██╔██╗ ",  # X row 4
    "██╔╝ ██╗",  # X row 5
    "╚═╝  ╚═╝",  # X row 6
]

_PUX_BANNER = "\n".join(p + " " + u + " " + x for p, u, x in zip(_P_PUX, _U_PUX, _X_PUX))

_PUX_UNICODE_BANNER = _PUX_BANNER
_PUX_ASCII_BANNER = _PUX_BANNER


def get_pux_banner(unicode_mode: bool = True) -> str:
    """Return the Pux ASCII art banner.

    Args:
        unicode_mode: Ignored — the box-drawing characters render correctly
            in both terminal modes.
    """
    return _PUX_BANNER


# ---------------------------------------------------------------------------
# Per-org subheaders
# ---------------------------------------------------------------------------

ORG_BRANDING: dict[str, dict[str, str]] = {
    "dev-bot": {
        "subheader": "Engineering mode. What are we building?",
    },
    "deep-research-engine": {
        "subheader": "Research mode. What should I investigate?",
    },
    "game-studio": {
        "subheader": "Studio mode. What's the creative brief?",
    },
    "invest": {
        "subheader": "Trading mode. What's the market signal?",
    },
    "social-media-pipeline": {
        "subheader": "Content mode. What are we publishing?",
    },
    "telegram-agent": {
        "subheader": "Telegram mode. What should I send?",
    },
    "twitter-agent": {
        "subheader": "Twitter mode. What's the tweet?",
    },
    "video-production": {
        "subheader": "Production mode. What are we rendering?",
    },
}

DEFAULT_BRANDING: dict[str, str] = {
    "subheader": "Pux ready. What would you like to do?",
}


def get_branding(org: str) -> dict[str, str]:
    """Return branding overrides for *org*, falling back to defaults."""
    return ORG_BRANDING.get(org, DEFAULT_BRANDING)
