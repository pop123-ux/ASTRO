"""Visual identities for the clean paper campaign.

The legacy figure palette remains untouched so historical figures reproduce
exactly.  New paper-campaign plots import this module instead.
"""

from style import *  # noqa: F401,F403
from style import COLORS, LABELS, MARKERS

COLORS.update({
    "adamuon_ref": COLORS["adamuon"],
    "muon_m90": "#F2A65A",
    "astro_v2_gamma0_nosplit": "#B9A7C8",
    "astro_v2_nosplit": "#A16CC1",
})

MARKERS.update({
    "adamuon_ref": "v",
    "muon_m90": "s",
    "astro_v2_gamma0_nosplit": "o",
    "astro_v2_nosplit": "h",
})

LABELS.update({
    "adamuon_ref": "AdaMuon (official rule)",
    "muon_m90": r"Muon ($\beta_1=0.90$)",
    "astro_v2_gamma0_nosplit": r"ASTRO control ($\gamma=0$, fused QKV)",
    "astro_v2_nosplit": "ASTRO-v2 (fused QKV)",
})
