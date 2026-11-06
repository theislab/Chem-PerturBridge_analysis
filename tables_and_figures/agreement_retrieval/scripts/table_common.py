"""Shared dataset labels and manuscript table row order."""

DATASET_DISPLAY = {
    "cigs_mce": "CIGS-MCE",
    "cigs_tcm": "CIGS-TCM",
    "l1000_phase1": "L1000 Phase I",
    "l1000_phase2": "L1000 Phase II",
    "sciplex": "sci-Plex",
    "tahoe": "Tahoe-100M",
}

PAIR_ORDER = [
    ("cigs_mce", "cigs_tcm"),
    ("l1000_phase1", "cigs_mce"),
    ("l1000_phase1", "l1000_phase2"),
    ("l1000_phase1", "sciplex"),
    ("l1000_phase1", "tahoe"),
    ("l1000_phase2", "cigs_mce"),
    ("l1000_phase2", "sciplex"),
    ("l1000_phase2", "tahoe"),
    ("tahoe", "sciplex"),
]

