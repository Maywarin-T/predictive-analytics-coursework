"""
Load and merge Living Costs and Food Survey (LCFS) household data.

This module handles loading raw LCFS tab-separated data files across multiple
survey years (2021-2023), harmonising column names, and computing derived
variables like the OECD modified equivalence scale and equivalised income.

Data source: UK Data Service
- UKDA-9123 (2021-2022)
- UKDA-9335 (2022-2023)
- UKDA-9468 (2023-2024)
"""

import os
import numpy as np
import pandas as pd


# ── Path configuration ──────────────────────────────────────────────────────
# Resolve the data directory relative to the location of this file (src/)
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# Mapping of survey year to the corresponding raw data filename
FILES = {
    2021: 'lcfs_2021_dvhh_ukanon.tab',
    2022: 'dvhh_ukanon_2022.tab',
    2023: 'dvhh_ukanon_v2_2023.tab',
}

# ── Column harmonisation ────────────────────────────────────────────────────
# The 2021 file uses different column names compared to 2022/2023.
# This mapping converts 2021 names to the common 2022/2023 convention,
# ensuring consistent column references across all survey years.
COLUMN_RENAME_2021 = {
    'incanon': 'anon_income',      # anonymised weekly household income
    'SexHRP': 'sexhrp',            # sex of household reference person
    'Gorx': 'gorx',                # government office region code
    'EqIncDMp': 'eqincdmp',        # equivalised income (modified OECD)
    'EqIncDOp': 'eqincdop',        # equivalised income (original OECD)
}


def load_single_year(year: int) -> pd.DataFrame:
    """
    Load a single year of LCFS household data from a tab-separated file.

    Steps:
    1. Read the raw .tab file from the data directory
    2. Normalise all column names to lowercase for consistency
    3. Apply year-specific column renames (2021 has different naming)
    4. Add a 'survey_year' column to track data provenance
    """
    filepath = os.path.join(DATA_DIR, FILES[year])
    df = pd.read_csv(filepath, sep='\t', low_memory=False)

    # Normalise column names to lowercase for consistent access across years
    df.columns = df.columns.str.lower()

    # Apply 2021-specific renames (must happen after lowercasing to match keys)
    if year == 2021:
        rename_map = {k.lower(): v for k, v in COLUMN_RENAME_2021.items()}
        df = df.rename(columns=rename_map)

    # Tag each row with its survey year for later filtering or analysis
    df['survey_year'] = year
    return df


def load_lcfs_data(years=None) -> pd.DataFrame:
    """
    Load and merge multiple years of LCFS data.

    Since each survey year may have slightly different columns (variables added
    or removed between waves), we only keep the intersection of columns that
    exist in ALL years — ensuring a consistent feature space.

    Parameters
    ----------
    years : list of int, optional
        Which years to load. Defaults to all available (2021, 2022, 2023).

    Returns
    -------
    pd.DataFrame
        Merged dataframe with a 'survey_year' column.
    """
    if years is None:
        years = list(FILES.keys())

    # Load each year independently
    frames = []
    for year in years:
        df = load_single_year(year)
        frames.append(df)
        print(f"Loaded {year}: {df.shape[0]} rows, {df.shape[1]} columns")

    # Find the intersection of column names across all years
    # This prevents errors when concatenating frames with mismatched schemas
    common_cols = set(frames[0].columns)
    for df in frames[1:]:
        common_cols &= set(df.columns)
    common_cols = sorted(common_cols)

    # Concatenate using only common columns, resetting the index
    merged = pd.concat([df[common_cols] for df in frames], ignore_index=True)
    print(f"\nMerged: {merged.shape[0]} rows, {merged.shape[1]} common columns")
    return merged


# ── Equivalised income computation ──────────────────────────────────────────

def compute_oecd_scale(df: pd.DataFrame) -> pd.Series:
    """
    Compute the OECD modified equivalence scale from household composition.

    The scale adjusts for household size so that income can be compared fairly
    across households of different compositions:
        Scale = 1 + 0.5 * (additional adults) + 0.3 * (children under 14)

    A single-adult household has scale = 1.0 (the reference point).
    A couple with 2 children has scale = 1 + 0.5 + 0.3*2 = 2.1.

    Uses:
    - a055: number of adults in household
    - a049: total household size (adults + children)
    - Children = a049 - a055 (derived)

    Returns
    -------
    pd.Series
        OECD modified equivalence scale for each household.
    """
    # Coerce to numeric in case of mixed types; default to 1 (single person)
    n_adults = pd.to_numeric(df['a055'], errors='coerce').fillna(1)
    n_total = pd.to_numeric(df['a049'], errors='coerce').fillna(1)

    # Derive number of children (cannot be negative)
    n_children = (n_total - n_adults).clip(lower=0)

    # Apply the OECD modified formula:
    # 1.0 for the first adult, 0.5 per additional adult, 0.3 per child
    scale = 1 + 0.5 * (n_adults - 1) + 0.3 * n_children
    return scale


def compute_equivalised_income(df: pd.DataFrame) -> pd.Series:
    """
    Calculate equivalised income = anon_income / OECD modified equivalence scale.

    This adjusts raw household income for household size, producing a measure
    of purchasing power that is comparable across households of different sizes.
    For example, a single person earning £500/week has equivalised income of £500,
    while a couple with 2 children earning £500/week has equivalised income of ~£238.
    """
    scale = compute_oecd_scale(df)
    # Divide raw anonymised income by the equivalence scale
    equiv_income = df['anon_income'] / scale
    return equiv_income


# ── Feature group definitions ───────────────────────────────────────────────
# These lists define which columns in the LCFS data are used as features
# for the income prediction task. They are separated into expenditure
# (continuous) and demographic (mixed type) groups.

# Key expenditure variables (weekly household expenditure categories, in £)
# These represent different COICOP-style spending categories
EXPENDITURE_FEATURES = [
    'p600',    # total expenditure (sum of all categories)
    'p601',    # food & non-alcoholic drinks (alternate variable)
    'p530',    # housing (net) — rent/mortgage after benefits
    'p531',    # fuel, light & power (gas, electricity, etc.)
    'p532',    # food & non-alcoholic drinks
    'p533',    # alcoholic drinks
    'p534',    # tobacco
    'p535p',   # clothing & footwear
    'p536p',   # household goods (furniture, appliances, etc.)
    'p537',    # household services (insurance, repairs, etc.)
    'p538',    # personal goods & services (toiletries, hairdressing)
    'p539',    # motoring (vehicle costs, fuel, etc.)
    'p540',    # fares & other travel (public transport, flights)
    'p541',    # leisure goods (books, toys, sports equipment)
    'p542',    # leisure services (holidays, eating out, subscriptions)
    'p543',    # miscellaneous (education, charitable giving)
    'p544',    # other items (not elsewhere classified)
]

# Demographic / household composition features
# These describe the household structure, geography, and socio-economic position
DEMOGRAPHIC_FEATURES = [
    'a049',    # household size (total number of people)
    'a055',    # number of adults in household
    'a099',    # household composition type (e.g. couple, lone parent)
    'sexhrp',  # sex of household reference person (1=male, 2=female)
    'gorx',    # government office region (1-12, e.g. 7=London)
    'a121',    # tenure type (1=owned outright, 2=mortgage, 3=rented, etc.)
    'a116',    # age of HRP — grouped into bands
    'a103',    # economic position of HRP (employed, retired, etc.)
]

# Variables to EXCLUDE from features (income-derived — would cause target leakage)
# Including any of these would give the model direct or indirect access to the
# target variable (equivalised income), making predictions artificially accurate
# but useless for real-world deployment where income is unknown.
LEAKAGE_VARIABLES = [
    'anon_income',          # raw income (used to derive the target)
    'equivalised_income',   # target variable itself (anon_income / oecd_scale)
    'oecd_scale',           # equivalence scale (derived from household composition)
    'p630p',                # equivalised expenditure (income-adjusted)
    'p630cp',               # equivalised expenditure (children)
    'p630tp',               # equivalised expenditure (total)
    'eqincdmp',             # equivalised income (modified OECD) — pre-computed by ONS
    'eqincdop',             # equivalised income (OECD original) — pre-computed by ONS
    'weighta',              # survey weight (not a predictor, affects sampling)
]


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Return the list of feature columns that exist in the dataframe,
    excluding any leakage variables.

    This function serves as a safety gate: it only returns columns that are
    both (a) in our pre-defined feature lists and (b) NOT in the leakage
    exclusion list. This prevents accidental inclusion of income-derived
    variables that would invalidate the predictive task.
    """
    # Combine expenditure and demographic feature lists
    all_features = EXPENDITURE_FEATURES + DEMOGRAPHIC_FEATURES

    # Filter to only columns that actually exist in this dataframe
    # (some variables may not be present in all survey years)
    available = [c for c in all_features if c in df.columns]

    # Build a set of leakage variable names (lowercase) for fast lookup
    leakage = set(c.lower() for c in LEAKAGE_VARIABLES)

    # Exclude any feature that appears in the leakage list
    safe = [c for c in available if c.lower() not in leakage]
    return safe
