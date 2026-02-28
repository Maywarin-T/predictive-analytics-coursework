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
    exist in ALL years, ensuring a consistent feature space.

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

    A single adult household has scale = 1.0 (the reference point).
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
# Features were selected using Pearson correlation with equivalised income
# across all 3 survey years, after removing leakage variables and
# de-duplicating near identical columns (r > 0.99).
# All features below exist in the common column set across 2021-2023.

# Expenditure features selected by correlation with equivalised income
# These are weekly household spending amounts (£) from COICOP categories
EXPENDITURE_FEATURES = [
    # Aggregate expenditure totals (r > 0.30)
    'p600',    # total expenditure (r=0.34)
    'p550tp',  # total expenditure including housing costs (r=0.38)
    'p531',    # fuel, light and power (r=0.43)
    'p538',    # personal goods and services (r=0.30)
    'p128t',   # total regular outgoings: mortgage, council tax, insurance (r=0.40)
    'p071h',   # mortgage and rent payments (r=0.42)
    'p153t',   # total other regular outgoing payments (r=0.30)
    # Sub-category expenditure (r = 0.15 to 0.30)
    'p620tp',  # clothing and footwear (r=0.27)
    'p601',    # food and non-alcoholic drinks (r=0.25)
    'p073hp',  # mortgage interest payments (r=0.26)
    'p515tp',  # household goods and services (r=0.24)
    'p611',    # alcoholic drinks and tobacco (r=0.20)
    'p607',    # non-alcoholic drinks (r=0.22)
    'p537',    # household services: insurance, repairs (r=0.18)
    'p612',    # clothing sub-category (r=0.20)
    'p516tp',  # furniture and furnishings (r=0.20)
    'p548',    # miscellaneous goods and services (r=0.19)
    'p545',    # recreation and culture (r=0.18)
    'p609',    # food sub-category (r=0.15)
    'p220p',   # council tax or rates (r=-0.18)
    # COICOP detailed sub-categories (r > 0.15)
    'c11711',  # catering: restaurant and cafe meals (r=0.21)
    'cb1111',  # food: bread and cereals (r=0.18)
    'c11731',  # catering: canteen meals (r=0.18)
    'ctrbpcnt', # travel: public transport season tickets (r=-0.19)
    'c11761',  # catering: other take-away food (r=0.16)
    'c11711l', # catering: restaurant meals (alt code) (r=0.17)
    'cb1311',  # food: fish (r=0.18)
]

# Demographic and household composition features
# These describe household structure and socio-economic characteristics
DEMOGRAPHIC_FEATURES = [
    # High correlation demographics (|r| > 0.30)
    'a055',    # number of adults in household (r=-0.54)
    'a099',    # household composition type: couple, lone parent, etc. (r=-0.52)
    'a054',    # number of earners in household (r=0.41)
    'a056',    # number of persons with income (r=0.40)
    'a069p',   # household composition (3 categories) (r=0.31)
    'a091',    # socio-economic classification of HRP (r=-0.36)
    'a094',    # government office region (r=-0.34)
    'a093',    # socio-economic group of HRP (r=-0.32)
    # Moderate correlation demographics (|r| = 0.15 to 0.30)
    'a160',    # number of rooms (r=0.29)
    'g018',    # number of adults (alternative code) (r=0.29)
    'a124',    # number of adults in employment (r=0.28)
    'a143p',   # number of cars or vans owned (r=0.28)
    'a149',    # central heating type (r=0.27)
    'a062',    # age of HRP, banded 1-30 (r=0.28)
    'a184',    # dependent children indicator (r=0.24)
    'a049',    # household size, total persons (r=0.24)
    'a044',    # number of dependent children under 16 (r=0.19)
    'a043',    # number of children in household (r=0.21)
    'a024',    # number of persons aged 65+ (r=0.17)
    'a023',    # number of full time workers (r=0.18)
    'a1646p',  # internet access indicator (r=0.17)
    'a065p',   # employment status of HRP (r=-0.18)
    'a1661',   # council tax band (r=-0.17)
    # Standard demographic controls (low correlation but important context)
    'sexhrp',  # sex of household reference person (r=-0.11)
    'a121',    # tenure type: own, mortgage, rent (r=0.07)
]

# ── Leakage variable exclusion ──────────────────────────────────────────────
# Variables to EXCLUDE from features because they are derived from income
# or encode income information directly or indirectly.
#
# Including any of these would give the model access to the target variable,
# making predictions artificially accurate but useless for real world
# deployment where income is unknown.
#
# Categories of leakage:
# 1. Income variables (anon_income and aggregates in p300-p399)
# 2. Tax and deductions (p400-p499, directly calculated from income)
# 3. Pre-computed equivalised income (eqincdmp, eqincdop)
# 4. Income quantile groupings (a060)
# 5. Means-tested benefits (all b-codes, amounts determined by income)
# 6. Payroll deductions and welfare linked variables
# 7. Survey weights and OECD scale (not predictors)

LEAKAGE_VARIABLES = [
    # Direct income variables
    'anon_income',          # raw anonymised household income (target source)
    'equivalised_income',   # target variable itself (anon_income / oecd_scale)
    # Income aggregates (p300-p399 range is the income section of LCFS)
    'p344p',                # gross household income (r=0.999 with anon_income)
    'p352p',                # income sub-total (r=0.986)
    'p389p',                # income sub-total (r=0.983)
    'p431p',                # income tax paid (r=0.954)
    'p392p',                # income aggregate (r=0.853)
    'p356p',                # income aggregate (r=0.844)
    'p300p',                # income aggregate (r=0.843)
    'p390p',                # income sub-total (r=0.801)
    'p388p',                # income sub-total (r=0.793)
    'p348',                 # income component (r=-0.302)
    # Tax and deductions (calculated from income)
    'p493p',                # national insurance and tax total (r=0.537)
    'p425',                 # tax band code (r=-0.510)
    # Pre-computed equivalised income from ONS
    'eqincdmp',             # equivalised income, modified OECD (r=0.828)
    'eqincdop',             # equivalised income, original OECD (r=0.844)
    # Income quantile grouping
    'a060',                 # income quantile band (Spearman r=0.97 with income)
    # Equivalised expenditure (income adjusted)
    'p630p',                # equivalised expenditure
    'p630cp',               # equivalised expenditure (children)
    'p630tp',               # equivalised expenditure (total)
    # Payroll and welfare linked
    'p281p',                # pension contributions deducted from pay (r=0.387)
    'p206p',                # rent rebate or council tax reduction (income tested)
    # OECD equivalence scale (used to compute target)
    'oecd_scale',           # derived scale column
    'oecd',                 # raw OECD scale from dataset
    # Survey weights (not predictors)
    'weighta',              # survey weight
    'weightq',              # quarterly weight
    'non_response_weight',  # non-response adjustment weight
    # Survey metadata
    'survey_year',          # year indicator (not a household characteristic)
]

# All b-codes (means-tested benefits) are excluded as a category.
# UK benefits like Universal Credit, Housing Benefit, and Income Support
# are income-tested: their amounts directly encode the household's income
# level, constituting indirect target leakage.
LEAKAGE_PREFIXES = ['b']


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Return the list of feature columns that exist in the dataframe,
    excluding any leakage variables.

    This function serves as a safety gate: it only returns columns that are
    both (a) in our pre-defined feature lists and (b) NOT in the leakage
    exclusion list. This prevents accidental inclusion of income derived
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
    # Also exclude any feature whose prefix matches a leakage prefix (e.g. b-codes)
    safe = []
    for c in available:
        if c.lower() in leakage:
            continue
        if any(c.lower().startswith(prefix) for prefix in LEAKAGE_PREFIXES):
            continue
        safe.append(c)
    return safe
