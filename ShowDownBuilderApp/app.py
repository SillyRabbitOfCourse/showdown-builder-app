import random
from typing import Dict, List, Tuple
import pandas as pd
import streamlit as st

# ================================================================
#                GLOBALS / DEFAULT CONFIG VALUES
# ================================================================

DEFAULT_NUM_LINEUPS = 20
DEFAULT_SALARY_CAP = 50000
DEFAULT_MIN_SALARY = 48500
DEFAULT_RANDOM_SEED = 42

BUILD_TYPE_COUNTS = {
    "5-1": (5, 1),
    "4-2": (4, 2),
    "3-3": (3, 3),
    "2-4": (2, 4),
    "1-5": (1, 5),
}

DEFAULT_BUILD_WEIGHTS = {
    "4-2": 0.4,
    "3-3": 0.4,
    "5-1": 0.1,
    "2-4": 0.05,
    "1-5": 0.05,
}

# Will be built dynamically from UI
CAPTAIN_CONFIG: Dict[str, Dict] = {}

EXCLUDED_PLAYERS: List[str] = []  # Global player-level excludes (optional)
EXCLUDED_TEAMS: List[str] = []    # You said no team-level exclusion in UI

SLOT_ORDER = ["CPT", "FLEX1", "FLEX2", "FLEX3", "FLEX4", "FLEX5"]


# ================================================================
#                        DATA LOADING
# ================================================================

def load_showdown_pool(source) -> pd.DataFrame:
    """
    Load DK Showdown CSV. Assumes standard DKSalaries format.
    Separate CPT and FLEX rows have their own salaries (as in DK).
    """
    df = pd.read_csv(source)

    # Strip whitespace
    df.columns = [c.strip() for c in df.columns]
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    df["Salary"] = df["Salary"].astype(int)
    return df


def apply_exclusions(df: pd.DataFrame) -> pd.DataFrame:
    if EXCLUDED_PLAYERS:
        df = df[~df["Name"].isin(EXCLUDED_PLAYERS)]
    if EXCLUDED_TEAMS:
        df = df[~df["TeamAbbrev"].isin(EXCLUDED_TEAMS)]
    return df.reset_index(drop=True)


def split_cpt_flex(df: pd.DataFrame):
    """
    Splits into CPT and FLEX pools based on 'Roster Position'.
    NOTE: This preserves the separate CPT and FLEX salaries from the file.
    """
    cpt = df[df["Roster Position"] == "CPT"].reset_index(drop=True)
    flex = df[df["Roster Position"] == "FLEX"].reset_index(drop=True)
    return cpt, flex


# ================================================================
#                      HELPER FUNCTIONS
# ================================================================

def salary_of_lineup(entries: List[Dict]) -> int:
    return sum(e["Player"]["Salary"] for e in entries)


def lineup_to_key(entries: List[Dict]) -> Tuple:
    return tuple(sorted(e["Player"]["ID"] for e in entries))


def weighted_random_choice(weights: Dict[str, float]) -> str:
    """
    Generic weighted random choice.
    weights: mapping from item -> weight (non-negative).
    """
    items = list(weights.items())
    total = sum(w for _, w in items if w > 0)

    if total == 0:
        # All zero weights, pick uniformly
        return random.choice([k for k, _ in items])

    r = random.random() * total
    acc = 0.0
    for k, w in items:
        if w > 0:
            acc += w
            if acc >= r:
                return k
    return items[-1][0]


def determine_teams(df: pd.DataFrame) -> List[str]:
    teams = sorted(df["TeamAbbrev"].unique().tolist())
    if len(teams) != 2:
        raise ValueError(f"File must include exactly 2 teams. Found: {teams}")
    return teams


def choose_build_type(captain: str) -> str:
    """
    Uses per-captain build_weights from CAPTAIN_CONFIG.
    Falls back to DEFAULT_BUILD_WEIGHTS if captain has no non-zero weights.
    """
    bw = CAPTAIN_CONFIG[captain].get("build_weights", DEFAULT_BUILD_WEIGHTS)
    bw = {k: v for k, v in bw.items() if v > 0}
    if not bw:
        bw = DEFAULT_BUILD_WEIGHTS
    return weighted_random_choice(bw)


def get_build_rules(captain: str, build_type: str) -> Dict[str, List[str]]:
    rules = CAPTAIN_CONFIG[captain]["build_rules"].get(build_type, {})
    return {
        "include": rules.get("include", []),
        "exclude": rules.get("exclude", []),
        "locks": rules.get("locks", []),
    }


def build_captain_plan(n: int, captain_config: Dict[str, Dict]) -> List[str]:
    """
    Build a list of length n of captain names based on exposure.
    Exposures should already be normalized to sum to 1.0,
    but we still handle rounding and leftover.
    """
    cfg = {k: v for k, v in captain_config.items() if v["exposure"] > 0}
    if not cfg:
        # No exposures set -> equal shares among all captains in config
        cfg = captain_config

    plan: List[str] = []
    # First allocate deterministic counts
    for name, data in cfg.items():
        cnt = int(data["exposure"] * n)
        plan.extend([name] * cnt)

    # Fill remaining slots stochastically according to exposure weights
    leftover = n - len(plan)
    weights = {k: v["exposure"] for k, v in cfg.items()}
    for _ in range(leftover):
        plan.append(weighted_random_choice(weights))

    random.shuffle(plan)
    return plan[:n]


# ================================================================
#                    LINEUP BUILDING LOGIC
# ================================================================

def build_lineup_for_captain(
    captain: str,
    df_cpt: pd.DataFrame,
    df_flex: pd.DataFrame,
    teams: List[str],
    min_salary: int,
    salary_cap: int,
):
    """
    Builds a single lineup for a given captain name.
    Uses CPT row for captain (with CPT salary) and FLEX rows (with FLEX salaries)
    for the remaining 5 slots.
    """
    cpt = df_cpt[df_cpt["Name"] == captain]
    if cpt.empty:
        return None

    cpt_row = cpt.iloc[0]
    captain_team = cpt_row["TeamAbbrev"]
    opp_team = teams[1] if teams[0] == captain_team else teams[0]

    build_type = choose_build_type(captain)
    need_primary, need_opp = BUILD_TYPE_COUNTS[build_type]

    rules = get_build_rules(captain, build_type)
    include = set(rules["include"]) if rules["include"] else None
    exclude = set(rules["exclude"])
    locks = rules["locks"] or []

    # Start with captain
    entries = [{"Slot": "CPT", "Player": cpt_row}]
    used_ids = {cpt_row["ID"]}
    used_names = {cpt_row["Name"]}

    team_counts = {captain_team: 1, opp_team: 0}

    def pool(team: str) -> pd.DataFrame:
        p = df_flex[df_flex["TeamAbbrev"] == team]
        p = p[~p["ID"].isin(used_ids)]
        p = p[~p["Name"].isin(used_names)]
        if include:
            p = p[p["Name"].isin(include)]
        if exclude:
            p = p[~p["Name"].isin(exclude)]
        return p.reset_index(drop=True)

    # ---------------- LOCKS ----------------
    for lock in locks:
        cand = df_flex[
            (df_flex["Name"] == lock)
            & (~df_flex["ID"].isin(used_ids))
            & (~df_flex["Name"].isin(used_names))
        ]
        if cand.empty:
            return None

        row = cand.sample(1).iloc[0]
        t = row["TeamAbbrev"]

        max_allowed = need_primary if t == captain_team else need_opp
        if team_counts[t] + 1 > max_allowed:
            return None

        slot = f"FLEX{len(entries)}"
        entries.append({"Slot": slot, "Player": row})
        used_ids.add(row["ID"])
        used_names.add(row["Name"])
        team_counts[t] += 1

    # ---------------- FILL REMAINING ----------------
    while len(entries) < 6:
        need_c = need_primary - team_counts[captain_team]
        need_o = need_opp - team_counts[opp_team]

        choices: List[str] = []
        if need_c > 0:
            choices += [captain_team] * need_c
        if need_o > 0:
            choices += [opp_team] * need_o
        if not choices:
            choices = [captain_team, opp_team]

        random.shuffle(choices)

        picked = False
        for t in choices:
            p = pool(t)
            if p.empty:
                continue
            row = p.sample(1).iloc[0]
            slot = f"FLEX{len(entries)}"
            entries.append({"Slot": slot, "Player": row})
            used_ids.add(row["ID"])
            used_names.add(row["Name"])
            team_counts[t] += 1
            picked = True
            break

        if not picked:
            return None

    total_salary = salary_of_lineup(entries)
    if not (min_salary <= total_salary <= salary_cap):
        return None

    return entries, build_type


# ================================================================
#                           OUTPUT
# ================================================================

def lineups_to_df(lineups: List[Tuple[List[Dict], str]]) -> pd.DataFrame:
    rows = []
    for i, (L, build_type) in enumerate(lineups, 1):
        row = {"LineupID": i, "Build_Type": build_type}
        total = 0
        for slot in SLOT_ORDER:
            p = next(e["Player"] for e in L if e["Slot"] == slot)
            row[slot] = p["Name + ID"]
            total += p["Salary"]
        row["Total_Salary"] = total
        rows.append(row)
    return pd.DataFrame(rows)


# ================================================================
#                STREAMLIT UI HELPERS (PARSING)
# ================================================================

def parse_multiline_names(text: str) -> List[str]:
    """
    Option B: one name per line.
    Empty lines are ignored.
    """
    if not text:
        return []
    return [line.strip() for line in text.splitlines() if line.strip()]


def normalize_dict(d: Dict[str, float]) -> Dict[str, float]:
    total = sum(v for v in d.values() if v > 0)
    if total <= 0:
        # If everything is zero, return equal weights
        n = len(d) or 1
        return {k: 1.0 / n for k in d.keys()}
    return {k: (v / total if v > 0 else 0.0) for k, v in d.items()}


# ================================================================
#                         STREAMLIT APP
# ================================================================

def run_app():
    global CAPTAIN_CONFIG

    st.title("🏈 DraftKings Showdown Lineup Builder")
    st.write("Upload a DraftKings **DKSalaries.csv** file to generate lineups with full captain/build rules.")

    uploaded = st.file_uploader("Upload DKSalaries.csv", type=["csv"])

    if not uploaded:
        st.info("Please upload a `DKSalaries.csv` file to continue.")
        return

    # Global settings in sidebar
    st.sidebar.header("Global Settings")
    num_lineups = st.sidebar.number_input(
        "Number of Lineups",
        min_value=1,
        max_value=150,
        value=DEFAULT_NUM_LINEUPS,
        step=1,
    )

    salary_cap = st.sidebar.number_input(
        "Salary Cap",
        min_value=1000,
        max_value=50000,
        value=DEFAULT_SALARY_CAP,
        step=500,
    )

    min_salary = st.sidebar.number_input(
        "Minimum Total Salary",
        min_value=0,
        max_value=salary_cap,
        value=DEFAULT_MIN_SALARY,
        step=500,
    )

    random_seed = st.sidebar.number_input(
        "Random Seed (-1 for None)",
        value=DEFAULT_RANDOM_SEED,
        step=1,
    )

    # Load pool
    try:
        df = load_showdown_pool(uploaded)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return

    df = apply_exclusions(df)
    df_cpt, df_flex = split_cpt_flex(df)

    if df_cpt.empty:
        st.error("No CPT rows found in file. Check that 'Roster Position' has 'CPT'.")
        return

    try:
        teams = determine_teams(df)
    except ValueError as e:
        st.error(str(e))
        return

    st.write(f"Detected teams: **{teams[0]}** vs **{teams[1]}**")
    st.write(f"CPT pool size: **{len(df_cpt)}**, FLEX pool size: **{len(df_flex)}**")

    captain_names = sorted(df_cpt["Name"].unique().tolist())

    st.markdown("---")
    st.subheader("Captain Exposures & Build Config")

    # ---------------- CAPTAIN EXPOSURES ----------------
    st.markdown("### Captain Exposures (will be normalized to 100%)")

    exposure_raw: Dict[str, float] = {}
    for cap in captain_names:
        key = f"exp_{cap}"
        # Start everyone at equal-ish exposure by default
        default_exp = round(100.0 / len(captain_names), 1) if captain_names else 0.0
        exposure_raw[cap] = st.slider(
            f"{cap} Exposure (%)",
            min_value=0.0,
            max_value=100.0,
            value=default_exp,
            step=1.0,
            key=key,
        )

    # Normalize exposures to sum to 1.0
    exposure_raw_frac = {k: v / 100.0 for k, v in exposure_raw.items()}
    exposure_norm = normalize_dict(exposure_raw_frac)

    # ---------------- CAPTAIN BUILD & RULES ----------------
    st.markdown("### Build Weights & Rules Per Captain")

    # Build up CAPTAIN_CONFIG from UI
    captain_config: Dict[str, Dict] = {}

    for cap in captain_names:
        with st.expander(f"Captain: {cap}", expanded=False):
            st.markdown("**Build Weights** (per captain, normalized per captain)")

            bw_raw: Dict[str, float] = {}
            for build_type, _ in BUILD_TYPE_COUNTS.items():
                default_pct = DEFAULT_BUILD_WEIGHTS.get(build_type, 0.0) * 100.0
                bw_raw[build_type] = st.slider(
                    f"{build_type} Weight (%) for {cap}",
                    min_value=0.0,
                    max_value=100.0,
                    value=default_pct,
                    step=1.0,
                    key=f"bw_{cap}_{build_type}",
                )

            # Normalize build weights per captain
            bw_frac = {k: v / 100.0 for k, v in bw_raw.items()}
            bw_norm = normalize_dict(bw_frac)  # you asked: Yes, they should sum to 1

            # Per-build rules: include, exclude, locks (one name per line)
            build_rules: Dict[str, Dict[str, List[str]]] = {}
            for build_type in BUILD_TYPE_COUNTS.keys():
                with st.expander(f"Rules for {cap} - {build_type}", expanded=False):
                    inc_txt = st.text_area(
                        f"Include (one player name per line) - {cap} {build_type}",
                        key=f"inc_{cap}_{build_type}",
                        height=80,
                    )
                    exc_txt = st.text_area(
                        f"Exclude (one player name per line) - {cap} {build_type}",
                        key=f"exc_{cap}_{build_type}",
                        height=80,
                    )
                    lock_txt = st.text_area(
                        f"Locks (one player name per line) - {cap} {build_type}",
                        key=f"lock_{cap}_{build_type}",
                        height=80,
                    )

                build_rules[build_type] = {
                    "include": parse_multiline_names(inc_txt),
                    "exclude": parse_multiline_names(exc_txt),
                    "locks": parse_multiline_names(lock_txt),
                }

            captain_config[cap] = {
                "exposure": exposure_norm.get(cap, 0.0),
                "build_weights": bw_norm,
                "build_rules": build_rules,
            }

    # Update global CAPTAIN_CONFIG used by lineup logic
    CAPTAIN_CONFIG = captain_config

    st.markdown("---")
    if st.button("🚀 Build Lineups"):
        if random_seed >= 0:
            random.seed(int(random_seed))

        if num_lineups <= 0:
            st.error("Number of lineups must be at least 1.")
            return

        # Build captain plan from exposures
        captain_plan = build_captain_plan(num_lineups, CAPTAIN_CONFIG)

        max_overall_attempts = num_lineups * 500
        lineups: List[Tuple[List[Dict], str]] = []
        seen = set()
        attempts = 0

        for captain in captain_plan:
            built = False
            while not built and attempts < max_overall_attempts:
                attempts += 1
                result = build_lineup_for_captain(
                    captain,
                    df_cpt,
                    df_flex,
                    teams,
                    min_salary=min_salary,
                    salary_cap=salary_cap,
                )
                if not result:
                    continue

                L, build_type = result
                key = lineup_to_key(L)
                if key in seen:
                    continue

                seen.add(key)
                lineups.append((L, build_type))
                built = True

            if not built:
                st.warning(f"Could not build lineup for captain: {captain}")

        st.write(f"Built **{len(lineups)}** lineups (requested {num_lineups}).")

        if not lineups:
            st.error("No valid lineups generated. Try relaxing constraints or adjusting rules.")
            return

        df_out = lineups_to_df(lineups)

        st.subheader("Generated Lineups")
        st.dataframe(df_out)

        csv_bytes = df_out.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Lineups CSV",
            data=csv_bytes,
            file_name="showdown_lineups_brain.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    run_app()
