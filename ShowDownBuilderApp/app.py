import random
from typing import Dict, List, Tuple
import pandas as pd
import streamlit as st

# ================================================================
#                GLOBALS / DEFAULT CONFIG VALUES
# ================================================================

DEFAULT_NUM_LINEUPS = 20
DEFAULT_SALARY_CAP = 50000      # DK showdown cap
DEFAULT_MIN_SALARY = 48500
DEFAULT_RANDOM_SEED = 42

BUILD_TYPE_COUNTS = {
    "5-1": (5, 1),
    "4-2": (4, 2),
    "3-3": (3, 3),
    "2-4": (2, 4),
    "1-5": (1, 5),
}

CAPTAIN_CONFIG: Dict[str, Dict] = {}

SLOT_ORDER = ["CPT", "FLEX1", "FLEX2", "FLEX3", "FLEX4", "FLEX5"]


# ================================================================
#                        DATA LOADING
# ================================================================

def load_showdown_pool(source) -> pd.DataFrame:
    df = pd.read_csv(source)
    df.columns = [c.strip() for c in df.columns]

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    df["Salary"] = df["Salary"].astype(int)
    return df


def split_cpt_flex(df: pd.DataFrame):
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
    items = list(weights.items())
    total = sum(w for _, w in items if w > 0)

    if total == 0:
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
    bw = CAPTAIN_CONFIG[captain]["build_weights"]
    return weighted_random_choice(bw)


def get_build_rules(captain: str, build_type: str) -> Dict[str, List[str]]:
    rules = CAPTAIN_CONFIG[captain]["build_rules"].get(build_type, {})
    return {
        "include": rules.get("include", []),
        "exclude": rules.get("exclude", []),
        "locks": rules.get("locks", []),
    }


def build_captain_plan(n: int, captain_config: Dict[str, Dict]) -> List[str]:
    weights = {k: v["exposure"] for k, v in captain_config.items()}
    return [weighted_random_choice(weights) for _ in range(n)]


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
    locks = rules["locks"]

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

    # ---- LOCKS ----
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

        entries.append({"Slot": f"FLEX{len(entries)}", "Player": row})
        used_ids.add(row["ID"])
        used_names.add(row["Name"])
        team_counts[t] += 1

    # ---- FILL REMAINING ----
    while len(entries) < 6:
        need_c = need_primary - team_counts[captain_team]
        need_o = need_opp - team_counts[opp_team]

        choices = []
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
            entries.append({"Slot": f"FLEX{len(entries)}", "Player": row})
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
#                         STREAMLIT APP
# ================================================================

def run_app():
    global CAPTAIN_CONFIG

    st.title("🏈 DraftKings Showdown Lineup Builder")

    uploaded = st.file_uploader("Upload DKSalaries.csv", type=["csv"])
    if not uploaded:
        st.info("Please upload a `DKSalaries.csv` file.")
        return

    st.sidebar.header("Global Settings")
    num_lineups = st.sidebar.number_input("Number of Lineups", 1, 150, DEFAULT_NUM_LINEUPS)
    salary_cap = st.sidebar.number_input("Salary Cap", 0, 50000, DEFAULT_SALARY_CAP, 500)
    min_salary = st.sidebar.number_input("Minimum Salary", 0, salary_cap, DEFAULT_MIN_SALARY, 500)
    random_seed = st.sidebar.number_input("Random Seed (-1 for None)", value=DEFAULT_RANDOM_SEED)

    df = load_showdown_pool(uploaded)
    df_cpt, df_flex = split_cpt_flex(df)

    try:
        teams = determine_teams(df)
    except Exception as e:
        st.error(str(e))
        return

    st.write(f"Detected teams: **{teams[0]}** vs **{teams[1]}**")
    st.write(f"CPT pool size: **{len(df_cpt)}**, FLEX pool size: **{len(df_flex)}**")

    captain_names = sorted(df_cpt["Name"].unique().tolist())
    flex_names_all = sorted(df_flex["Name"].unique().tolist())

    st.subheader("Captain Configurations")

    captain_config = {}
    exposure_raw = {}

    # --------- MAIN CAPTAIN CONFIG UI ---------
    for cap in captain_names:
        with st.expander(f"Captain: {cap}", expanded=False):

            exp_val = st.slider(
                f"{cap} Exposure (%)",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=1.0,
                key=f"exp_{cap}",
            )
            exposure_raw[cap] = exp_val

            # FLEX players EXCLUDING this captain
            available_flex = [p for p in flex_names_all if p != cap]

            tabs = st.tabs(list(BUILD_TYPE_COUNTS.keys()))
            build_weights = {}
            build_rules = {}

            for tab, build_type in zip(tabs, BUILD_TYPE_COUNTS.keys()):
                with tab:
                    bw = st.slider(
                        f"{build_type} Weight (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=0.0,
                        step=1.0,
                        key=f"bw_{cap}_{build_type}",
                    )
                    build_weights[build_type] = bw

                    include_sel = st.multiselect(
                        f"Include - {build_type}",
                        options=available_flex,
                        default=[],
                        key=f"inc_{cap}_{build_type}"
                    )

                    exclude_sel = st.multiselect(
                        f"Exclude - {build_type}",
                        options=available_flex,
                        default=[],
                        key=f"exc_{cap}_{build_type}"
                    )

                    lock_sel = st.multiselect(
                        f"Locks - {build_type}",
                        options=available_flex,
                        default=[],
                        key=f"lock_{cap}_{build_type}"
                    )

                    build_rules[build_type] = {
                        "include": include_sel,
                        "exclude": exclude_sel,
                        "locks": lock_sel,
                    }

            captain_config[cap] = {
                "exposure": exp_val,
                "build_weights": build_weights,
                "build_rules": build_rules,
            }

    CAPTAIN_CONFIG = captain_config

    # --------- BUILD LINEUPS BUTTON ---------
    if st.button("🚀 Build Lineups"):
        if random_seed >= 0:
            random.seed(int(random_seed))

        if all(v == 0 for v in exposure_raw.values()):
            st.error("All captain exposures are zero. Set at least one exposure > 0.")
            return

        # Only require build weights for captains with exposure > 0
        for cap, cfg in CAPTAIN_CONFIG.items():
            if cfg["exposure"] > 0:
                if all(w == 0 for w in cfg["build_weights"].values()):
                    st.error(f"Captain {cap} has exposure > 0 but all build weights are 0.")
                    return

        captain_plan = build_captain_plan(num_lineups, CAPTAIN_CONFIG)

        lineups = []
        seen = set()
        attempts = 0
        max_attempts = num_lineups * 500

        for cap in captain_plan:
            built = False
            while not built and attempts < max_attempts:
                attempts += 1
                result = build_lineup_for_captain(
                    cap, df_cpt, df_flex, teams, min_salary, salary_cap
                )
                if not result:
                    continue

                lineup, build_type = result
                key = lineup_to_key(lineup)
                if key in seen:
                    continue

                seen.add(key)
                lineups.append((lineup, build_type))
                built = True

        if not lineups:
            st.error("No valid lineups could be built. Loosen constraints.")
            return

        st.success(f"Built {len(lineups)} lineups!")

        df_out = lineups_to_df(lineups)
        st.dataframe(df_out)

        st.download_button(
            "Download CSV",
            df_out.to_csv(index=False).encode("utf-8"),
            "showdown_lineups.csv",
            "text/csv"
        )


if __name__ == "__main__":
    run_app()
