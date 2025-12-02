import random
import pandas as pd
import streamlit as st

# ================================================================
#                     GLOBAL CONFIG (USER SETTINGS)
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

CAPTAIN_CONFIG = {
    "": {
        "exposure": 0.0,
        "build_weights": {
            "4-2": 0.0,
            "3-3": 0.0,
            "5-1": 0.0,
            "2-4": 0.0,
            "1-5": 0.0,
        },
        "build_rules": {
            "4-2": {"include": [], "exclude": [], "locks": []},
            "3-3": {"include": [], "exclude": [], "locks": []},
            "5-1": {"include": [], "exclude": [], "locks": []},
            "2-4": {"include": [], "exclude": [], "locks": []},
            "1-5": {"include": [], "exclude": [], "locks": []},
        },
    }
}

DEFAULT_BUILD_WEIGHTS = {
    "4-2": 0.4,
    "3-3": 0.4,
    "5-1": 0.1,
    "2-4": 0.05,
    "1-5": 0.05,
}

ALLOWED_CAPTAIN_POSITIONS = {"QB", "RB", "WR", "TE", "K", "DST"}
EXCLUDED_PLAYERS = []
EXCLUDED_TEAMS = []

SLOT_ORDER = ["CPT", "FLEX1", "FLEX2", "FLEX3", "FLEX4", "FLEX5"]


# ================================================================
#                        DATA LOADING
# ================================================================

def load_showdown_pool(source):
    df = pd.read_csv(source)
    df.columns = [c.strip() for c in df.columns]

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    df["Salary"] = df["Salary"].astype(int)
    return df


def apply_exclusions(df):
    if EXCLUDED_PLAYERS:
        df = df[~df["Name"].isin(EXCLUDED_PLAYERS)]
    if EXCLUDED_TEAMS:
        df = df[~df["TeamAbbrev"].isin(EXCLUDED_TEAMS)]
    return df.reset_index(drop=True)


def split_cpt_flex(df):
    cpt = df[df["Roster Position"] == "CPT"].reset_index(drop=True)
    flex = df[df["Roster Position"] == "FLEX"].reset_index(drop=True)
    return cpt, flex


# ================================================================
#                      HELPER FUNCTIONS
# ================================================================

def salary_of_lineup(entries):
    return sum(e["Player"]["Salary"] for e in entries)


def lineup_to_key(entries):
    return tuple(sorted(e["Player"]["ID"] for e in entries))


def weighted_random_choice(weights):
    items = list(weights.items())
    total = sum(w for _, w in items if w > 0)

    if total == 0:
        return random.choice([k for k, _ in items])

    r = random.random() * total
    acc = 0
    for k, w in items:
        if w > 0:
            acc += w
            if acc >= r:
                return k
    return items[-1][0]


def determine_teams(df):
    teams = sorted(df["TeamAbbrev"].unique().tolist())
    if len(teams) != 2:
        raise ValueError(f"File must include exactly 2 teams. Found: {teams}")
    return teams


def choose_build_type(captain):
    bw = CAPTAIN_CONFIG[captain].get("build_weights", DEFAULT_BUILD_WEIGHTS)
    bw = {k: v for k, v in bw.items() if v > 0}
    if not bw:
        bw = DEFAULT_BUILD_WEIGHTS
    return weighted_random_choice(bw)


def get_build_rules(captain, build_type):
    rules = CAPTAIN_CONFIG[captain]["build_rules"][build_type]
    return {
        "include": rules.get("include", []),
        "exclude": rules.get("exclude", []),
        "locks": rules.get("locks", []),
    }


def build_lineup_for_captain(captain, df_cpt, df_flex, teams):
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

    def pool(team):
        p = df_flex[df_flex["TeamAbbrev"] == team]
        p = p[~p["ID"].isin(used_ids)]
        p = p[~p["Name"].isin(used_names)]
        if include:
            p = p[p["Name"].isin(include)]
        if exclude:
            p = p[~p["Name"].isin(exclude)]
        return p.reset_index(drop=True)

    # LOCKS
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

    # FILL REST
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
    if not (DEFAULT_MIN_SALARY <= total_salary <= DEFAULT_SALARY_CAP):
        return None

    return entries, build_type


# ================================================================
#                        MAIN STREAMLIT APP
# ================================================================

def run_app():
    st.title("🏈 DraftKings Showdown Lineup Builder")
    st.write("Upload a DraftKings **DKSalaries.csv** file to generate lineups.")

    uploaded = st.file_uploader("Upload DKSalaries.csv", type=["csv"])
    if not uploaded:
        st.info("Please upload a file to continue.")
        return

    # Sidebar settings
    st.sidebar.header("Settings")
    num_lineups = st.sidebar.number_input(
        "Number of Lineups",
        min_value=1,
        max_value=150,
        value=DEFAULT_NUM_LINEUPS,
        step=1,
    )

    min_salary = st.sidebar.number_input(
        "Minimum Salary",
        min_value=0,
        max_value=50000,
        value=DEFAULT_MIN_SALARY,
        step=500,
    )

    random_seed = st.sidebar.number_input(
        "Random Seed (-1 for None)", value=DEFAULT_RANDOM_SEED
    )

    if st.button("Build Lineups"):
        if random_seed >= 0:
            random.seed(int(random_seed))

        try:
            df = load_showdown_pool(uploaded)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            return

        df = apply_exclusions(df)

        df_cpt, df_flex = split_cpt_flex(df)

        try:
            teams = determine_teams(df)
        except ValueError as e:
            st.error(str(e))
            return

        captain_plan = [""] * num_lineups  # No exposure config yet

        lineups = []
        seen = set()
        attempts = 0
        max_attempts = num_lineups * 500

        for captain in captain_plan:
            built = False
            while not built and attempts < max_attempts:
                attempts += 1
                result = build_lineup_for_captain(captain, df_cpt, df_flex, teams)
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
                st.warning(f"Could not build lineup for captain {captain}")

        st.success(f"Built {len(lineups)} lineups.")

        if not lineups:
            st.error("No valid lineups generated.")
            return

        df_out = lineups_to_df(lineups)
        st.dataframe(df_out)

        st.download_button(
            "Download CSV",
            df_out.to_csv(index=False).encode("utf-8"),
            "showdown_lineups.csv",
            "text/csv",
        )


def lineups_to_df(lineups):
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


if __name__ == "__main__":
    run_app()
