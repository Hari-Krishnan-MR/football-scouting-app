# ============================================================
# STEP 8 - Streamlit scouting app
# "Beyond Goals and Assists": VAEP player values from one XGBoost model
# trained on World Cup 2022 and tested on Euro 2024 + Copa America 2024.
# Run locally:  streamlit run app.py
# ============================================================
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

DATA = Path(__file__).parent / "app_data"
TOURNAMENTS = {"euro2024": "Euro 2024 (unseen test)",
               "copa2024": "Copa América 2024 (unseen test)",
               "wc2022": "World Cup 2022 (training)"}
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#a3a29c"
TYPE_NAMES = {"pass": "Pass", "cross": "Cross", "carry": "Carry", "take_on": "Dribble",
              "shot": "Shot", "shot_freekick": "Free-kick shot", "shot_penalty": "Penalty",
              "freekick": "Free kick", "corner": "Corner", "throw_in": "Throw-in",
              "goal_kick": "Goal kick", "tackle": "Tackle", "interception": "Interception",
              "clearance": "Clearance", "ball_recovery": "Ball recovery", "bad_touch": "Bad touch",
              "dispossessed": "Dispossessed", "foul": "Foul", "own_goal": "Own goal"}

st.set_page_config(page_title="Attacker Scouting – VAEP", page_icon="⚽", layout="wide")


# ---------------- data and model (cached, loaded once) ----------------
@st.cache_data
def load_rankings(key):
    r = pd.read_csv(DATA / f"{key}_attacker_rankings.csv")
    r["id_player_id"] = r.id_player_id.astype(int)
    r["label"] = r.player + " (" + r.team + ")"
    return r.sort_values("rank_vaep")


@st.cache_data
def load_actions(key):
    a = pd.read_csv(DATA / f"{key}_attacker_actions.csv.gz")
    a["id_player_id"] = a.id_player_id.astype(int)
    a["action"] = a.id_type.map(TYPE_NAMES).fillna(a.id_type)
    a["minute"] = (a.id_time_seconds // 60).astype(int)
    return a


@st.cache_resource
def load_models():
    out = {}
    for lab in ["scores", "concedes"]:
        m = xgb.XGBClassifier()
        m.load_model(DATA / f"xgb_{lab}.json")
        out[lab] = m
    return out


models = load_models()
n_features = len(pd.read_csv(DATA / "feature_list.csv", header=None))

# ---------------- sidebar ----------------
st.sidebar.title("⚽ Attacker scouting")
key = st.sidebar.selectbox("Tournament", list(TOURNAMENTS), format_func=TOURNAMENTS.get)
rank = load_rankings(key)
acts = load_actions(key)

positions = sorted(rank.main_position.unique())
pos = st.sidebar.multiselect("Positions", positions, default=positions)
min_minutes = st.sidebar.slider("Minimum minutes played", 180, int(rank.minutes.max()), 180, step=30)
view = rank[rank.main_position.isin(pos) & (rank.minutes >= min_minutes)]

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Model:** XGBoost (one design, two outputs)  \n"
    f"P(score): {models['scores'].get_booster().num_boosted_rounds()} trees · "
    f"P(concede): {models['concedes'].get_booster().num_boosted_rounds()} trees  \n"
    f"{n_features} features · trained on World Cup 2022 only")
st.sidebar.caption("Data: StatsBomb Open Data")

st.title("Beyond Goals and Assists")
st.markdown("Finding **undervalued attacking players** for clubs with smaller budgets. "
            "Every on-ball action is valued by how much it changes the team's chance of "
            "scoring minus conceding in the next 10 actions (**VAEP**).")
if key == "wc2022":
    st.info("World Cup 2022 was the model's training data. Euro 2024 and Copa América 2024 are the honest out-of-sample views.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🔎 Hidden talent shortlist", "👤 Player profile", "📋 Full ranking", "📈 Model evidence", "ℹ️ About the data"])


# ---------------- pitch drawing ----------------
def pitch_figure():
    fig = go.Figure()
    line = dict(color="#bdbcb5", width=1.5)
    shapes = [dict(type="rect", x0=0, y0=0, x1=120, y1=80),
              dict(type="line", x0=60, y0=0, x1=60, y1=80),
              dict(type="circle", x0=50, y0=30, x1=70, y1=50),
              dict(type="rect", x0=0, y0=18, x1=18, y1=62), dict(type="rect", x0=102, y0=18, x1=120, y1=62),
              dict(type="rect", x0=0, y0=30, x1=6, y1=50), dict(type="rect", x0=114, y0=30, x1=120, y1=50)]
    for s in shapes:
        fig.add_shape(**s, line=line, layer="below")
    fig.update_xaxes(range=[-2, 122], visible=False)
    fig.update_yaxes(range=[82, -2], visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(plot_bgcolor="white", height=470, margin=dict(l=10, r=10, t=30, b=10),
                      showlegend=True, legend=dict(orientation="h", y=-0.02))
    return fig


# ---------------- TAB 1: shortlist ----------------
with tab1:
    hidden = view[view.hidden_talent].sort_values("undervalued_gap", ascending=False)
    c1, c2, c3 = st.columns(3)
    c1.metric("Attackers in view", len(view))
    c2.metric("Hidden talents", len(hidden))
    c3.metric("…with high confidence", int((hidden.confidence == "High").sum()))
    st.markdown("**Hidden talent** = top half by VAEP per 90 **and** ranked at least 20% of places "
                "higher by VAEP than by goals + assists. Their market price, often driven by goals "
                "and assists, may not reflect the threat they create.")

    fig = go.Figure()
    others = view[~view.hidden_talent]
    fig.add_trace(go.Scatter(x=others.GA_per90, y=others.vaep_per90, mode="markers", name="Other attackers",
                             marker=dict(color=GREY, size=9, line=dict(color="white", width=1)),
                             text=others.label, hovertemplate="%{text}<br>G+A/90 %{x:.2f}<br>VAEP/90 %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=hidden.GA_per90, y=hidden.vaep_per90, mode="markers", name="Hidden talent",
                             marker=dict(color=BLUE, size=13, line=dict(color="white", width=1.5)),
                             text=hidden.label, hovertemplate="%{text}<br>G+A/90 %{x:.2f}<br>VAEP/90 %{y:.2f}<extra></extra>"))
    fig.add_vline(x=view.GA_per90.median(), line_dash="dash", line_color=GREY)
    fig.add_hline(y=view.vaep_per90.median(), line_dash="dash", line_color=GREY)
    fig.update_layout(height=460, plot_bgcolor="white", xaxis_title="Goals + assists per 90 (traditional stats)",
                      yaxis_title="VAEP per 90 (threat created)", margin=dict(t=20),
                      legend=dict(orientation="h", y=1.05))
    fig.update_xaxes(gridcolor="#eeeeee"); fig.update_yaxes(gridcolor="#eeeeee")
    st.plotly_chart(fig, width="stretch")

    st.dataframe(
        hidden[["player", "team", "main_position", "minutes", "goals", "assists", "vaep_per90",
                "ci_low", "ci_high", "rank_vaep", "rank_GA", "undervalued_gap", "confidence"]].round(2),
        hide_index=True, width="stretch",
        column_config={"vaep_per90": "VAEP/90", "ci_low": "95% low", "ci_high": "95% high",
                       "rank_vaep": "VAEP rank", "rank_GA": "G+A rank", "undervalued_gap": "Places undervalued"})


# ---------------- TAB 2: player profile ----------------
with tab2:
    if view.empty:
        st.warning("No players match the filters.")
    else:
        default = int(np.argmax(view.label.str.contains("Yamal|Saka|Messi").to_numpy())) \
            if view.label.str.contains("Yamal|Saka|Messi").any() else 0
        choice = st.selectbox("Player", view.label.tolist(), index=default)
        p = view[view.label == choice].iloc[0]
        pa = acts[acts.id_player_id == p.id_player_id]

        if p.hidden_talent:
            st.success(f"Hidden talent · {p.confidence} confidence · ranked {int(p.undervalued_gap)} places higher by VAEP than by goals + assists")
        c = st.columns(5)
        c[0].metric("VAEP rank", f"#{int(p.rank_vaep)} of {len(rank)}")
        c[1].metric("G+A rank", f"#{int(p.rank_GA)}")
        c[2].metric("VAEP per 90", f"{p.vaep_per90:.2f}", help=f"95% range {p.ci_low:.2f} to {p.ci_high:.2f}")
        c[3].metric("Goals / assists", f"{int(p.goals)} / {int(p.assists)}")
        c[4].metric("Confidence", p.confidence, help="High = 360+ minutes and 95% range above zero")
        st.caption(f"{p.main_position} · {int(p.minutes)} minutes in {int(p.matches)} matches · "
                   f"95% range of VAEP per 90: {p.ci_low:.2f} to {p.ci_high:.2f}")

        left, right = st.columns([3, 2])
        with left:
            n = st.slider("Show the most valuable actions", 5, 30, 15)
            best = pa.nlargest(n, "vaep_value")
            fig = pitch_figure()
            vmax = max(best.vaep_value.max(), 1e-6)
            for _, a in best.iterrows():
                col = ORANGE if a.id_type.startswith("shot") else BLUE
                fig.add_annotation(x=a.id_end_x, y=a.id_end_y, ax=a.id_start_x, ay=a.id_start_y,
                                   xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                                   arrowhead=2, arrowsize=1, arrowwidth=1 + 3 * a.vaep_value / vmax, arrowcolor=col)
            fig.add_trace(go.Scatter(x=best.id_start_x, y=best.id_start_y, mode="markers", name="Action start",
                                     marker=dict(size=9, color=[ORANGE if t.startswith("shot") else BLUE for t in best.id_type]),
                                     customdata=np.stack([best.action, best.minute, best.vaep_value, best.p_scores], axis=1),
                                     hovertemplate="%{customdata[0]} · min %{customdata[1]}<br>VAEP %{customdata[2]:.3f}"
                                                   "<br>P(score) after %{customdata[3]:.1%}<extra></extra>"))
            fig.update_layout(title=f"Top {n} actions (orange = shot, blue = pass/carry/dribble) → attacking direction")
            st.plotly_chart(fig, width="stretch")
        with right:
            by_type = (pa.groupby("action").vaep_value.agg(["sum", "size"])
                         .rename(columns={"sum": "total VAEP", "size": "actions"})
                         .sort_values("total VAEP", ascending=False))
            fig = go.Figure(go.Bar(x=by_type["total VAEP"], y=by_type.index, orientation="h",
                                   marker_color=[BLUE if v >= 0 else ORANGE for v in by_type["total VAEP"]]))
            fig.update_layout(title="Where the value comes from", height=470, plot_bgcolor="white",
                              yaxis=dict(autorange="reversed"), xaxis_title="Total VAEP", margin=dict(t=40))
            st.plotly_chart(fig, width="stretch")

        st.markdown("**Top actions with the model's probabilities**")
        st.dataframe(best[["minute", "action", "id_success", "p_scores", "p_concedes",
                           "offensive_value", "defensive_value", "vaep_value"]].round(3),
                     hide_index=True, width="stretch",
                     column_config={"id_success": "success", "p_scores": "P(score) after",
                                    "p_concedes": "P(concede) after", "vaep_value": "VAEP"})


# ---------------- TAB 3: full ranking ----------------
with tab3:
    st.dataframe(
        view[["rank_vaep", "player", "team", "main_position", "minutes", "goals", "assists",
              "vaep_per90", "ci_low", "ci_high", "rank_GA", "undervalued_gap", "hidden_talent", "confidence"]].round(2),
        hide_index=True, width="stretch", height=600,
        column_config={"rank_vaep": "VAEP rank", "vaep_per90": "VAEP/90", "ci_low": "95% low",
                       "ci_high": "95% high", "rank_GA": "G+A rank", "undervalued_gap": "Places undervalued"})
    rho = view[["vaep_per90", "GA_per90"]].corr(method="spearman").iloc[0, 1] if len(view) > 2 else np.nan
    st.caption(f"Spearman correlation between VAEP/90 and (goals + assists)/90 for these players: {rho:.2f}. "
               "Related but not the same: VAEP also credits build-up and chance creation.")
    st.download_button("Download this list (CSV)", view.to_csv(index=False), f"{key}_attackers.csv", "text/csv")


# ---------------- TAB 4: model evidence ----------------
with tab4:
    st.markdown("One XGBoost model design, trained on **World Cup 2022 only**, tested on two tournaments "
                "it never saw. [ ] = 95% bootstrap range (resampling whole matches).")
    st.dataframe(pd.read_csv(DATA / "test_results.csv"), hide_index=True, width="stretch")
    st.caption("ROC-AUC: how well the model ranks goal-leading actions above others (0.5 = random). "
               "Brier: lower is better; beats always guessing the average. "
               "ECE: average gap between predicted and actual chance (0 = perfect).")
    c1, c2 = st.columns(2)
    c1.image(str(DATA / "calibration_test.png"), caption="Calibration on unseen tournaments")
    c2.image(str(DATA / "fig1_shap_feature_groups.png"), caption="What drives P(score) – SHAP")


# ---------------- TAB 5: about ----------------
with tab5:
    st.markdown("""
**Data:** StatsBomb Open Data (free, public event data) – FIFA World Cup 2022, UEFA Euro 2024,
Copa América 2024: 147 matches. Penalty shootouts removed. Goals checked against official scores.

**Method:** VAEP (Decroos et al., KDD 2019). Value of an action = change in P(score in next 10 actions)
minus change in P(concede). Probabilities from one XGBoost model; game state = current + 2 previous actions.

**Players shown:** attacking positions with at least 180 minutes. Values are per 90 minutes with
95% bootstrap ranges over each player's matches.

**Limits:** tournaments are short, so small samples – use the confidence label. Event data only
(no off-ball runs). A shortlist tool to support scouts, not replace them. No betting use.

**Where it runs:** this app is hosted on Streamlit Community Cloud from a GitHub repository;
the data and trained model files are in that repository, so it does not depend on any laptop.
""")
    st.caption("Data provided by StatsBomb (Hudl) under the StatsBomb Open Data licence: "
               "https://github.com/statsbomb/open-data")
