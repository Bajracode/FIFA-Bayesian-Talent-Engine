# fifa_bayesian_engine_full.py
import streamlit as st
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="FIFA Bayesian Talent Engine (Full)", layout="wide")

st.title("FIFA Bayesian Talent Engine — Full (Diagnostics, PPC, Model Comparison)")

st.markdown("""
- Added: Posterior Predictive Checks (PPC), sampler diagnostics (R-hat, ESS), baseline model, WAIC/LOO, team-level visuals, uncertainty ribbons by age.
- Requirements: `pyro-ppl`, `torch`, `arviz`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `streamlit`.
""")

# ---------- helper: imports that may fail ----------
try:
    import pyro
    import pyro.distributions as dist
    from pyro.infer import MCMC, NUTS
except Exception as e:
    st.error("Missing pyro or pyro import failed. Install pyro-ppl. Error: " + str(e))
    st.stop()

try:
    import arviz as az
except Exception as e:
    st.warning("ArviZ not installed. Diagnostics/WAIC/LOO will not run. Install `arviz` for full diagnostics.")
    az = None

# ---------- Load data ----------
@st.cache_data
def load_fifa_data(path="fifa_players.csv"):
    df_local = pd.read_csv(path)
    return df_local

data_path = st.text_input("Path to FIFA CSV", value="fifa_players.csv")
try:
    df = load_fifa_data(data_path)
except Exception as e:
    st.error(f"Failed to load data from {data_path}: {e}")
    st.stop()

st.write("Preview (first 5 rows):")
st.dataframe(df.head())

# Basic checks
required_cols = {"Name", "Team", "Preferred Position", "Age", "OVR", "POT"}
if not required_cols.issubset(set(df.columns)):
    st.error(f"CSV missing required columns. Need at least: {required_cols}")
    st.stop()

# ---------- Preprocess ----------
df = df.dropna(subset=["Team", "Preferred Position", "Age", "POT"]).reset_index(drop=True)
df["Age"] = df["Age"].astype(int)

teams = df['Team'].unique()
positions = df['Preferred Position'].unique()

team_idx = {team: i for i, team in enumerate(teams)}
pos_idx = {pos: i for i, pos in enumerate(positions)}

df['team_code'] = df['Team'].map(team_idx)
df['pos_code'] = df['Preferred Position'].map(pos_idx)

# Convert to torch tensors
device = torch.device("cpu")
age_all = torch.tensor(df['Age'].values, dtype=torch.float32, device=device)
team_all = torch.tensor(df['team_code'].values, dtype=torch.long, device=device)
pos_all = torch.tensor(df['pos_code'].values, dtype=torch.long, device=device)
pot_all = torch.tensor(df['POT'].values, dtype=torch.float32, device=device)

n_teams = len(teams)
n_positions = len(positions)
n_players = len(df)

# ---------- Model definitions ----------
def hierarchical_model(age, team, position, pot=None):
    # Hyperpriors
    team_intercept_mean = pyro.sample("team_intercept_mean", dist.Normal(70., 10.))
    team_intercept_sd = pyro.sample("team_intercept_sd", dist.HalfNormal(10.))
    with pyro.plate("teams_plate", n_teams):
        team_intercept = pyro.sample("team_intercept", dist.Normal(team_intercept_mean, team_intercept_sd))
    team_slope_mean = pyro.sample("team_slope_mean", dist.Normal(0., 2.))
    team_slope_sd = pyro.sample("team_slope_sd", dist.HalfNormal(2.))
    with pyro.plate("teams_plate2", n_teams):
        team_slope = pyro.sample("team_slope", dist.Normal(team_slope_mean, team_slope_sd))
    pos_effects = pyro.sample("pos_effects", dist.Normal(0., 3.).expand([n_positions]).to_event(1))

    sigma_player = pyro.sample("sigma_player", dist.HalfNormal(5.))

    mu = team_intercept[team] + team_slope[team] * age + pos_effects[position]

    with pyro.plate("players", len(age)):
        pyro.sample("obs", dist.Normal(mu, sigma_player), obs=pot)

def baseline_model(age, team, position, pot=None):
    # Simple baseline: global intercept + age slope only (no team, no pos)
    intercept = pyro.sample("intercept", dist.Normal(70., 10.))
    slope = pyro.sample("slope", dist.Normal(0., 1.))
    sigma = pyro.sample("sigma", dist.HalfNormal(5.))
    mu = intercept + slope * age
    with pyro.plate("players", len(age)):
        pyro.sample("obs", dist.Normal(mu, sigma), obs=pot)

# ---------- UI: sampling options ----------
st.sidebar.subheader("MCMC settings")
NUM_CHAINS = st.sidebar.selectbox("Number of chains", options=[1, 2], index=1)
NUM_SAMPLES = st.sidebar.number_input("Posterior samples per chain", min_value=100, max_value=5000, value=800, step=100)
NUM_WARMUP = st.sidebar.number_input("Warmup steps", min_value=50, max_value=2000, value=400, step=50)
fast_preview = st.sidebar.checkbox("Fast Preview (subsample dataset)", value=False)
subsample_n = st.sidebar.number_input("Subsample N (if fast)", min_value=100, max_value=n_players, value=min(500, n_players), step=50)

# ---------- Subsample for faster runs ----------
if fast_preview:
    rng = np.random.default_rng(42)
    subsample_idx = rng.choice(n_players, subsample_n, replace=False)
    df_sub = df.iloc[subsample_idx].reset_index(drop=True)
    age = torch.tensor(df_sub['Age'].values, dtype=torch.float32)
    team = torch.tensor(df_sub['team_code'].values, dtype=torch.long)
    position = torch.tensor(df_sub['pos_code'].values, dtype=torch.long)
    pot = torch.tensor(df_sub['POT'].values, dtype=torch.float32)
else:
    df_sub = df.copy()
    age = age_all
    team = team_all
    position = pos_all
    pot = pot_all

# ---------- Run models ----------
col1, col2 = st.columns(2)
with col1:
    run_hier = st.button("Run Hierarchical Model (NUTS)")
with col2:
    run_base = st.button("Run Baseline Model (NUTS)")

# storage keys
if "mcmc_hier" not in st.session_state:
    st.session_state["mcmc_hier"] = None
if "mcmc_base" not in st.session_state:
    st.session_state["mcmc_base"] = None

def run_mcmc(model_fn):
    try:
        nuts_kernel = NUTS(model_fn, target_accept_prob=0.8)
        mcmc = MCMC(nuts_kernel, num_samples=NUM_SAMPLES, warmup_steps=NUM_WARMUP, num_chains=NUM_CHAINS)
        mcmc.run(age, team, position, pot)
        return mcmc
    except Exception as e:
        st.error("MCMC failed: " + str(e))
        return None

if run_hier:
    st.info("Running hierarchical model...")
    mcmc_h = run_mcmc(hierarchical_model)
    if mcmc_h is not None:
        st.session_state.mcmc_hier = mcmc_h
        st.success("Hierarchical model finished")

if run_base:
    st.info("Running baseline model...")
    mcmc_b = run_mcmc(baseline_model)
    if mcmc_b is not None:
        st.session_state.mcmc_base = mcmc_b
        st.success("Baseline model finished")

# ---------- Helper to convert pyro MCMC -> ArviZ InferenceData (if az available) ----------
def pyro_to_arviz(mcmc_obj, model_name="model"):
    """
    Convert pyro MCMC to arviz InferenceData using az.from_pyro if available.
    Fallback: construct a dict of arrays and use az.from_dict.
    """
    if az is None:
        return None
    try:
        # az has helper from_pyro for MCMC objects
        idata = az.from_pyro(mcmc_obj)
        idata.attrs["model_name"] = model_name
        return idata
    except Exception:
        # fallback: gather samples and create dict
        samples = mcmc_obj.get_samples(group_by_chain=True)  # try group_by_chain True
        # samples is dict: name -> tensor with shape (chains, draws, ...) when group_by_chain=True
        samples_np = {}
        for k, v in samples.items():
            try:
                samples_np[k] = (v.cpu().numpy() if isinstance(v, torch.Tensor) else np.array(v))
            except Exception:
                samples_np[k] = np.array(v)
        try:
            idata = az.from_dict(posterior=samples_np)
            idata.attrs["model_name"] = model_name
            return idata
        except Exception:
            return None

# ---------- Posterior analysis if hierarchical model present ----------
if st.session_state.mcmc_hier is not None:
    st.markdown("## Hierarchical Model Results and Diagnostics")
    mcmc = st.session_state.mcmc_hier
    # get samples (group_by_chain True to preserve chains for arviz)
    try:
        posterior_samples = mcmc.get_samples(group_by_chain=True)
    except TypeError:
        # older pyro versions may ignore group_by_chain flag
        posterior_samples = mcmc.get_samples()

    # Convert to ArviZ if possible
    idata_h = pyro_to_arviz(mcmc, model_name="hierarchical") if az is not None else None

    # Diagnostics: R-hat and ESS via arviz when available
    if az is not None and idata_h is not None:
        try:
            summary_df = az.summary(idata_h, round_to=3)
            st.subheader("Sampler diagnostics (R-hat, ESS, mean, sd)")
            st.dataframe(summary_df)
        except Exception as e:
            st.warning("Failed to compute arviz summary: " + str(e))

        # WAIC / LOO require log_likelihood in idata; attempt to compute if present
        try:
            loo_h = az.loo(idata_h)
            waic_h = az.waic(idata_h)
            st.write("Model fit metrics (hierarchical):")
            st.write(loo_h)
            st.write(waic_h)
        except Exception:
            st.info("LOO/WAIC not available for hierarchical model (requires log_likelihood in posterior predictive).")

    # Extract posterior means for team intercepts and slopes if present
    # handle shape variations: posterior_samples['team_intercept'] may be shape (chains, draws, n_teams) or (chains*draws, n_teams)
    def tensor_to_numpy_arr(x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return np.array(x)

    # Attempt to collect team intercept and slope arrays
    team_intercept_samples = None
    team_slope_samples = None
    if 'team_intercept' in posterior_samples:
        team_intercept_samples = tensor_to_numpy_arr(posterior_samples['team_intercept'])
    if 'team_slope' in posterior_samples:
        team_slope_samples = tensor_to_numpy_arr(posterior_samples['team_slope'])
    if team_intercept_samples is not None:
        # If flattened shape (chains*draws, n_teams), try to reshape into (chains, draws, n_teams) if num_chains>1
        if team_intercept_samples.ndim == 2 and NUM_CHAINS > 1:
            # best-effort split by chain using MCMC.num_samples_per_chain if available
            try:
                draws_per_chain = mcmc.num_samples
                team_intercept_samples = team_intercept_samples.reshape((NUM_CHAINS, draws_per_chain, -1))
            except Exception:
                # leave as-is
                pass

        # Compute summary
        mean_intercepts = np.mean(team_intercept_samples.reshape(-1, team_intercept_samples.shape[-1]), axis=0)
        intercept_df = pd.DataFrame({
            "Team": teams,
            "Intercept_mean": mean_intercepts
        }).sort_values("Intercept_mean", ascending=False).reset_index(drop=True)
        st.subheader("Team intercepts (posterior mean)")
        st.dataframe(intercept_df)

        # Radar / bar for top teams
        st.subheader("Top 10 teams by intercept (posterior mean)")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(y="Team", x="Intercept_mean", data=intercept_df.head(10), ax=ax)
        ax.set_xlabel("Posterior mean intercept")
        st.pyplot(fig)

    # Posterior predictive check (PPC): simulate predictive POT for players
    st.subheader("Posterior Predictive Check (Predicted POT vs Actual POT)")
    # Build posterior mu for each draw: posterior_samples['team_intercept'][:, team_sub] + ...
    # Need to index team positions for the subset age/team/pos variables
    # We'll build mu_draws with shape (n_draws_total, n_data)
    try:
        # Helper to get draws x n_data for a param
        def param_draws_to_data(param_name, indexer):
            p = posterior_samples[param_name]
            p_np = tensor_to_numpy_arr(p)
            # if p_np dims: (chains, draws, n_entities) or (draws, n_entities)
            if p_np.ndim == 3:
                # (chains, draws, n_entities) => combine chains and draws into first axis
                p_flat = p_np.reshape(-1, p_np.shape[-1])
            elif p_np.ndim == 2:
                p_flat = p_np
            else:
                raise ValueError(f"Unexpected dim for {param_name}: {p_np.shape}")
            # indexer is an array of length n_data giving entity id per observation
            return p_flat[:, indexer]

        team_idx_array = np.array(team.cpu() if isinstance(team, torch.Tensor) else team)
        pos_idx_array = np.array(position.cpu() if isinstance(position, torch.Tensor) else position)
        # team_intercept and team_slope
        if 'team_intercept' in posterior_samples and 'team_slope' in posterior_samples and 'pos_effects' in posterior_samples:
            t_intercept_draws = param_draws_to_data('team_intercept', team_idx_array)
            t_slope_draws = param_draws_to_data('team_slope', team_idx_array)
            pos_draws = param_draws_to_data('pos_effects', pos_idx_array)

            # compute mu_draws: (n_draws, n_data)
            age_np = np.array(age.cpu() if isinstance(age, torch.Tensor) else age)
            mu_draws = t_intercept_draws + (t_slope_draws * age_np[np.newaxis, :]) + pos_draws

            # If sigma_player in samples:
            if 'sigma_player' in posterior_samples:
                sigma_draws = tensor_to_numpy_arr(posterior_samples['sigma_player'])
                # reshape sigma to (n_draws, 1)
                if sigma_draws.ndim == 2:
                    sigma_flat = sigma_draws.reshape(-1)
                else:
                    sigma_flat = sigma_draws.reshape(-1)
            else:
                sigma_flat = None

            # Sample predictive values (one sample per posterior draw)
            rng = np.random.default_rng(123)
            if sigma_flat is not None:
                pred_draws = mu_draws + rng.normal(size=mu_draws.shape) * sigma_flat[:, np.newaxis]
            else:
                pred_draws = mu_draws

            actual_pot = np.array(pot.cpu() if isinstance(pot, torch.Tensor) else pot)

            # Plot predicted mean vs actual
            pred_mean = pred_draws.mean(axis=0)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(actual_pot, pred_mean, alpha=0.6, s=10)
            ax.plot([actual_pot.min(), actual_pot.max()], [actual_pot.min(), actual_pot.max()], color="k", linestyle="--")
            ax.set_xlabel("Actual POT")
            ax.set_ylabel("Predicted POT (posterior mean)")
            ax.set_title("PPC: predicted vs actual")
            st.pyplot(fig)

            # Residuals histogram
            resid = pred_mean - actual_pot
            fig, ax = plt.subplots(figsize=(6, 3))
            sns.histplot(resid, kde=True, ax=ax)
            ax.set_title("Residuals (predicted_mean - actual)")
            st.pyplot(fig)

        else:
            st.info("Not enough posterior parameters saved to perform PPC (expected 'team_intercept', 'team_slope', 'pos_effects').")
    except Exception as e:
        st.warning("PPC generation failed: " + str(e))

    # Uncertainty ribbon: expected POT vs age for a selected team+position or for global mean
    st.subheader("Uncertainty ribbon: Expected POT vs Age")
    sel_team_name = st.selectbox("Select team (for ribbon)", options=list(teams), index=0)
    sel_team_id = team_idx[sel_team_name]
    # produce ages 16..40
    ages_grid = np.arange(16, 41)
    try:
        if 'team_intercept' in posterior_samples and 'team_slope' in posterior_samples and 'pos_effects' in posterior_samples:
            ti_draws = param_draws_to_data('team_intercept', np.full(len(ages_grid), sel_team_id))
            ts_draws = param_draws_to_data('team_slope', np.full(len(ages_grid), sel_team_id))
            # pick a position to show, allow choice
            sel_pos_name = st.selectbox("Select position (for ribbon)", options=list(positions), index=0)
            sel_pos_id = pos_idx[sel_pos_name]
            pos_draws_grid = param_draws_to_data('pos_effects', np.full(len(ages_grid), sel_pos_id))
            # compute mu_draws_grid (n_draws, n_ages)
            mu_grid = ti_draws + ts_draws * ages_grid[np.newaxis, :] + pos_draws_grid
            mu_mean = mu_grid.mean(axis=0)
            mu_hpd_low = np.percentile(mu_grid, 5, axis=0)
            mu_hpd_high = np.percentile(mu_grid, 95, axis=0)

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(ages_grid, mu_mean, label="Posterior mean")
            ax.fill_between(ages_grid, mu_hpd_low, mu_hpd_high, alpha=0.3, label="5-95% CI")
            ax.set_xlabel("Age")
            ax.set_ylabel("Expected POT")
            ax.set_title(f"Expected POT vs Age — Team: {sel_team_name} — Pos: {sel_pos_name}")
            ax.legend()
            st.pyplot(fig)
    except Exception as e:
        st.warning("Uncertainty ribbon failed: " + str(e))

    # Player-level posterior histogram (what user had before)
    st.subheader("Player Posterior Distribution (growth headroom)")
    try:
        # Recompute growth headroom: posterior_mu - pot
        if 'team_intercept' in posterior_samples and 'team_slope' in posterior_samples and 'pos_effects' in posterior_samples:
            # reuse mu_draws, pred_draws from PPC block if available; otherwise recompute for first N draws
            # here just compute mu_draws mean growth
            growth_draws = (mu_draws - np.array(pot.cpu() if isinstance(pot, torch.Tensor) else pot)[np.newaxis, :])
            df_headroom = df_sub.copy()
            df_headroom['Expected Growth'] = growth_draws.mean(axis=0)
            p5 = np.percentile(growth_draws, 5, axis=0)
            p95 = np.percentile(growth_draws, 95, axis=0)
            df_headroom['Growth 5%'] = p5
            df_headroom['Growth 95%'] = p95
            st.dataframe(df_headroom[['Name', 'Team', 'Preferred Position', 'Age', 'OVR', 'POT', 'Expected Growth', 'Growth 5%', 'Growth 95%']].sort_values("Expected Growth", ascending=False).head(20))
            # histogram for selected player
            player_idx = st.selectbox("Select player to view posterior growth", options=list(range(len(df_headroom))), format_func=lambda i: df_headroom.iloc[i]['Name'])
            fig, ax = plt.subplots(figsize=(6,3))
            sns.histplot(growth_draws[:, player_idx], kde=True, ax=ax)
            ax.set_xlabel("Predicted Growth (POT - OVR)")
            st.pyplot(fig)
        else:
            st.info("Posterior growth cannot be computed — required posterior variables missing.")
    except Exception as e:
        st.warning("Player posterior distribution plotting failed: " + str(e))

    # Download predictions CSV
    try:
        csv = df_headroom.to_csv(index=False).encode()
        st.download_button("Download Growth Predictions CSV", csv, "growth_predictions.csv")
    except Exception:
        pass

# ---------- Baseline model diagnostics and comparison ----------
if st.session_state.mcmc_base is not None:
    st.markdown("## Baseline Model Results and Diagnostics")
    mcmc_b = st.session_state.mcmc_base
    if az is not None:
        idata_b = pyro_to_arviz(mcmc_b, model_name="baseline")
        if idata_b is not None:
            try:
                st.subheader("Baseline diagnostics (summary)")
                st.dataframe(az.summary(idata_b).round(3))
            except Exception as e:
                st.warning("ArviZ summary for baseline failed: " + str(e))

            # Attempt to compare models if both idata present
            if st.session_state.mcmc_hier is not None:
                mcmc_h = st.session_state.mcmc_hier
                idata_h2 = pyro_to_arviz(mcmc_h, model_name="hierarchical")
                if idata_h2 is not None:
                    st.subheader("Model comparison (if LOO/WAIC available)")
                    try:
                        # compute loo for both
                        loo_b = az.loo(idata_b)
                        loo_h = az.loo(idata_h2)
                        comp = az.compare({"baseline": idata_b, "hierarchical": idata_h2}, method="stacking")
                        st.write("LOO baseline:", loo_b)
                        st.write("LOO hierarchical:", loo_h)
                        st.write("Comparison table (az.compare):")
                        st.dataframe(comp)
                    except Exception as e:
                        st.info("WAIC/LOO/compare failed (may need log_likelihood in posterior_predictive). Error: "+str(e))
                else:
                    st.info("Could not convert hierarchical MCMC to arviz InferenceData for comparison.")
        else:
            st.info("Could not convert baseline MCMC to arviz InferenceData.")
    else:
        st.info("Install ArviZ (`pip install arviz`) to enable model comparison and diagnostics.")

# ---------- Extra: quick summary panel ----------
st.sidebar.subheader("Quick Checklist (what to look for after run)")
st.sidebar.markdown("""
- Posterior predictive scatter: points should be near diagonal.
- R-hat near 1.00 for all parameters.
- ESS not too small.
- Compare baseline -> hierarchical via LOO/WAIC (lower is better).
- Look at team intercepts to see meaningful variance across teams.
""")

st.sidebar.subheader("Troubleshooting")
st.sidebar.markdown("""
- If MCMC fails: try lowering `num_chains` to 1, reduce `num_samples` or increase `warmup`.
- If imports fail: install packages and restart Streamlit.
- If ArviZ compare fails: In some Pyro versions you may need to compute `log_likelihood` in the model or use `az.from_dict` fallback.
""")
