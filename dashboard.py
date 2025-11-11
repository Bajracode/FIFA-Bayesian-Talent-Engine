"""
Streamlit app: FIFA Bayesian Talent Engine (fixed, NumPyro backend)
- Background MCMC via ThreadPoolExecutor (non-blocking)
- NumPyro/JAX backend (fast, GPU-ready)
- Minimal invasive changes: UI unchanged, only MCMC backend and conversions adjusted
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import logging
import concurrent.futures
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial
import time
from typing import Dict, Any

# --- New imports for NumPyro/JAX ---
try:
    import jax
    import jax.numpy as jnp
    from jax.config import config as jax_config
    jax_config.update("jax_enable_x64", False)  # use float32 for speed unless you need 64
    import numpyro
    import numpyro.distributions as npdist
    from numpyro.infer import MCMC, NUTS
    from numpyro import sample as n_sample, plate as n_plate, deterministic as n_deterministic
    numpyro_available = True
except Exception as e:
    numpyro_available = False
    # We'll still attempt to fall back to pyro if present; otherwise error later.
    # Do not show an error here; we will show when user tries to run model.

# ---------- Setup logging ----------
logger = logging.getLogger("fifa_bayes")
if not logger.handlers:
    handler = logging.FileHandler("fifa_bayes.log")
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

st.set_page_config(page_title="FIFA Bayesian Talent Engine (Fixed)", layout="wide")
st.title("FIFA Bayesian Talent Engine — Fixed")

# ---------- Constants (named magic numbers) ----------
TEAM_PRIOR_MEAN = 70.0
TEAM_PRIOR_SD = 10.0
SLOPE_PRIOR_SD = 2.0
POS_EFFECT_SD = 3.0
SIGMA_PRIOR_SD = 5.0

# ---------- Optional: import arviz ----------
try:
    import arviz as az
    arviz_available = True
except Exception:
    az = None
    arviz_available = False
    st.warning("ArviZ not installed. Diagnostics like LOO/WAIC will be unavailable.")

# ---------- Helper: Data loading and preprocessing ----------
@st.cache_data(max_entries=4)
def load_fifa_data(path: str = "fifa_players.csv") -> pd.DataFrame:
    """
    Load CSV and perform minimal checks.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        logger.exception("CSV file not found at %s", path)
        raise
    # minimal required columns
    req = {"Name", "Team", "Preferred Position", "Age", "OVR", "POT"}
    if not req.issubset(set(df.columns)):
        missing = req - set(df.columns)
        logger.error("CSV missing required columns: %s", missing)
        raise KeyError(f"CSV missing required columns: {missing}")
    df = df.dropna(subset=list(req)).copy()
    # ensure ints
    df["Age"] = df["Age"].astype(int)
    df["OVR"] = pd.to_numeric(df["OVR"], errors="coerce").fillna(df["OVR"].median())
    df["POT"] = pd.to_numeric(df["POT"], errors="coerce").fillna(df["POT"].median())
    df = df.reset_index(drop=True)
    return df

def prepare_tensors(df: pd.DataFrame, device: torch.device):
    """
    Map teams/positions, center age, and return torch tensors used by plotting code.
    (We leave these as torch tensors for compatibility with existing plotting code.)
    """
    teams = df["Team"].astype(str).unique()
    positions = df["Preferred Position"].astype(str).unique()
    team_idx = {team: i for i, team in enumerate(teams)}
    pos_idx = {pos: i for i, pos in enumerate(positions)}
    df["team_code"] = df["Team"].map(team_idx).astype(int)
    df["pos_code"] = df["Preferred Position"].map(pos_idx).astype(int)

    # center age so intercept is meaningful
    age_mean = df["Age"].mean()
    df["age_c"] = df["Age"] - age_mean

    age_t = torch.tensor(df["age_c"].values, dtype=torch.float32, device=device)
    team_t = torch.tensor(df["team_code"].values, dtype=torch.long, device=device)
    pos_t = torch.tensor(df["pos_code"].values, dtype=torch.long, device=device)
    pot_t = torch.tensor(df["POT"].values, dtype=torch.float32, device=device)
    ovr_t = torch.tensor(df["OVR"].values, dtype=torch.float32, device=device)

    meta = {
        "teams": list(teams),
        "positions": list(positions),
        "age_mean": float(age_mean),
        "n_teams": len(teams),
        "n_positions": len(positions),
        "n_players": len(df)
    }
    return df, age_t, team_t, pos_t, pot_t, ovr_t, meta

# ---------- NumPyro model definitions (mirror your Pyro models) ----------
# We'll define the same hierarchical structure but in NumPyro (JAX)
if numpyro_available:
    def hierarchical_model(age, team, position, pot=None, ovr=None, n_teams=None, n_positions=None):
        """
        NumPyro hierarchical model:
        team intercepts & slopes, position effects, sigma, optional beta_ovr
        `age`, `pot`, `ovr` are expected as jnp arrays; `team`/`position` as int arrays.
        """
        # Hyperpriors (on transformed scale; age is centered in preprocessing)
        team_intercept_mean = n_sample("team_intercept_mean", npdist.Normal(TEAM_PRIOR_MEAN, TEAM_PRIOR_SD))
        team_intercept_sd = n_sample("team_intercept_sd", npdist.HalfNormal(TEAM_PRIOR_SD))
        with n_plate("teams", n_teams):
            team_intercept = n_sample("team_intercept", npdist.Normal(team_intercept_mean, team_intercept_sd))

        team_slope_mean = n_sample("team_slope_mean", npdist.Normal(0.0, SLOPE_PRIOR_SD))
        team_slope_sd = n_sample("team_slope_sd", npdist.HalfNormal(SLOPE_PRIOR_SD))
        with n_plate("teams_slope", n_teams):
            team_slope = n_sample("team_slope", npdist.Normal(team_slope_mean, team_slope_sd))

        pos_effects = n_sample("pos_effects", npdist.Normal(0.0, POS_EFFECT_SD).expand([n_positions]).to_event(1))

        sigma_player = n_sample("sigma_player", npdist.HalfNormal(SIGMA_PRIOR_SD))

        mu = team_intercept[team] + team_slope[team] * age + pos_effects[position]
        if ovr is not None:
            beta_ovr = n_sample("beta_ovr", npdist.Normal(0.0, 1.0))
            mu = mu + beta_ovr * (ovr - jnp.mean(ovr))

        with n_plate("players", age.shape[0]):
            n_sample("obs", npdist.Normal(mu, sigma_player), obs=pot)
            # numpyro deterministic value for log_likelihood (shape: n_data)
            n_deterministic("log_likelihood", npdist.Normal(mu, sigma_player).log_prob(pot))

    def baseline_model(age, team, position, pot=None, ovr=None, n_teams=None, n_positions=None):
        intercept = n_sample("intercept", npdist.Normal(TEAM_PRIOR_MEAN, TEAM_PRIOR_SD))
        slope = n_sample("slope", npdist.Normal(0.0, 1.0))
        pos_effects_baseline = n_sample("pos_effects_baseline", npdist.Normal(0.0, POS_EFFECT_SD).expand([n_positions]).to_event(1))
        beta_ovr = n_sample("beta_ovr", npdist.Normal(0.0, 1.0))
        sigma = n_sample("sigma", npdist.HalfNormal(SIGMA_PRIOR_SD))
        mu = intercept + slope * age + pos_effects_baseline[position] + beta_ovr * (ovr - jnp.mean(ovr))
        with n_plate("players", age.shape[0]):
            n_sample("obs", npdist.Normal(mu, sigma), obs=pot)
            n_deterministic("log_likelihood", npdist.Normal(mu, sigma).log_prob(pot))
else:
    # Keep placeholders (so UI doesn't break); will raise if user attempts to run without numpyro.
    def hierarchical_model(*args, **kwargs):
        raise RuntimeError("NumPyro not available. Install numpyro and jax to run models.")
    def baseline_model(*args, **kwargs):
        raise RuntimeError("NumPyro not available. Install numpyro and jax to run models.")

# ---------- Helper conversion functions ----------
def torch_to_numpy(x):
    """Convert a torch tensor to numpy on CPU."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)

def tensors_to_jnp(age_t, team_t, pos_t, pot_t, ovr_t, use_gpu_flag: bool):
    """
    Convert torch/np arrays to jnp arrays for NumPyro run.
    We intentionally keep original torch tensors for plotting; conversion is only for the MCMC call.
    """
    age_np = torch_to_numpy(age_t)
    team_np = torch_to_numpy(team_t).astype(int)
    pos_np = torch_to_numpy(pos_t).astype(int)
    pot_np = torch_to_numpy(pot_t)
    ovr_np = torch_to_numpy(ovr_t)
    # Optionally convert to jnp on chosen device (jax uses default backend selection)
    age_j = jnp.array(age_np)
    team_j = jnp.array(team_np)
    pos_j = jnp.array(pos_np)
    pot_j = jnp.array(pot_np)
    ovr_j = jnp.array(ovr_np)
    return age_j, team_j, pos_j, pot_j, ovr_j

def posterior_to_numpy(mcmc_obj):
    """
    Convert an MCMC object (NumPyro or Pyro) to a dict of numpy arrays with draws as first dim.
    For NumPyro: get_samples(group_by_chain=True) returns (chains, draws, ...), we merge chains.
    For Pyro (fallback) the earlier code's approach is retained.
    """
    out = {}
    # NumPyro MCMC has get_samples() returning array with dims (num_samples, ...) when group_by_chain=False,
    # or get_samples(group_by_chain=True) -> shape (num_chains, num_samples, ...).
    try:
        samples = mcmc_obj.get_samples(group_by_chain=True)
        # samples: dict name -> ndarray shape (chains, draws, ...)
        for k, v in samples.items():
            arr = np.array(v)
            if arr.ndim >= 3 and arr.shape[0] > 1:
                arr = arr.reshape(-1, *arr.shape[2:])
            out[k] = arr
        return out
    except Exception:
        # fallback to pyro-like behaviour (you had this)
        try:
            samples = mcmc_obj.get_samples()
        except Exception as e:
            raise RuntimeError("Unable to extract samples from MCMC object: " + str(e))
        for name, val in samples.items():
            if isinstance(val, torch.Tensor):
                arr = val.cpu().numpy()
            else:
                arr = np.array(val)
            if arr.ndim >= 3 and arr.shape[0] > 1:
                arr = arr.reshape(-1, *arr.shape[2:])
            out[name] = arr
        return out

# ---------- Background runner for MCMC using NumPyro ----------
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

def run_mcmc_background_numpyro(model_fn, age_t, team_t, pos_t, pot_t, ovr_t, n_teams, n_positions,
                                num_samples, warmup_steps, num_chains, seed=1234, use_gpu=False):
    """
    Runs NumPyro MCMC in a background thread and returns a Future.
    We set JAX devices implicitly: JAX will use GPU if available in the environment.
    """
    def _task():
        if not numpyro_available:
            raise RuntimeError("NumPyro/JAX not available in this Python environment.")
        # Convert tensors to jnp arrays for MCMC call
        age_j, team_j, pos_j, pot_j, ovr_j = tensors_to_jnp(age_t, team_t, pos_t, pot_t, ovr_t, use_gpu_flag=use_gpu)
        # PRNG key
        key = jax.random.PRNGKey(int(seed))
        # Build kernel and MCMC
        kernel = NUTS(model_fn, target_accept_prob=0.8)
        mcmc = MCMC(kernel, num_warmup=int(warmup_steps), num_samples=int(num_samples), num_chains=int(num_chains))
        start = time.time()
        logger.info("NumPyro MCMC start: samples=%s warmup=%s chains=%s seed=%s", num_samples, warmup_steps, num_chains, seed)
        # run (pass meta sizes)
        mcmc.run(key, age=age_j, team=team_j, position=pos_j, pot=pot_j, ovr=ovr_j, n_teams=n_teams, n_positions=n_positions)
        elapsed = time.time() - start
        logger.info("NumPyro MCMC finished in %.1fs", elapsed)
        return mcmc
    future = executor.submit(_task)
    return future

# ---------- Streamlit UI ----------
with st.sidebar:
    st.header("Execution & Diagnostics")
    data_path = st.text_input("Path to FIFA CSV", value="fifa_players.csv")
    use_gpu = st.checkbox("Use GPU (jax) if available", value=False)
    seed = st.number_input("RNG seed", min_value=0, max_value=2**31-1, value=1234, step=1)
    num_chains = st.selectbox("Number of chains", [1, 2], index=1)
    num_samples = st.number_input("Posterior samples per chain", min_value=100, max_value=5000, value=800, step=100)
    num_warmup = st.number_input("Warmup steps", min_value=50, max_value=2000, value=400, step=50)
    fast_preview = st.checkbox("Fast Preview (subsample)", value=False)
    subsample_n = st.number_input("Subsample N", min_value=100, max_value=10000, value=500, step=50)

# Load data (with error handling)
try:
    df = load_fifa_data(data_path)
except FileNotFoundError:
    st.error("Data file not found. Update path in sidebar.")
    st.stop()
except KeyError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error("Failed to load data: " + str(e))
    logger.exception("Failed to load data")
    st.stop()

# Device selection for torch parts (keep for plotting tensors)
if use_gpu and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Prepare tensors and meta
try:
    df_prepared, age_t, team_t, pos_t, pot_t, ovr_t, meta = prepare_tensors(df, device)
except Exception as e:
    st.error("Failed to prepare tensors: " + str(e))
    logger.exception("prepare_tensors failed")
    st.stop()

# Subsample if requested
if fast_preview:
    subsample_n = min(int(subsample_n), meta["n_players"])
    rng = np.random.default_rng(seed)
    idx = rng.choice(meta["n_players"], subsample_n, replace=False)
    df_sub = df_prepared.iloc[idx].reset_index(drop=True)
    # re-prepare tensors for subsample (center age again)
    df_sub["age_c"] = df_sub["Age"] - df_sub["Age"].mean()
    age_t = torch.tensor(df_sub["age_c"].values, dtype=torch.float32, device=device)
    team_t = torch.tensor(df_sub["team_code"].values, dtype=torch.long, device=device)
    pos_t = torch.tensor(df_sub["pos_code"].values, dtype=torch.long, device=device)
    pot_t = torch.tensor(df_sub["POT"].values, dtype=torch.float32, device=device)
    ovr_t = torch.tensor(df_sub["OVR"].values, dtype=torch.float32, device=device)
    meta["n_players"] = len(df_sub)
    df_view = df_sub.copy()
else:
    df_view = df_prepared.copy()

# Buttons to run models
col1, col2 = st.columns([1, 1])
with col1:
    run_hier = st.button("Run Hierarchical Model (background)")
with col2:
    run_base = st.button("Run Baseline Model (background)")

# Initialize session_state holders
if "futures" not in st.session_state:
    st.session_state["futures"] = {}
if "mcmc_results" not in st.session_state:
    st.session_state["mcmc_results"] = {}
if "meta" not in st.session_state:
    st.session_state["meta"] = meta

# Submit jobs (NumPyro)
if run_hier:
    if not numpyro_available:
        st.error("NumPyro/JAX not available. Install numpyro and jax to run fast MCMC.")
    else:
        fut = run_mcmc_background_numpyro(hierarchical_model, age_t, team_t, pos_t, pot_t, ovr_t,
                                         n_teams=meta["n_teams"], n_positions=meta["n_positions"],
                                         num_samples=num_samples, warmup_steps=num_warmup, num_chains=num_chains,
                                         seed=int(seed), use_gpu=use_gpu)
        st.session_state["futures"]["hier"] = fut
        st.info("Hierarchical model submitted — running in background. App will refresh; use panel below to monitor progress.")
        logger.info("Hierarchical model job submitted")

if run_base:
    if not numpyro_available:
        st.error("NumPyro/JAX not available. Install numpyro and jax to run fast MCMC.")
    else:
        fut = run_mcmc_background_numpyro(baseline_model, age_t, team_t, pos_t, pot_t, ovr_t,
                                         n_teams=meta["n_teams"], n_positions=meta["n_positions"],
                                         num_samples=num_samples, warmup_steps=num_warmup, num_chains=num_chains,
                                         seed=int(seed)+1, use_gpu=use_gpu)
        st.session_state["futures"]["base"] = fut
        st.info("Baseline model submitted — running in background.")
        logger.info("Baseline model job submitted")

# Monitor futures and collect results
for key, fut in list(st.session_state["futures"].items()):
    if fut.done():
        try:
            mcmc_obj = fut.result()
            st.session_state["mcmc_results"][key] = mcmc_obj
            del st.session_state["futures"][key]
            st.success(f"Model '{key}' finished and cached.")
            logger.info("Model %s finished and cached", key)
        except Exception as e:
            st.error(f"Model {key} failed: {e}")
            logger.exception("Background MCMC failed for %s", key)
            del st.session_state["futures"][key]
    else:
        st.info(f"Model '{key}' is running... (refresh to update status)")

# ---------------- UI Tabs ----------------
tabs = st.tabs(["Data", "Posterior Diagnostics", "Predictive Checks", "Team/Player Insights"])
# Data tab
with tabs[0]:
    st.subheader("Data preview")
    st.write(f"Players: {meta['n_players']}, Teams: {meta['n_teams']}, Positions: {meta['n_positions']}")
    st.dataframe(df_view.head(200))
    st.markdown("**Quick stats**")
    st.write(df_view[["Age", "OVR", "POT"]].describe().T)

# Posterior Diagnostics tab
with tabs[1]:
    st.subheader("Posterior Diagnostics")
    if "hier" in st.session_state["mcmc_results"]:
        mcmc_h = st.session_state["mcmc_results"]["hier"]
        st.write("Hierarchical model available.")
        # convert to numpy
        try:
            samples_h = posterior_to_numpy(mcmc_h)
            if arviz_available:
                try:
                    # Convert NumPyro MCMC to ArviZ InferenceData
                    try:
                        idata_h = az.from_numpyro(mcmc_h)
                    except Exception:
                        # fallback: build dict
                        idata_h = None
                    if idata_h is not None:
                        st.write("ArviZ summary (selected):")
                        summary_df = az.summary(idata_h).reset_index()
                        # show only key cols
                        keep_cols = [c for c in ["index", "mean", "sd", "r_hat", "ess_bulk"] if c in summary_df.columns]
                        st.dataframe(summary_df[keep_cols].sort_values("r_hat"))
                    else:
                        st.info("ArviZ conversion returned None; you may need a recent arviz.")
                except Exception as e:
                    st.warning("ArviZ conversion failed: " + str(e))
            else:
                st.info("Install ArviZ to view richer diagnostics (pip install arviz).")
        except Exception as e:
            st.error("Failed to process hierarchical posterior: " + str(e))
            logger.exception("posterior_to_numpy failed")
    else:
        st.info("Hierarchical model not yet run. Submit from sidebar.")

    st.markdown("---")
    if "base" in st.session_state["mcmc_results"]:
        mcmc_b = st.session_state["mcmc_results"]["base"]
        st.write("Baseline model available.")
        try:
            samples_b = posterior_to_numpy(mcmc_b)
            if arviz_available:
                try:
                    idata_b = az.from_numpyro(mcmc_b)
                    st.write("Baseline ArviZ summary (selected):")
                    summary_df_b = az.summary(idata_b).reset_index()
                    keep_cols = [c for c in ["index", "mean", "sd", "r_hat", "ess_bulk"] if c in summary_df_b.columns]
                    st.dataframe(summary_df_b[keep_cols])
                except Exception as e:
                    st.warning("ArviZ conversion for baseline failed: " + str(e))
            else:
                st.info("Install ArviZ to view richer diagnostics.")
        except Exception as e:
            st.error("Failed to process baseline posterior: " + str(e))
            logger.exception("posterior_to_numpy baseline failed")
    else:
        st.info("Baseline model not yet run. Submit from sidebar.")

    # Try model comparison if both run and arviz available
    if arviz_available and "hier" in st.session_state["mcmc_results"] and "base" in st.session_state["mcmc_results"]:
        try:
            idata_h = az.from_numpyro(st.session_state["mcmc_results"]["hier"])
            idata_b = az.from_numpyro(st.session_state["mcmc_results"]["base"])
            st.subheader("Model comparison (LOO if available)")
            try:
                loo_h = az.loo(idata_h)
                loo_b = az.loo(idata_b)
                st.write("LOO hierarchical:", loo_h)
                st.write("LOO baseline:", loo_b)
                comp = az.compare({"hierarchical": idata_h, "baseline": idata_b}, method="stacking")
                st.dataframe(comp)
            except Exception as e:
                st.info("LOO/compare failed (need log_likelihood in idata). Error: " + str(e))
        except Exception as e:
            st.warning("Failed to convert models for comparison: " + str(e))

# Predictive Checks tab
with tabs[2]:
    st.subheader("Posterior Predictive Checks")
    if "hier" in st.session_state["mcmc_results"]:
        mcmc = st.session_state["mcmc_results"]["hier"]
        try:
            samples = posterior_to_numpy(mcmc)
            # compute mu_draws
            team_idx_array = np.array(team_t.cpu() if team_t.device.type == "cpu" else team_t.detach().cpu())
            pos_idx_array = np.array(pos_t.cpu() if pos_t.device.type == "cpu" else pos_t.detach().cpu())
            age_array = np.array(age_t.cpu() if age_t.device.type == "cpu" else age_t.detach().cpu())
            ovr_array = np.array(ovr_t.cpu() if ovr_t.device.type == "cpu" else ovr_t.detach().cpu())
            mu_draws, sigma_draws = compute_mu_draws(samples, team_idx_array, pos_idx_array, age_array, ovr_array)
            # predictive draws (one per posterior draw)
            rng = np.random.default_rng(seed)
            if sigma_draws is not None:
                pred_draws = mu_draws + rng.normal(size=mu_draws.shape) * sigma_draws[:, np.newaxis]
            else:
                pred_draws = mu_draws

            actual = np.array(pot_t.cpu() if pot_t.device.type == "cpu" else pot_t.detach().cpu())
            pred_mean = pred_draws.mean(axis=0)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(actual, pred_mean, alpha=0.6, s=8)
            ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()], "--", color="k")
            ax.set_xlabel("Actual POT")
            ax.set_ylabel("Predicted POT (posterior mean)")
            ax.set_title("PPC: predicted mean vs actual")
            plt.tight_layout()
            st.pyplot(fig)

            # residuals
            resid = pred_mean - actual
            fig2, ax2 = plt.subplots(figsize=(6, 3))
            sns.histplot(resid, kde=True, ax=ax2)
            ax2.set_title("Residuals (predicted_mean - actual)")
            plt.tight_layout()
            st.pyplot(fig2)

            # example player posterior distribution
            st.markdown("### Player posterior growth headroom")
            growth_draws = (mu_draws - actual[np.newaxis, :])
            df_headroom = df_view.copy()
            df_headroom["Expected Growth"] = growth_draws.mean(axis=0)
            df_headroom["Growth 5%"] = np.percentile(growth_draws, 5, axis=0)
            df_headroom["Growth 95%"] = np.percentile(growth_draws, 95, axis=0)
            st.dataframe(df_headroom[["Name", "Team", "Preferred Position", "Age", "OVR", "POT", "Expected Growth"]].sort_values("Expected Growth", ascending=False).head(20))
            # single-player histogram
            player_idx = st.selectbox("Select player to inspect", options=list(range(len(df_headroom))),
                                      format_func=lambda i: df_headroom.iloc[i]["Name"])
            fig3, ax3 = plt.subplots(figsize=(6, 3))
            sns.histplot(growth_draws[:, player_idx], kde=True, ax=ax3)
            ax3.set_xlabel("Predicted Growth (POT - actual)")
            plt.tight_layout()
            st.pyplot(fig3)

        except Exception as e:
            st.error("PPC failed: " + str(e))
            logger.exception("PPC failure")
    else:
        st.info("Run hierarchical model first to inspect PPC.")

# Team / Player insights tab
with tabs[3]:
    st.subheader("Team / Player Insights")
    # Show team intercept means if hierarchical run exists
    if "hier" in st.session_state["mcmc_results"]:
        mcmc = st.session_state["mcmc_results"]["hier"]
        try:
            samples = posterior_to_numpy(mcmc)
            if "team_intercept" in samples:
                intercepts = samples["team_intercept"]  # (n_draws, n_teams)
                team_means = intercepts.mean(axis=0)
                df_teams = pd.DataFrame({"Team": meta["teams"], "Intercept_mean": team_means})
                st.subheader("Team intercepts (posterior means)")
                st.dataframe(df_teams.sort_values("Intercept_mean", ascending=False))
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(x="Intercept_mean", y="Team", data=df_teams.sort_values("Intercept_mean", ascending=False).head(10), ax=ax)
                ax.set_xlabel("Posterior mean intercept")
                plt.tight_layout()
                st.pyplot(fig)
            if "team_slope" in samples:
                slopes = samples["team_slope"].mean(axis=0)
                df_slope = pd.DataFrame({"Team": meta["teams"], "Slope_mean": slopes})
                st.subheader("Team slopes (age effect, posterior mean)")
                st.dataframe(df_slope.sort_values("Slope_mean", ascending=False))
                fig2, ax2 = plt.subplots(figsize=(8, 4))
                sns.barplot(x="Slope_mean", y="Team", data=df_slope.sort_values("Slope_mean", ascending=False).head(10), ax=ax2)
                ax2.set_xlabel("Posterior mean slope")
                plt.tight_layout()
                st.pyplot(fig2)
        except Exception as e:
            st.error("Failed to compute team-level summaries: " + str(e))
            logger.exception("team-level summaries failed")
    else:
        st.info("Run hierarchical model to get team-level insights.")

# Optional: Save outputs if user wants
st.sidebar.markdown("---")
if st.sidebar.button("Export last hierarchical posterior to CSV"):
    if "hier" in st.session_state["mcmc_results"]:
        try:
            samples = posterior_to_numpy(st.session_state["mcmc_results"]["hier"])
            # create a compact CSV of selected parameters (posterior mean per team/pos)
            out = {}
            if "team_intercept" in samples:
                out["team_intercept_mean"] = samples["team_intercept"].mean(axis=0)
            if "team_slope" in samples:
                out["team_slope_mean"] = samples["team_slope"].mean(axis=0)
            # flatten to DataFrame
            df_out = pd.DataFrame(out)
            df_out.insert(0, "Team", meta["teams"])
            csv = df_out.to_csv(index=False).encode()
            st.download_button("Download team-level posterior means CSV", csv, file_name="team_posteriors.csv")
        except Exception as e:
            st.error("Export failed: " + str(e))
    else:
        st.error("No hierarchical posterior cached.")

st.sidebar.markdown("App notes:")
st.sidebar.write("""
- This app now runs MCMC using NumPyro/JAX when available (much faster, especially with GPU).
- Install `numpyro` and the appropriate `jax` / `jaxlib` for GPU to benefit.
- For large datasets, prefer `fast preview` or run with GPU.
""")
