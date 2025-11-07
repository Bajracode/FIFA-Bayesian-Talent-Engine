# fifa_bayesian_engine.py
import streamlit as st
import pandas as pd
import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import MCMC, NUTS
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------
# --- Streamlit Page Config
# ---------------------
st.set_page_config(page_title="FIFA Bayesian Talent Engine", layout="wide")

st.title("FIFA Bayesian Talent Engine")
st.markdown(
    """
    Explore probabilistic growth predictions for FIFA players.
    Each club has a latent development curve; positions and age trajectories influence POT.
    Growth headroom (POT − OVR) is modeled probabilistically.
    """
)

# ---------------------
# --- Load FIFA Data
# ---------------------
@st.cache_data
def load_fifa_data():
    df = pd.read_csv("fifa_players.csv")  # Replace with your CSV
    return df

df = load_fifa_data()
st.write("Top players preview:", df.head())

# ---------------------
# --- Preprocess
# ---------------------
teams = df['Team'].unique()
positions = df['Preferred Position'].unique()

team_idx = {team: i for i, team in enumerate(teams)}
pos_idx = {pos: i for i, pos in enumerate(positions)}

df['team_code'] = df['Team'].map(team_idx)
df['pos_code'] = df['Preferred Position'].map(pos_idx)

# Convert to tensors
age = torch.tensor(df['Age'].values, dtype=torch.float)
team = torch.tensor(df['team_code'].values, dtype=torch.long)
position = torch.tensor(df['pos_code'].values, dtype=torch.long)
pot = torch.tensor(df['POT'].values, dtype=torch.float)

n_teams = len(teams)
n_positions = len(positions)
n_players = len(df)

# ---------------------
# --- Pyro Hierarchical Model
# ---------------------
def hierarchical_model(age, team, position, pot=None):
    # Priors for team intercepts and slopes
    team_intercept_mean = pyro.sample("team_intercept_mean", dist.Normal(70., 10.))
    team_intercept_sd = pyro.sample("team_intercept_sd", dist.HalfNormal(10.))
    team_intercept = pyro.sample("team_intercept", dist.Normal(team_intercept_mean, team_intercept_sd).expand([n_teams]))
    
    team_slope_mean = pyro.sample("team_slope_mean", dist.Normal(0., 2.))
    team_slope_sd = pyro.sample("team_slope_sd", dist.HalfNormal(2.))
    team_slope = pyro.sample("team_slope", dist.Normal(team_slope_mean, team_slope_sd).expand([n_teams]))
    
    # Position effects
    pos_effects = pyro.sample("pos_effects", dist.Normal(0., 3.).expand([n_positions]))
    
    sigma_player = pyro.sample("sigma_player", dist.HalfNormal(5.))
    
    # Expected POT
    mu = team_intercept[team] + team_slope[team] * age + pos_effects[position]
    
    with pyro.plate("players", len(age)):
        pyro.sample("obs", dist.Normal(mu, sigma_player), obs=pot)

# ---------------------
# --- MCMC Settings (Hardcoded)
# ---------------------
NUM_SAMPLES = 1000
NUM_WARMUP = 400

fast_preview = st.checkbox("Fast Preview Mode (Subsample for quick results)")

# Subsample if fast preview is checked
if fast_preview:
    subsample_idx = np.random.choice(n_players, min(500, n_players), replace=False)
    age_sub = age[subsample_idx]
    team_sub = team[subsample_idx]
    position_sub = position[subsample_idx]
    pot_sub = pot[subsample_idx]
    df_sub = df.iloc[subsample_idx].reset_index(drop=True)
else:
    age_sub = age
    team_sub = team
    position_sub = position
    pot_sub = pot
    df_sub = df.copy()

# ---------------------
# --- Run Bayesian Model
# ---------------------
if st.button("Run Bayesian Model"):
    st.info("Running MCMC... this may take a minute.")
    nuts_kernel = NUTS(hierarchical_model)
    mcmc = MCMC(nuts_kernel, num_samples=NUM_SAMPLES, warmup_steps=NUM_WARMUP)
    mcmc.run(age_sub, team_sub, position_sub, pot_sub)
    st.session_state.mcmc = mcmc
    st.success("MCMC finished!")

# ---------------------
# --- Posterior Analysis
# ---------------------
if 'mcmc' in st.session_state:
    mcmc = st.session_state.mcmc
    posterior_samples = mcmc.get_samples()
    
    # Compute posterior growth headroom
    posterior_mu = posterior_samples['team_intercept'][:, team_sub] + \
                   posterior_samples['team_slope'][:, team_sub] * age_sub.unsqueeze(0) + \
                   posterior_samples['pos_effects'][:, position_sub]
    
    growth_headroom = posterior_mu - pot_sub.unsqueeze(0)
    
    df_headroom = df_sub.copy()
    df_headroom['Expected Growth'] = growth_headroom.mean(0).numpy()
    df_headroom['Growth 5-95%'] = list(zip(
        np.percentile(growth_headroom.numpy(), 5, axis=0),
        np.percentile(growth_headroom.numpy(), 95, axis=0)
    ))
    
    # --- Player Table ---
    st.subheader("Player Growth Headroom")
    st.dataframe(df_headroom[['Name', 'Team', 'OVR', 'POT', 'Expected Growth', 'Growth 5-95%']])
    
    # --- Top 10 Players ---
    st.subheader("Top 10 Players by Expected Growth")
    st.dataframe(df_headroom.sort_values("Expected Growth", ascending=False).head(10))
    
    # --- Growth by Position ---
    st.subheader("Average Growth by Position")
    pos_growth = df_headroom.groupby("Preferred Position")["Expected Growth"].mean()
    sns.barplot(x=pos_growth.index, y=pos_growth.values)
    plt.xticks(rotation=45)
    plt.ylabel("Expected Growth")
    st.pyplot(plt.gcf())
    
    # --- Growth by Team ---
    st.subheader("Average Growth by Team")
    team_growth = df_headroom.groupby("Team")["Expected Growth"].mean()
    sns.barplot(x=team_growth.index, y=team_growth.values)
    plt.xticks(rotation=45)
    plt.ylabel("Expected Growth")
    st.pyplot(plt.gcf())
    
    # --- Heatmap by Team and Position ---
    st.subheader("Growth Heatmap: Team vs Position")
    heatmap_data = df_headroom.pivot_table(index='Team', columns='Preferred Position', values='Expected Growth')
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="viridis")
    st.pyplot(plt.gcf())
    
    # --- Posterior Distribution per Player ---
    st.subheader("Player Posterior Distribution")
    player_idx = st.selectbox("Select player to visualize", range(len(df_headroom)), format_func=lambda x: df_headroom.iloc[x]['Name'])
    sns.histplot(growth_headroom[:, player_idx].numpy(), kde=True, color='skyblue')
    plt.xlabel("Growth Headroom (POT − OVR)")
    plt.ylabel("Posterior Density")
    st.pyplot(plt.gcf())
    
    # --- What-If Scenario ---
    st.subheader("What-If Simulator")
    new_age = st.slider("Simulate Age for Selected Player", 16, 40, int(df_headroom.iloc[player_idx]['Age']))
    simulated_mu = posterior_samples['team_intercept'][:, team_sub[player_idx]] + \
                   posterior_samples['team_slope'][:, team_sub[player_idx]] * new_age + \
                   posterior_samples['pos_effects'][:, position_sub[player_idx]]
    simulated_growth = simulated_mu - pot_sub[player_idx]
    st.write(f"Expected Growth at age {new_age}: {simulated_growth.mean().item():.2f} ± {np.percentile(simulated_growth.numpy(), [5,95])}")
    
    # --- Download Predictions ---
    csv = df_headroom.to_csv(index=False).encode()
    st.download_button("Download Growth Predictions CSV", csv, "growth_predictions.csv")