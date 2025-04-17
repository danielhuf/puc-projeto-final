# %% [markdown]
# # Reddit r/AmItheAsshole Data Scraper
#
# This notebook fetches data from the r/AmItheAsshole subreddit using the PRAW API.

# %% [markdown]
# ## Import Libraries

# %%
import praw
from dotenv import load_dotenv
import os

load_dotenv()

# %% [markdown]
# ## Initialize Reddit API Client
#
# Using credentials from .env file:
# - `REDDIT_CLIENT_ID`: Your Reddit API client ID
# - `REDDIT_CLIENT_SECRET`: Your Reddit API client secret
# - `REDDIT_USERNAME`: Your Reddit username

# %%
# Initialize the Reddit API client
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=f"script:aita_scraper:v1.0 (by /u/{os.getenv('REDDIT_USERNAME')})",
)

# %% [markdown]
# ## Define Functions
