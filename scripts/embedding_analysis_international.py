# %% [markdown]
### Embedding Similarity Analysis
#
# This notebook analyzes similarities between the embeddings of the ethical dilemma dataset in both Portuguese, German, Spanish, and French in three main ways:
# 1. **Scenario-wise analysis**: Compare different actors' responses to the same scenario (ethical dilemma)
# 2. **Actor-wise analysis**: Compare a same actor's responses to different scenarios
# 3. **Reason-wise analysis**: Compare different reasoning versions for a same actor in the same scenario
#
# The actors considered for this analysis are:
# - **LLM Models**: GPT-3.5, GPT-4, Claude Haiku, Gemini 2, Gemma 7B, Mistral 7B, and Llama 2.
# - **Human Redditors**: The author of the top comment of each scenario submission.
# %% Import libraries
import pandas as pd
from embedding_utils import (
    load_embeddings,
    load_or_compute_similarities,
    identify_actors_and_reasons,
    plot_row_similarity_distribution,
    summarize_row_characteristics,
    save_analysis_results,
    display_edge_llm_human_similarities,
    analyze_column_similarities,
    plot_column_similarity_comparison,
    summarize_column_characteristics,
    display_edge_scenario_similarities,
    plot_reason_similarity_comparison,
    summarize_reason_characteristics,
    cross_analyze_actor_similarity,
)

# %% Load and explore the embeddings data
embeddings_dict_br = load_embeddings("../data/embeddings_br.csv")
embeddings_dict_de = load_embeddings("../data/embeddings_de.csv")
embeddings_dict_es = load_embeddings("../data/embeddings_es.csv")
embeddings_dict_fr = load_embeddings("../data/embeddings_fr.csv")

# %% Identify actors and reason types
actors_br, reason_types_br = identify_actors_and_reasons(embeddings_dict_br)
actors_de, reason_types_de = identify_actors_and_reasons(embeddings_dict_de)
actors_es, reason_types_es = identify_actors_and_reasons(embeddings_dict_es)
actors_fr, reason_types_fr = identify_actors_and_reasons(embeddings_dict_fr)

# %% [markdown]
# ## 1. Scenario-wise Analysis
#
# This analysis compares how human redditors and LLM models respond to the same ethical dilemma.
# For each scenario (row), embedding similarities are calculated between all pairs of actors.

# %% Scenario-wise similarity analysis
row_similarities_br = load_or_compute_similarities(
    "br", embeddings_dict_br, actors_br, reason_types_br, "row"
)
row_similarities_de = load_or_compute_similarities(
    "de", embeddings_dict_de, actors_de, reason_types_de, "row"
)
row_similarities_es = load_or_compute_similarities(
    "es", embeddings_dict_es, actors_es, reason_types_es, "row"
)
row_similarities_fr = load_or_compute_similarities(
    "fr", embeddings_dict_fr, actors_fr, reason_types_fr, "row"
)

# %% Visualize scenario-wise similarities
plot_row_similarity_distribution(row_similarities_br, "Portuguese")
plot_row_similarity_distribution(row_similarities_de, "German")
plot_row_similarity_distribution(row_similarities_es, "Spanish")
plot_row_similarity_distribution(row_similarities_fr, "French")

# %% Statistical summary of scenario-wise similarities
row_summary_df_br = summarize_row_characteristics(row_similarities_br, "Portuguese")
row_summary_df_de = summarize_row_characteristics(row_similarities_de, "German")
row_summary_df_es = summarize_row_characteristics(row_similarities_es, "Spanish")
row_summary_df_fr = summarize_row_characteristics(row_similarities_fr, "French")

# %% Save scenario-wise analysis results
save_analysis_results("br", row_summary_df_br, "scenario_wise")
save_analysis_results("de", row_summary_df_de, "scenario_wise")
save_analysis_results("es", row_summary_df_es, "scenario_wise")
save_analysis_results("fr", row_summary_df_fr, "scenario_wise")

# %% Display LLM-Human similarity edge cases
df_cleaned_br = pd.read_csv("../data/ethical_dilemmas_cleaned_br.csv")
df_cleaned_de = pd.read_csv("../data/ethical_dilemmas_cleaned_de.csv")
df_cleaned_es = pd.read_csv("../data/ethical_dilemmas_cleaned_es.csv")
df_cleaned_fr = pd.read_csv("../data/ethical_dilemmas_cleaned_fr.csv")

display_edge_llm_human_similarities(row_similarities_br, df_cleaned_br, "Portuguese")
display_edge_llm_human_similarities(row_similarities_de, df_cleaned_de, "German")
display_edge_llm_human_similarities(row_similarities_es, df_cleaned_es, "Spanish")
display_edge_llm_human_similarities(row_similarities_fr, df_cleaned_fr, "French")

# %% [markdown]
# ## 2. Actor-wise Analysis
#
# This analysis compares how a same actor responds to different ethical dilemmas.
# For each actor, we calculate the similarity between all pairs of scenarios.

# %% Actor-wise similarity analysis
column_similarities_br = analyze_column_similarities(
    embeddings_dict_br, actors_br, reason_types_br
)
column_similarities_de = analyze_column_similarities(
    embeddings_dict_de, actors_de, reason_types_de
)
column_similarities_es = analyze_column_similarities(
    embeddings_dict_es, actors_es, reason_types_es
)
column_similarities_fr = analyze_column_similarities(
    embeddings_dict_fr, actors_fr, reason_types_fr
)

# %% Visualize actor-wise similarities
plot_column_similarity_comparison(column_similarities_br, "Portuguese")
plot_column_similarity_comparison(column_similarities_de, "German")
plot_column_similarity_comparison(column_similarities_es, "Spanish")
plot_column_similarity_comparison(column_similarities_fr, "French")

# %% Statistical summary of actor-wise differences
column_summary_df_br = summarize_column_characteristics(
    column_similarities_br, "Portuguese"
)
column_summary_df_de = summarize_column_characteristics(
    column_similarities_de, "German"
)
column_summary_df_es = summarize_column_characteristics(
    column_similarities_es, "Spanish"
)
column_summary_df_fr = summarize_column_characteristics(
    column_similarities_fr, "French"
)

# %% Save column-wise analysis results
save_analysis_results("br", column_summary_df_br, "actor_wise")
save_analysis_results("de", column_summary_df_de, "actor_wise")
save_analysis_results("es", column_summary_df_es, "actor_wise")
save_analysis_results("fr", column_summary_df_fr, "actor_wise")

# %% Display scenario similarity edge cases
display_edge_scenario_similarities(
    embeddings_dict_br, actors_br, reason_types_br, df_cleaned_br, "Portuguese"
)
display_edge_scenario_similarities(
    embeddings_dict_de, actors_de, reason_types_de, df_cleaned_de, "German"
)
display_edge_scenario_similarities(
    embeddings_dict_es, actors_es, reason_types_es, df_cleaned_es, "Spanish"
)
display_edge_scenario_similarities(
    embeddings_dict_fr, actors_fr, reason_types_fr, df_cleaned_fr, "French"
)

# %% [markdown]
# ## 3. Reason-wise Analysis
#
# This analysis compares how consistent each actor's reasonings are when answering the same ethical dilemma.

# %% Reason-wise similarity analysis
reason_similarities_br = load_or_compute_similarities(
    "br", embeddings_dict_br, actors_br, reason_types_br, "reason"
)
reason_similarities_de = load_or_compute_similarities(
    "de", embeddings_dict_de, actors_de, reason_types_de, "reason"
)
reason_similarities_es = load_or_compute_similarities(
    "es", embeddings_dict_es, actors_es, reason_types_es, "reason"
)
reason_similarities_fr = load_or_compute_similarities(
    "fr", embeddings_dict_fr, actors_fr, reason_types_fr, "reason"
)

# %% Visualize reason-wise similarities
plot_reason_similarity_comparison(reason_similarities_br, "Portuguese")
plot_reason_similarity_comparison(reason_similarities_de, "German")
plot_reason_similarity_comparison(reason_similarities_es, "Spanish")
plot_reason_similarity_comparison(reason_similarities_fr, "French")

# %% Statistical summary of reason-wise characteristics
reason_summary_df_br = summarize_reason_characteristics(
    reason_similarities_br, "Portuguese"
)
reason_summary_df_de = summarize_reason_characteristics(
    reason_similarities_de, "German"
)
reason_summary_df_es = summarize_reason_characteristics(
    reason_similarities_es, "Spanish"
)
reason_summary_df_fr = summarize_reason_characteristics(
    reason_similarities_fr, "French"
)

# %% Save reason-wise analysis results
save_analysis_results("br", reason_summary_df_br, "reason_wise")
save_analysis_results("de", reason_summary_df_de, "reason_wise")
save_analysis_results("es", reason_summary_df_es, "reason_wise")
save_analysis_results("fr", reason_summary_df_fr, "reason_wise")

# %% Cross-analysis: Intra-Actor similarity vs. inter-Actor similarity
cross_analysis_df_br = cross_analyze_actor_similarity(
    row_similarities_br, column_similarities_br, reason_similarities_br, "Portuguese"
)
cross_analysis_df_de = cross_analyze_actor_similarity(
    row_similarities_de, column_similarities_de, reason_similarities_de, "German"
)
cross_analysis_df_es = cross_analyze_actor_similarity(
    row_similarities_es, column_similarities_es, reason_similarities_es, "Spanish"
)
cross_analysis_df_fr = cross_analyze_actor_similarity(
    row_similarities_fr, column_similarities_fr, reason_similarities_fr, "French"
)

# %% Save cross-analysis results
save_analysis_results("br", cross_analysis_df_br, "cross_analysis")
save_analysis_results("de", cross_analysis_df_de, "cross_analysis")
save_analysis_results("es", cross_analysis_df_es, "cross_analysis")
save_analysis_results("fr", cross_analysis_df_fr, "cross_analysis")

# %% [markdown]
# ## Summary of Findings
#
# This analysis examines embedding similarities across all available reasoning types for 7 LLM actors and human responses on ethical scenarios in German, revealing distinct behavioral patterns in ethical decision-making.
#
# ### Key Findings:
#
# 1. **Inter-Actor Agreement** (LLM-to-LLM similarity range: 9.5% - 55.9%):
#    - **GPT-3.5** and **GPT-4** show highest agreement with other LLMs (55.9% similarity)
#    - **Gemma** shows lowest agreement with other LLMs (9.5% with Llama, 17.4% with Mistral)
#    - LLM actors generally show low to moderate consensus (mean ~25%) across ethical scenarios
#    - **Human responses** show much lower agreement with LLMs (3.7% - 31.0% similarity)
#    - Human-LLM alignment is consistently lower than inter-LLM agreement, indicating distinct reasoning patterns
#
# 2. **Intra-Actor Agreement** (Range: 27.1% - 44.8%):
#    - **Gemma** shows highest intra-actor agreement (44.8% ± 9.4%) - most predictable across scenarios
#    - **Mistral** responses are least internally consistent (27.1% ± 12.7%) - most context-dependent
#    - **Human** shows moderate intra-actor agreement (29.9% ± 9.0%) - balanced variability
#    - **GPT-4** shows moderate intra-actor agreement (42.7% ± 8.5%) - consistent across scenarios
#    - Internal agreement range of 17.7% indicates moderate diversity in actor response patterns
#
# 3. **Reason-wise Consistency** (Range: 29.2% - 100%):
#    - **Human** shows perfect reasoning consistency (100%) - single reasoning approach per scenario
#    - **Llama** shows highest LLM reasoning coherence (72.2% ± 15.3%) - most consistent across reasoning approaches
#    - **Mistral** shows lowest reasoning coherence (29.2% ± 19.1%) - most variable across reasoning approaches
#    - Mean LLM reason-wise consistency (59.1% ± 15.8%) much higher than intra-actor consistency
#    - This indicates actors are more consistent within reasoning types than across different scenarios
#
# 4. **Three-Dimensional Actor Profiles**:
#    - **Claude**: Moderate inter-actor agreement (36.8%), moderate intra-actor agreement (44.1%), moderate reasoning consistency (66.0%)
#    - **Gemma**: Low inter-actor agreement (24.4%), highest intra-actor agreement (44.8%), low reasoning consistency (38.3%)
#    - **GPT-4**: Moderate inter-actor agreement (37.9%), moderate intra-actor agreement (42.7%), moderate reasoning consistency (63.8%)
#    - **Llama**: Low inter-actor agreement (14.5%), moderate intra-actor agreement (42.2%), highest reasoning consistency (72.2%)
#    - **Human**: Low inter-actor agreement (24.8%), moderate intra-actor agreement (29.9%), perfect reasoning consistency (100%)
#
# 5. **Human-LLM Alignment** (Range: 3.7% - 31.0%):
#    - **Claude** shows highest human alignment (31.0% ± 9.9%) - most human-like reasoning
#    - **Llama** shows lowest human alignment (3.7% ± 7.6%) - least human-like reasoning
#    - **GPT-3.5** shows moderate human alignment (31.7% ± 9.5%) - balanced human similarity
#    - Range of 27.3% indicates high variation in human alignment across LLMs
#    - All LLMs show substantial variability (7.6% - 9.9% std), suggesting context-dependent human alignment
#
# ### Practical Implications:
#
# - **Most Predictable Ethics**: Gemma (highest internal consistency across scenarios)
# - **Most Coherent Reasoning**: Llama (most consistent across different reasoning approaches)
# - **Most Diverse Perspectives**: Mistral and Human (high variability in responses)
# - **Most Human-Like Reasoning**: Claude and GPT-3.5 (highest human alignment at ~31%)
# - **Best Overall Balance**: GPT-4 (reliable consensus + coherent reasoning + moderate diversity)
# - **Best for Human Collaboration**: Claude and GPT-3.5 (highest human alignment scores)
# - **Most Independent Reasoning**: Llama and Mistral (lowest inter-actor alignment, most unique perspectives)
# - **Most Context-Dependent**: Mistral responses (lowest internal consistency, suggesting high situational awareness)
