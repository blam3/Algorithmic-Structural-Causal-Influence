# Algorithmic-Structural-Causal-Influence
Metrics to quantify and assess algorithmic fairness in dynamic and longitudinal settings, particularly when there are feedback loops that perpetuate inequality.

File	          Contents
config.py	      All constants and hyperparameters (N, T, TAU, G, VAR matrix, harm weights, etc.)
metrics.py	    Harm functions, dSCI, eSCI path, DP/EO gaps, Liu curve, CVaR, tail analysis
simulation.py	  initialize_population, make_decisions, var_feedback_transition, update_structural_attribute, counterfactual_intervention, run_simulation
monte_carlo.py	run_monte_carlo — runs N independent seeds and aggregates all metrics
plots.py	      All plot_* functions and print_results_table
main.py	        Entry point — python main.py reproduces the full analysis
