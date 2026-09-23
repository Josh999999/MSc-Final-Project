from Evolution_Experiment import modular_environment, run_environment




FIGURES_OUTPUT = "Experiment8"

N_GENERATIONS = 20_000
SWITCH_EVERY  = 1_500
RECORD_EVERY  = 200
SEEDS         = 5

# Environment: N genes, K modules (2^K targets), fixed module patterns.
MOD_N, MOD_K, MOD_ENV_SEED = 16, 4, 7




if __name__ == "__main__":
    targets, ideal = modular_environment(MOD_N, MOD_K, MOD_ENV_SEED)
    run_environment("modular", targets, ideal, FIGURES_OUTPUT,
                    N_GENERATIONS, SWITCH_EVERY, RECORD_EVERY, SEEDS)
