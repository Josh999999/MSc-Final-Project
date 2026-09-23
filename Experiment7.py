from Evolution_Experiment import standard_environment, run_environment




FIGURES_OUTPUT = "Experiment7"

N_GENERATIONS = 20_000
SWITCH_EVERY  = 1_500
RECORD_EVERY  = 200
SEEDS         = 5




if __name__ == "__main__":
    targets, ideal = standard_environment()
    run_environment("standard", targets, ideal, FIGURES_OUTPUT,
                    N_GENERATIONS, SWITCH_EVERY, RECORD_EVERY, SEEDS)
