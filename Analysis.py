"""External Imports (Libraries and APIs)"""


"""Local Imports"""
from Config import Config




def check_convergence(F: float, cfg: Config, l: float) -> bool:
    optimal_F_lb = l * cfg.optimal_fitness

    if optimal_F_lb <= F <= cfg.optimal_fitness:

        return True


    return False