import random

import cpmpy as cp
import numpy as np
from cpmpy.expressions.utils import all_pairs
from cpmpy.transformations.normalize import toplevel_list
from ..answering_queries.constraint_oracle import ConstraintOracle
from ..problem_instance import ProblemInstance, absvar


def construct_job_shop_scheduling_problem(n_jobs=10, machines=2, horizon=50, seed=0):
    """
    :return: a ProblemInstance object, along with a constraint-based oracle
    """
    random.seed(seed)
    max_time = horizon // n_jobs

    duration = [[0] * machines for i in range(0, n_jobs)]
    for i in range(0, n_jobs):
        for j in range(0, machines):
            duration[i][j] = random.randint(1, max_time)

    task_to_mach = [list(range(0, machines)) for i in range(0, n_jobs)]

    for i in range(0, n_jobs):
        random.shuffle(task_to_mach[i])

    precedence = [[(i, j) for j in task_to_mach[i]] for i in range(0, n_jobs)]

    # convert to numpy
    task_to_mach = np.array(task_to_mach)
    duration = np.array(duration)
    precedence = np.array(precedence)

    machines = set(task_to_mach.flatten().tolist())

    # decision variables
    start = cp.intvar(1, horizon, shape=task_to_mach.shape, name="start")
    end = cp.intvar(1, horizon, shape=task_to_mach.shape, name="end")

    model = cp.Model()

    grid = cp.cpm_array(np.expand_dims(np.concatenate([start.flatten(), end.flatten()]), 0))

    # precedence constraints
    for chain in precedence:
        for (j1, t1), (j2, t2) in zip(chain[:-1], chain[1:]):
            model += end[j1, t1] <= start[j2, t2]

    # duration constraints
    model += (start + duration == end)

    C_T = list(set(toplevel_list(model.constraints)))

    print("Duration: ", duration)
    max_duration = max(max(duration[i]) for i in range(n_jobs))
    print("Max duration: ", max_duration)

    # Create the language:
    AV = absvar(2)  # create abstract vars - as many as maximum arity

    # create abstract relations using the abstract vars
    lang = [AV[0] == AV[1], AV[0] != AV[1], AV[0] < AV[1], AV[0] > AV[1], AV[0] >= AV[1], AV[0] <= AV[1]] + \
           [AV[0] + i == AV[1] for i in range(1, max_duration + 1)] + \
           [AV[1] + i == AV[0] for i in range(1, max_duration + 1)]

    instance = ProblemInstance(variables=grid, language=lang,
                               name=f"job_shop_jobs{n_jobs}_machines{machines}_horizon{horizon}_seed{seed}")

    #instance.construct_bias()
    oracle = ConstraintOracle(C_T)

    return instance, oracle
