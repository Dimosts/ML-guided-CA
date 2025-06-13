import cpmpy as cp
from ..answering_queries.constraint_oracle import ConstraintOracle
from ..problem_instance import ProblemInstance, absvar
from cpmpy.transformations.normalize import toplevel_list

def construct_nqueens(N=50):

    # Create a dictionary with the parameters
    parameters = {"NQueens": N}


    # Variables (one per row)
    queens = cp.intvar(1,N, shape=N, name="queens")

    # Constraints on columns and left/right diagonal
    model = cp.Model([
        cp.AllDifferent(queens).decompose(),
        cp.AllDifferent([queens[i] + i for i in range(N)]).decompose(),
        cp.AllDifferent([queens[i] - i for i in range(N)]).decompose(),
    ])
    
    C_T = list(set(toplevel_list(model.constraints)))

    # Create the language:
    AV = absvar(2)  # create abstract vars - as many as maximum arity

    # create abstract relations using the abstract vars
    lang = [AV[0] == AV[1], AV[0] != AV[1], AV[0] < AV[1], AV[0] > AV[1], AV[0] >= AV[1], AV[0] <= AV[1]]

    instance = ProblemInstance(variables=queens, params=parameters, language=lang, name="nqueens")

    oracle = ConstraintOracle(C_T)
    print("Constraints: ", len(C_T))
    input("Press Enter to continue...")
    return instance, oracle