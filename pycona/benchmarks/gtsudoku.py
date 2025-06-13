import cpmpy as cp
from ..answering_queries.constraint_oracle import ConstraintOracle
from ..problem_instance import ProblemInstance, absvar
from cpmpy.transformations.normalize import toplevel_list

def construct_gtsudoku(block_size_row=3, block_size_col=3, grid_size=9):
    """
    Constructs a Greater Than Sudoku problem instance.
    
    :param block_size_row: Number of rows in each block (default 3)
    :param block_size_col: Number of columns in each block (default 3)
    :param grid_size: Size of the grid (default 9)
    :return: a ProblemInstance object, along with a constraint-based oracle
    """
    # Create a dictionary with the parameters
    parameters = {
        "block_size_row": block_size_row,
        "block_size_col": block_size_col,
        "grid_size": grid_size
    }

    # Variables
    grid = cp.intvar(1, grid_size, shape=(grid_size, grid_size), name="grid")

    # Add greater-than constraints
    model = cp.Model(
        grid[0,2] < grid[1,2],
        grid[0,4] > grid[1,4],
        grid[0,5] < grid[1,5],
        grid[0,8] < grid[1,8],
        
        grid[1,1] > grid[1,2],
        grid[1,0] < grid[2,0],
        grid[1,6] > grid[1,7],
        grid[1,8] < grid[2,8],

        grid[2,4] < grid[2,5],

        grid[3,0] < grid[3,1],
        grid[3,6] < grid[3,7],
        grid[3,7] < grid[3,8],
        grid[3,2] < grid[4,2],
        grid[3,7] > grid[4,7],
        grid[3,8] < grid[4,8],

        grid[4,1] > grid[5,1],
        grid[4,2] < grid[5,2],
        grid[4,4] < grid[5,4],
        grid[4,5] < grid[5,5],
        grid[4,8] < grid[5,8],


        grid[6,5] < grid[6,6],

        grid[7,1] > grid[8,1],
        grid[7,3] > grid[7,4],
        grid[7,5] > grid[8,5],
        grid[7,6] > grid[8,6],

        grid[8,2] > grid[8,3],
        )

    # Standard Sudoku constraints
    # Constraints on rows and columns
    for row in grid:
        model += cp.AllDifferent(row).decompose()

    for col in grid.T:  # numpy's Transpose
        model += cp.AllDifferent(col).decompose()

    # Constraints on blocks
    for i in range(0, grid_size, block_size_row):
        for j in range(0, grid_size, block_size_col):
            model += cp.AllDifferent(grid[i:i + block_size_row, j:j + block_size_col]).decompose()

    C_T = list(set(toplevel_list(model.constraints)))

    # Create the language:
    AV = absvar(2)  # create abstract vars - as many as maximum arity

    # create abstract relations using the abstract vars
    lang = [
        AV[0] == AV[1],
        AV[0] != AV[1],
        AV[0] < AV[1],
        AV[0] > AV[1],
        AV[0] >= AV[1],
        AV[0] <= AV[1]
    ]

    instance = ProblemInstance(variables=grid, params=parameters, language=lang, name="gtsudoku")
    oracle = ConstraintOracle(C_T)

    return instance, oracle 