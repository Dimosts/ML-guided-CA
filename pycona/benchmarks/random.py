import cpmpy as cp
from ..answering_queries.constraint_oracle import ConstraintOracle
from ..problem_instance import ProblemInstance, absvar
from cpmpy.transformations.normalize import toplevel_list
from ..utils import get_scope
from pycona.utils import get_variables_from_constraints
import pickle


def construct_random():
    with open("model.pickle", "rb") as f:
        model = pickle.load(f)

    C_T = list(set(toplevel_list(model.constraints)))

    print("len(C_T): ", len(C_T))
    for c in C_T:
        print(c)
    input()

    # Create the language:
    AV = absvar(2)  # create abstract vars - as many as maximum arity

    # create abstract relations using the abstract vars
    lang = [AV[0] != AV[1], AV[0] < AV[1], AV[0] > AV[1], AV[0] >= AV[1], AV[0] <= AV[1]]

    instance = ProblemInstance(variables=get_variables_from_constraints(C_T), language=lang, name="random")

    oracle = ConstraintOracle(C_T)

    return instance, oracle

def construct_random495():
    # Variables
    grid = cp.intvar(1, 5, shape=(1, 100), name="grid")

    # 495 constraints randomly generated
    scopes = [{grid[0, 68], grid[0, 97]}, {grid[0, 67], grid[0, 99]}, {grid[0, 84], grid[0, 94]},
              {grid[0, 15], grid[0, 92]}, {grid[0, 2], grid[0, 40]}, {grid[0, 27], grid[0, 39]},
              {grid[0, 53], grid[0, 61]}, {grid[0, 29], grid[0, 85]}, {grid[0, 14], grid[0, 33]},
              {grid[0, 31], grid[0, 36]}, {grid[0, 22], grid[0, 69]}, {grid[0, 21], grid[0, 25]},
              {grid[0, 36], grid[0, 76]}, {grid[0, 1], grid[0, 98]}, {grid[0, 52], grid[0, 91]},
              {grid[0, 44], grid[0, 97]}, {grid[0, 43], grid[0, 48]}, {grid[0, 12], grid[0, 32]},
              {grid[0, 1], grid[0, 13]}, {grid[0, 85], grid[0, 96]}, {grid[0, 15], grid[0, 40]},
              {grid[0, 27], grid[0, 85]}, {grid[0, 45], grid[0, 51]}, {grid[0, 58], grid[0, 90]},
              {grid[0, 23], grid[0, 57]}, {grid[0, 23], grid[0, 78]}, {grid[0, 88], grid[0, 94]},
              {grid[0, 73], grid[0, 81]}, {grid[0, 9], grid[0, 34]}, {grid[0, 60], grid[0, 87]},
              {grid[0, 64], grid[0, 76]}, {grid[0, 14], grid[0, 63]}, {grid[0, 15], grid[0, 84]},
              {grid[0, 2], grid[0, 37]}, {grid[0, 7], grid[0, 52]}, {grid[0, 11], grid[0, 69]},
              {grid[0, 0], grid[0, 71]}, {grid[0, 21], grid[0, 27]}, {grid[0, 38], grid[0, 89]},
              {grid[0, 40], grid[0, 91]}, {grid[0, 12], grid[0, 99]}, {grid[0, 85], grid[0, 87]},
              {grid[0, 5], grid[0, 29]}, {grid[0, 3], grid[0, 64]}, {grid[0, 42], grid[0, 94]},
              {grid[0, 21], grid[0, 34]}, {grid[0, 37], grid[0, 57]}, {grid[0, 59], grid[0, 81]},
              {grid[0, 58], grid[0, 77]}, {grid[0, 24], grid[0, 66]}, {grid[0, 2], grid[0, 17]},
              {grid[0, 3], grid[0, 20]}, {grid[0, 76], grid[0, 96]}, {grid[0, 54], grid[0, 85]},
              {grid[0, 51], grid[0, 68]}, {grid[0, 8], grid[0, 94]}, {grid[0, 10], grid[0, 61]},
              {grid[0, 2], grid[0, 21]}, {grid[0, 24], grid[0, 42]}, {grid[0, 8], grid[0, 48]},
              {grid[0, 45], grid[0, 94]}, {grid[0, 7], grid[0, 48]}, {grid[0, 37], grid[0, 42]},
              {grid[0, 34], grid[0, 72]}, {grid[0, 20], grid[0, 36]}, {grid[0, 97], grid[0, 98]},
              {grid[0, 42], grid[0, 55]}, {grid[0, 91], grid[0, 99]}, {grid[0, 9], grid[0, 31]},
              {grid[0, 28], grid[0, 95]}, {grid[0, 4], grid[0, 45]}, {grid[0, 22], grid[0, 88]},
              {grid[0, 15], grid[0, 25]}, {grid[0, 17], grid[0, 22]}, {grid[0, 49], grid[0, 51]},
              {grid[0, 26], grid[0, 35]}, {grid[0, 26], grid[0, 42]}, {grid[0, 72], grid[0, 96]},
              {grid[0, 42], grid[0, 91]}, {grid[0, 72], grid[0, 81]}, {grid[0, 36], grid[0, 85]},
              {grid[0, 60], grid[0, 91]}, {grid[0, 28], grid[0, 52]}, {grid[0, 40], grid[0, 70]},
              {grid[0, 57], grid[0, 75]}, {grid[0, 27], grid[0, 87]}, {grid[0, 73], grid[0, 75]},
              {grid[0, 73], grid[0, 95]}, {grid[0, 16], grid[0, 70]}, {grid[0, 94], grid[0, 95]},
              {grid[0, 46], grid[0, 80]}, {grid[0, 73], grid[0, 94]}, {grid[0, 30], grid[0, 91]},
              {grid[0, 25], grid[0, 53]}, {grid[0, 24], grid[0, 75]}, {grid[0, 30], grid[0, 56]},
              {grid[0, 63], grid[0, 64]}, {grid[0, 53], grid[0, 56]}, {grid[0, 44], grid[0, 49]},
              {grid[0, 85], grid[0, 90]}, {grid[0, 36], grid[0, 73]}, {grid[0, 63], grid[0, 95]},
              {grid[0, 9], grid[0, 47]}, {grid[0, 2], grid[0, 5]}, {grid[0, 75], grid[0, 91]},
              {grid[0, 72], grid[0, 82]}, {grid[0, 8], grid[0, 42]}, {grid[0, 3], grid[0, 75]},
              {grid[0, 11], grid[0, 79]}, {grid[0, 25], grid[0, 26]}, {grid[0, 66], grid[0, 74]},
              {grid[0, 14], grid[0, 90]}, {grid[0, 16], grid[0, 26]}, {grid[0, 26], grid[0, 84]},
              {grid[0, 41], grid[0, 84]}, {grid[0, 18], grid[0, 32]}, {grid[0, 7], grid[0, 82]},
              {grid[0, 0], grid[0, 35]}, {grid[0, 3], grid[0, 60]}, {grid[0, 27], grid[0, 90]},
              {grid[0, 64], grid[0, 78]}, {grid[0, 50], grid[0, 93]}, {grid[0, 65], grid[0, 74]},
              {grid[0, 66], grid[0, 99]}, {grid[0, 50], grid[0, 68]}, {grid[0, 34], grid[0, 76]},
              {grid[0, 2], grid[0, 46]}, {grid[0, 6], grid[0, 44]}, {grid[0, 34], grid[0, 98]},
              {grid[0, 24], grid[0, 30]}, {grid[0, 15], grid[0, 51]}, {grid[0, 22], grid[0, 44]},
              {grid[0, 58], grid[0, 93]}, {grid[0, 66], grid[0, 77]}, {grid[0, 57], grid[0, 92]},
              {grid[0, 2], grid[0, 74]}, {grid[0, 36], grid[0, 62]}, {grid[0, 49], grid[0, 89]},
              {grid[0, 26], grid[0, 96]}, {grid[0, 36], grid[0, 64]}, {grid[0, 5], grid[0, 7]},
              {grid[0, 55], grid[0, 87]}, {grid[0, 60], grid[0, 76]}, {grid[0, 14], grid[0, 66]},
              {grid[0, 64], grid[0, 94]}, {grid[0, 25], grid[0, 51]}, {grid[0, 60], grid[0, 70]},
              {grid[0, 16], grid[0, 34]}, {grid[0, 29], grid[0, 94]}, {grid[0, 2], grid[0, 56]},
              {grid[0, 67], grid[0, 89]}, {grid[0, 17], grid[0, 89]}, {grid[0, 32], grid[0, 38]},
              {grid[0, 88], grid[0, 89]}, {grid[0, 29], grid[0, 48]}, {grid[0, 6], grid[0, 40]},
              {grid[0, 92], grid[0, 96]}, {grid[0, 45], grid[0, 74]}, {grid[0, 20], grid[0, 89]},
              {grid[0, 27], grid[0, 72]}, {grid[0, 18], grid[0, 62]}, {grid[0, 85], grid[0, 94]},
              {grid[0, 23], grid[0, 64]}, {grid[0, 39], grid[0, 49]}, {grid[0, 14], grid[0, 24]},
              {grid[0, 50], grid[0, 56]}, {grid[0, 13], grid[0, 38]}, {grid[0, 15], grid[0, 86]},
              {grid[0, 61], grid[0, 88]}, {grid[0, 28], grid[0, 79]}, {grid[0, 31], grid[0, 62]},
              {grid[0, 33], grid[0, 68]}, {grid[0, 5], grid[0, 85]}, {grid[0, 38], grid[0, 39]},
              {grid[0, 6], grid[0, 75]}, {grid[0, 1], grid[0, 33]}, {grid[0, 0], grid[0, 13]},
              {grid[0, 45], grid[0, 53]}, {grid[0, 48], grid[0, 94]}, {grid[0, 20], grid[0, 93]},
              {grid[0, 57], grid[0, 68]}, {grid[0, 49], grid[0, 75]}, {grid[0, 38], grid[0, 93]},
              {grid[0, 34], grid[0, 54]}, {grid[0, 72], grid[0, 89]}, {grid[0, 34], grid[0, 61]},
              {grid[0, 70], grid[0, 88]}, {grid[0, 78], grid[0, 82]}, {grid[0, 81], grid[0, 84]},
              {grid[0, 39], grid[0, 76]}, {grid[0, 17], grid[0, 50]}, {grid[0, 16], grid[0, 58]},
              {grid[0, 24], grid[0, 96]}, {grid[0, 28], grid[0, 44]}, {grid[0, 74], grid[0, 83]},
              {grid[0, 75], grid[0, 83]}, {grid[0, 18], grid[0, 72]}, {grid[0, 6], grid[0, 45]},
              {grid[0, 69], grid[0, 89]}, {grid[0, 1], grid[0, 73]}, {grid[0, 21], grid[0, 87]},
              {grid[0, 39], grid[0, 73]}, {grid[0, 65], grid[0, 66]}, {grid[0, 8], grid[0, 78]},
              {grid[0, 12], grid[0, 78]}, {grid[0, 48], grid[0, 64]}, {grid[0, 11], grid[0, 73]},
              {grid[0, 7], grid[0, 74]}, {grid[0, 43], grid[0, 75]}, {grid[0, 1], grid[0, 54]},
              {grid[0, 10], grid[0, 83]}, {grid[0, 22], grid[0, 99]}, {grid[0, 15], grid[0, 98]},
              {grid[0, 33], grid[0, 94]}, {grid[0, 41], grid[0, 71]}, {grid[0, 47], grid[0, 81]},
              {grid[0, 22], grid[0, 86]}, {grid[0, 18], grid[0, 27]}, {grid[0, 19], grid[0, 30]},
              {grid[0, 6], grid[0, 70]}, {grid[0, 54], grid[0, 77]}, {grid[0, 31], grid[0, 96]},
              {grid[0, 43], grid[0, 46]}, {grid[0, 48], grid[0, 68]}, {grid[0, 96], grid[0, 99]},
              {grid[0, 78], grid[0, 99]}, {grid[0, 93], grid[0, 98]}, {grid[0, 39], grid[0, 89]},
              {grid[0, 5], grid[0, 49]}, {grid[0, 2], grid[0, 95]}, {grid[0, 37], grid[0, 68]},
              {grid[0, 34], grid[0, 35]}, {grid[0, 1], grid[0, 15]}, {grid[0, 13], grid[0, 23]},
              {grid[0, 63], grid[0, 77]}, {grid[0, 62], grid[0, 82]}, {grid[0, 2], grid[0, 19]},
              {grid[0, 4], grid[0, 69]}, {grid[0, 30], grid[0, 41]}, {grid[0, 28], grid[0, 39]},
              {grid[0, 24], grid[0, 46]}, {grid[0, 1], grid[0, 25]}, {grid[0, 74], grid[0, 89]},
              {grid[0, 17], grid[0, 84]}, {grid[0, 0], grid[0, 65]}, {grid[0, 35], grid[0, 84]},
              {grid[0, 66], grid[0, 80]}, {grid[0, 14], grid[0, 88]}, {grid[0, 8], grid[0, 93]},
              {grid[0, 6], grid[0, 47]}, {grid[0, 42], grid[0, 64]}, {grid[0, 0], grid[0, 80]},
              {grid[0, 76], grid[0, 92]}, {grid[0, 25], grid[0, 33]}, {grid[0, 73], grid[0, 80]},
              {grid[0, 69], grid[0, 98]}, {grid[0, 17], grid[0, 74]}, {grid[0, 36], grid[0, 72]},
              {grid[0, 9], grid[0, 41]}, {grid[0, 33], grid[0, 82]}, {grid[0, 25], grid[0, 43]},
              {grid[0, 45], grid[0, 71]}, {grid[0, 17], grid[0, 48]}, {grid[0, 42], grid[0, 92]},
              {grid[0, 8], grid[0, 15]}, {grid[0, 11], grid[0, 91]}, {grid[0, 36], grid[0, 53]},
              {grid[0, 34], grid[0, 43]}, {grid[0, 44], grid[0, 68]}, {grid[0, 64], grid[0, 96]},
              {grid[0, 0], grid[0, 57]}, {grid[0, 25], grid[0, 28]}, {grid[0, 9], grid[0, 49]},
              {grid[0, 23], grid[0, 36]}, {grid[0, 1], grid[0, 68]}, {grid[0, 12], grid[0, 50]},
              {grid[0, 51], grid[0, 84]}, {grid[0, 0], grid[0, 91]}, {grid[0, 7], grid[0, 80]},
              {grid[0, 10], grid[0, 90]}, {grid[0, 11], grid[0, 53]}, {grid[0, 3], grid[0, 52]},
              {grid[0, 19], grid[0, 75]}, {grid[0, 27], grid[0, 56]}, {grid[0, 4], grid[0, 51]},
              {grid[0, 72], grid[0, 90]}, {grid[0, 40], grid[0, 82]}, {grid[0, 25], grid[0, 75]},
              {grid[0, 64], grid[0, 71]}, {grid[0, 8], grid[0, 80]}, {grid[0, 46], grid[0, 63]},
              {grid[0, 19], grid[0, 81]}, {grid[0, 80], grid[0, 84]}, {grid[0, 47], grid[0, 50]},
              {grid[0, 41], grid[0, 62]}, {grid[0, 61], grid[0, 93]}, {grid[0, 47], grid[0, 54]},
              {grid[0, 60], grid[0, 83]}, {grid[0, 78], grid[0, 93]}, {grid[0, 95], grid[0, 96]},
              {grid[0, 20], grid[0, 71]}, {grid[0, 48], grid[0, 82]}, {grid[0, 3], grid[0, 45]},
              {grid[0, 83], grid[0, 95]}, {grid[0, 10], grid[0, 22]}, {grid[0, 38], grid[0, 40]},
              {grid[0, 31], grid[0, 50]}, {grid[0, 32], grid[0, 82]}, {grid[0, 56], grid[0, 90]},
              {grid[0, 40], grid[0, 64]}, {grid[0, 46], grid[0, 95]}, {grid[0, 1], grid[0, 83]},
              {grid[0, 2], grid[0, 43]}, {grid[0, 18], grid[0, 28]}, {grid[0, 31], grid[0, 60]},
              {grid[0, 43], grid[0, 79]}, {grid[0, 17], grid[0, 68]}, {grid[0, 19], grid[0, 93]},
              {grid[0, 36], grid[0, 43]}, {grid[0, 13], grid[0, 67]}, {grid[0, 98], grid[0, 99]},
              {grid[0, 15], grid[0, 37]}, {grid[0, 0], grid[0, 25]}, {grid[0, 45], grid[0, 47]},
              {grid[0, 40], grid[0, 94]}, {grid[0, 61], grid[0, 97]}, {grid[0, 0], grid[0, 97]},
              {grid[0, 40], grid[0, 66]}, {grid[0, 90], grid[0, 94]}, {grid[0, 67], grid[0, 69]},
              {grid[0, 5], grid[0, 96]}, {grid[0, 5], grid[0, 17]}, {grid[0, 19], grid[0, 97]},
              {grid[0, 25], grid[0, 85]}, {grid[0, 19], grid[0, 41]}, {grid[0, 23], grid[0, 76]},
              {grid[0, 76], grid[0, 98]}, {grid[0, 50], grid[0, 69]}, {grid[0, 0], grid[0, 67]},
              {grid[0, 5], grid[0, 34]}, {grid[0, 42], grid[0, 76]}, {grid[0, 21], grid[0, 37]},
              {grid[0, 3], grid[0, 18]}, {grid[0, 25], grid[0, 56]}, {grid[0, 20], grid[0, 82]},
              {grid[0, 65], grid[0, 94]}, {grid[0, 40], grid[0, 45]}, {grid[0, 0], grid[0, 23]},
              {grid[0, 69], grid[0, 85]}, {grid[0, 31], grid[0, 49]}, {grid[0, 76], grid[0, 78]},
              {grid[0, 29], grid[0, 98]}, {grid[0, 31], grid[0, 72]}, {grid[0, 22], grid[0, 68]},
              {grid[0, 55], grid[0, 69]}, {grid[0, 14], grid[0, 38]}, {grid[0, 12], grid[0, 22]},
              {grid[0, 28], grid[0, 71]}, {grid[0, 57], grid[0, 58]}, {grid[0, 35], grid[0, 82]},
              {grid[0, 12], grid[0, 83]}, {grid[0, 17], grid[0, 34]}, {grid[0, 41], grid[0, 51]},
              {grid[0, 4], grid[0, 91]}, {grid[0, 75], grid[0, 84]}, {grid[0, 1], grid[0, 87]},
              {grid[0, 23], grid[0, 77]}, {grid[0, 69], grid[0, 71]}, {grid[0, 25], grid[0, 65]},
              {grid[0, 44], grid[0, 58]}, {grid[0, 16], grid[0, 59]}, {grid[0, 54], grid[0, 82]},
              {grid[0, 0], grid[0, 4]}, {grid[0, 31], grid[0, 80]}, {grid[0, 28], grid[0, 74]},
              {grid[0, 62], grid[0, 90]}, {grid[0, 77], grid[0, 84]}, {grid[0, 24], grid[0, 29]},
              {grid[0, 10], grid[0, 88]}, {grid[0, 34], grid[0, 44]}, {grid[0, 52], grid[0, 73]},
              {grid[0, 47], grid[0, 62]}, {grid[0, 1], grid[0, 91]}, {grid[0, 27], grid[0, 38]},
              {grid[0, 57], grid[0, 85]}, {grid[0, 58], grid[0, 73]}, {grid[0, 55], grid[0, 97]},
              {grid[0, 71], grid[0, 95]}, {grid[0, 49], grid[0, 50]}, {grid[0, 52], grid[0, 85]},
              {grid[0, 16], grid[0, 32]}, {grid[0, 17], grid[0, 20]}, {grid[0, 67], grid[0, 79]},
              {grid[0, 37], grid[0, 81]}, {grid[0, 27], grid[0, 76]}, {grid[0, 61], grid[0, 79]},
              {grid[0, 42], grid[0, 71]}, {grid[0, 7], grid[0, 69]}, {grid[0, 53], grid[0, 84]},
              {grid[0, 17], grid[0, 31]}, {grid[0, 24], grid[0, 56]}, {grid[0, 43], grid[0, 66]},
              {grid[0, 72], grid[0, 87]}, {grid[0, 10], grid[0, 30]}, {grid[0, 30], grid[0, 64]},
              {grid[0, 60], grid[0, 78]}, {grid[0, 36], grid[0, 52]}, {grid[0, 12], grid[0, 23]},
              {grid[0, 23], grid[0, 66]}, {grid[0, 16], grid[0, 53]}, {grid[0, 24], grid[0, 25]},
              {grid[0, 58], grid[0, 87]}, {grid[0, 41], grid[0, 79]}, {grid[0, 19], grid[0, 52]},
              {grid[0, 14], grid[0, 73]}, {grid[0, 16], grid[0, 68]}, {grid[0, 9], grid[0, 63]},
              {grid[0, 12], grid[0, 38]}, {grid[0, 51], grid[0, 85]}, {grid[0, 35], grid[0, 70]},
              {grid[0, 36], grid[0, 87]}, {grid[0, 27], grid[0, 84]}, {grid[0, 18], grid[0, 23]},
              {grid[0, 14], grid[0, 49]}, {grid[0, 5], grid[0, 47]}, {grid[0, 19], grid[0, 32]},
              {grid[0, 5], grid[0, 16]}, {grid[0, 30], grid[0, 39]}, {grid[0, 56], grid[0, 71]},
              {grid[0, 40], grid[0, 59]}, {grid[0, 6], grid[0, 32]}, {grid[0, 69], grid[0, 97]},
              {grid[0, 38], grid[0, 43]}, {grid[0, 9], grid[0, 22]}, {grid[0, 46], grid[0, 89]},
              {grid[0, 54], grid[0, 92]}, {grid[0, 37], grid[0, 71]}, {grid[0, 39], grid[0, 74]},
              {grid[0, 68], grid[0, 86]}, {grid[0, 37], grid[0, 89]}, {grid[0, 82], grid[0, 98]},
              {grid[0, 51], grid[0, 76]}, {grid[0, 60], grid[0, 62]}, {grid[0, 19], grid[0, 73]},
              {grid[0, 52], grid[0, 84]}, {grid[0, 44], grid[0, 95]}, {grid[0, 39], grid[0, 91]},
              {grid[0, 1], grid[0, 81]}, {grid[0, 15], grid[0, 97]}, {grid[0, 9], grid[0, 38]},
              {grid[0, 29], grid[0, 36]}, {grid[0, 41], grid[0, 52]}, {grid[0, 59], grid[0, 69]},
              {grid[0, 68], grid[0, 90]}, {grid[0, 30], grid[0, 42]}, {grid[0, 6], grid[0, 79]},
              {grid[0, 21], grid[0, 65]}, {grid[0, 45], grid[0, 59]}, {grid[0, 17], grid[0, 33]},
              {grid[0, 8], grid[0, 69]}, {grid[0, 40], grid[0, 96]}, {grid[0, 55], grid[0, 73]},
              {grid[0, 31], grid[0, 99]}, {grid[0, 18], grid[0, 35]}, {grid[0, 45], grid[0, 55]},
              {grid[0, 76], grid[0, 95]}, {grid[0, 58], grid[0, 86]}, {grid[0, 42], grid[0, 90]},
              {grid[0, 10], grid[0, 34]}, {grid[0, 8], grid[0, 38]}, {grid[0, 22], grid[0, 74]},
              {grid[0, 11], grid[0, 75]}, {grid[0, 24], grid[0, 69]}, {grid[0, 24], grid[0, 53]},
              {grid[0, 2], grid[0, 53]}, {grid[0, 18], grid[0, 98]}, {grid[0, 26], grid[0, 83]},
              {grid[0, 10], grid[0, 69]}, {grid[0, 9], grid[0, 40]}, {grid[0, 64], grid[0, 85]},
              {grid[0, 13], grid[0, 52]}, {grid[0, 57], grid[0, 81]}, {grid[0, 16], grid[0, 23]},
              {grid[0, 8], grid[0, 59]}, {grid[0, 83], grid[0, 99]}, {grid[0, 17], grid[0, 95]},
              {grid[0, 54], grid[0, 56]}, {grid[0, 16], grid[0, 79]}, {grid[0, 1], grid[0, 89]},
              {grid[0, 46], grid[0, 58]}, {grid[0, 15], grid[0, 89]}, {grid[0, 49], grid[0, 70]},
              {grid[0, 49], grid[0, 91]}, {grid[0, 70], grid[0, 77]}, {grid[0, 1], grid[0, 86]}, ]


    model = cp.Model()
    for i, j in scopes:
        model += i != j

    C_T = list(set(toplevel_list(model.constraints)))
    # Create the language:
    AV = absvar(2)  # create abstract vars - as many as maximum arity

    # create abstract relations using the abstract vars
    lang = [AV[0] != AV[1], AV[0] < AV[1], AV[0] > AV[1], AV[0] >= AV[1], AV[0] <= AV[1]]

    instance = ProblemInstance(variables=grid, language=lang, name="random")

    oracle = ConstraintOracle(C_T)

    return instance, oracle
