from solvers.at import Solver_AT
from solvers.nt import Solver_NT
from solvers.alp import Solver_ALP
from solvers.atda import Solver_ATDA


solver_selector = {
    'NT': Solver_NT,
    'AT': Solver_AT,
    'ALP': Solver_ALP,
    'ATDA': Solver_ATDA,
}

