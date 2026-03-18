import os

os.environ["MOSEKLM_LICENSE_FILE"] = "/home/rmineyev3/mosek/mosek.lic"

from pydrake.solvers import MosekSolver

solver = MosekSolver()
print(solver.available())
print(solver.enabled())
