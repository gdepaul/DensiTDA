import numpy as np
from math import comb
from ctypes import * 
import ctypes.util
import plotly.graph_objs as go
from collections import defaultdict
from plotly.offline import iplot
import itertools
from qpsolvers import Problem, solve_problem #solve_qp, 
from tqdm import tqdm
import qpsolvers
import gudhi
import matplotlib.pyplot as plot
from numpy import genfromtxt
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from scipy.sparse.csgraph import connected_components

