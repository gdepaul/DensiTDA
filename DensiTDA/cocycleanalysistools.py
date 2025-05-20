import numpy as np
from scipy.sparse import csr_matrix
from numpy import linalg as LA

class abstract_simplicial_complex():
    def __init__(self,coeffs=2):
        self.simplices = []
        self.coeffs = coeffs

    def get_subset_remove_i(self, A, i):
        B = []

        for k in range(len(A)):
            if k != i:
                B.append(A[k])

        return B;

    def new_simplice(self, A):

        if len(A) > 1:

            for i in range(len(A)):
                B = self.get_subset_remove_i(A, i);
                if B not in self.simplices:
                    print("Cannot add simplex - missing subset facets:", B);
                    return -1;

        if A not in self.simplices:
            self.simplices.append(A);

    def get_boundary_matrix(self):
        boundary_matrix = csr_matrix((len(self.simplices), len(self.simplices)), dtype=np.int8);

        # print(self.simplices)

        for i in range(len(self.simplices)):
            for j in range(i):
                if all(x in self.simplices[i] for x in self.simplices[j]) and len(self.simplices[j]) == len(
                        self.simplices[i]) - 1:
                    if i < j:
                        boundary_matrix[j, i] = 1
                    else:
                        boundary_matrix[j, i] = -1

        return boundary_matrix;

    def low(self, A, j):
        row_index = -1;

        m = len(self.simplices);

        for i in range(m):
            if A[i, j] != 0:
                row_index = i;

        return row_index

    def get_reduced_boundary_matrix(self):
        R = self.get_boundary_matrix();

        n = len(self.simplices);

        pivot_rows = [];
        pivot_columns = [];

        for j in range(n):
            low_j = self.low(R, j)
            while low_j != -1 and low_j in pivot_rows:
                for i in range(len(pivot_rows)):
                    if pivot_rows[i] == low_j:
                        j_0 = pivot_columns[i]
                        break;

                for i in range(n):
                    R[i, j] = (R[i, j] + R[i, j_0]) % 2;

                low_j = self.low(R, j)

            if low_j != -1 and low_j not in pivot_rows:
                pivot_rows.append(low_j);
                pivot_columns.append(j)

        return R, pivot_rows, pivot_columns
    
class abstract_filtration(abstract_simplicial_complex):

    def __init__(self):
        self.facet_values = []
        super(abstract_filtration, self).__init__();

    def new_simplice(self, facet, value):
        super(abstract_filtration, self).new_simplice(facet);
        self.facet_values.append(value);

    def get_persistence_pairs(self):
        R, pivot_rows, pivot_columns = super(abstract_filtration, self).get_reduced_boundary_matrix()

        persistence = [];
        epsilon = 10**(-6)

        for j in range(len(self.simplices)):

            if self.low(R, j) == -1:
                curr_birth = self.facet_values[j];

                if j in pivot_rows:
                    for i in range(len(pivot_rows)):
                        if pivot_rows[i] == j:
                            j_0 = pivot_columns[i]

                    curr_death = self.facet_values[j_0];
                else:
                    curr_death = float("inf");

                if curr_birth < curr_death - epsilon:
                    curr_dimension = len(self.simplices[j]) - 1;
                    
                    persistence.append((curr_dimension, (curr_birth, curr_death)));

        return persistence