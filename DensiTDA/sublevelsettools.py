import matplotlib.pyplot as plt
from scipy.sparse.csgraph import connected_components
import gudhi
import numpy as np
import math 
from tqdm import tqdm
from collections import defaultdict
import itertools
from scipy.spatial.distance import cdist #, pdist, squareform

class SimplexTree: 
    
    def __init__(self): 
        
        self.X = [-1, defaultdict(lambda: [ 0.0, defaultdict(list)] ) ]

    def contains_simplex(self, my_tuple): 
        
        curr_level = self.X
        for index in my_tuple: 
            if index in curr_level[1].keys():
                curr_level = curr_level[1][index] 
            else: 
                return False 
            
        return True
    
    def simplex_val(self, my_tuple): 
        
        curr_level = self.X
        for index in my_tuple: 
            if index in curr_level[1].keys():
                curr_level = curr_level[1][index] 
            else: 
                return math.inf 
            
        return curr_level[0]
    
    def simplex_leaves(self, my_tuple): 
        
        curr_level = self.X
        for index in my_tuple: 
            if index in curr_level[1].keys():
                curr_level = curr_level[1][index] 
            else: 
                return [] 
            
        return list(curr_level[1].keys())
    
    def add_simplex(self, new_simplex,val):
    
        curr_level = self.X
        for index in new_simplex[:-1]: 
            if index in curr_level[1].keys():
                curr_level = curr_level[1][index] 
            else: 
                return False 
        
        curr_level[1][new_simplex[-1]] = [ val, defaultdict(lambda: [ 0.0, defaultdict(list)] ) ]
        
        return True

def VoronoiRips(S, total_space, max_dimension = -1, max_radius = 1): 

    n = len(S)

    V = defaultdict(list) 

    assigned_voronoi_cells = np.argsort(cdist(total_space, S), axis=1)

    for i, curr_point in enumerate(total_space):
        V[assigned_voronoi_cells[i][0]].append(curr_point)

    def simplex_proportion(simplex): 
        
        fuzzy = 1
        
        simplex_set = set(simplex)
        
        count = 0
        for i in range(n): 
            subface = set(assigned_voronoi_cells[i,0:len(simplex) + 1])
            if simplex_set.issubset(subface):
                count += 1
        
        if count != 0:
            return 1/count
        else: 
            return np.inf

    if max_dimension < 0: 
        max_dimension = len(S[0,:])
        
    VR_complex = defaultdict(list)
    X = SimplexTree()
    
    for i, s in enumerate(S): 
        value = simplex_proportion([i])
        VR_complex[0].append(([i], value))
        X.add_simplex([i], value)

    print("Evaluating Dimension", 1)
    
    Y = np.zeros((len(S), len(S)))
    adjacency = np.zeros((len(S), len(S)))
    with tqdm(total = len(S) ** 2) as pbar:
        for i in range(len(S)):
            curr_row = []
            #mu_i, C_i = get_distribution([i])
            for j in range(len(S)): 
                
#                 if i != j: 
#                     mu_j, C_j = get_distribution([j])
#                     center_distance = np.linalg.norm(S[i] - S[j]) + symmetric_cosine_distance(i, j)
#                 else:
#                     center_distance = 0
                if i < j:
                    center_distance = simplex_proportion([i,j]) #voronoi_distance([i], [j])
                elif j < i:
                    center_distance = simplex_proportion([j,i])
                else: 
                    center_distance = simplex_proportion([i])
                    
                if center_distance < max_radius:
                    VR_complex[1].append(([i,j], center_distance))
                    Y[i,j] = center_distance
                    adjacency[i,j] = 1
                    X.add_simplex([i,j], center_distance)
                else:
                    Y[i,j] = math.inf

                pbar.update(1)
                
    print("\tNumber of Connected Components: ", connected_components(adjacency)[0])

    for curr_dim in range(2,max_dimension + 1):
        
        print("Estimating Number of Facets for dimension ", curr_dim, "Part 1:")
        
        facets_to_consider = VR_complex[curr_dim-1]
        visited_prev_words = SimplexTree()
        visited_prev_word_list = []
        
        if len(facets_to_consider) == 0:
            print("No facets to consider")
            break
        
        with tqdm(total = len(facets_to_consider)) as pbar:
            for facet, val in facets_to_consider:
                sub_facet = facet[:-1]
                if not visited_prev_words.contains_simplex(sub_facet):
                    visited_prev_words.add_simplex(sub_facet,0.0)
                    visited_prev_word_list.append(sub_facet)
                pbar.update(1)
                    
        print("Estimating Number of Facets for dimension ", curr_dim, "Part 2:")
        
        if len(visited_prev_word_list) == 0:
            print("No facets to consider")
            break
        
        Sigma = []
        with tqdm(total = len(visited_prev_word_list)) as pbar:
            for word in visited_prev_word_list:
                indices = X.simplex_leaves(word)
                for choose_pair in itertools.combinations(indices, r = 2):
                    suggested_word = word + list(choose_pair)
                    flag = True
                    for subsimplex in list(itertools.combinations(suggested_word, len(suggested_word) - 1)):
                        if not X.contains_simplex(subsimplex): 
                            flag = False
                            break

                    if flag:
                        Sigma.append(word + list(choose_pair))
                        
                pbar.update(1)
        
        print("Evaluating Dimension", curr_dim)
        
        if len(Sigma) == 0:
            print("No facets to consider")
            break
        
        with tqdm(total = len(Sigma)) as pbar:
            for simplex in Sigma:
#                 value = 0
#                 for subface in itertools.combinations(simplex, len(simplex) - 1):
#                     value = max(X.simplex_val(subface), value) 
                
                value = simplex_proportion(simplex)
#                 max_distance = 0
#                 for i in simplex:
#                     for j in simplex:
#                         if i != j:
#                             max_distance = max(max_distance, symmetric_cosine_distance(i, j))
#                 value += max_distance
    
#                 for subface_1 in itertools.combinations(simplex, len(simplex) - 1):
#                     mu_1, C_1 = get_distribution(subface_1)
#                     for subface_2 in itertools.combinations(simplex, len(simplex) - 1):
#                         mu_2, C_2 = get_distribution(subface_2)
#                         value += wasserstein_metric(mu_1, C_1, mu_2, C_2)
                        
                if value != math.inf and value < max_radius:
                    VR_complex[curr_dim].append((simplex, value))
                    X.add_simplex(simplex, value)

                pbar.update(1)

#         with tqdm(total = len(Sigma)) as pbar:
#             for simplex in Sigma:
#                 value = 0
#                 for subface in itertools.combinations(simplex, 2):
#                     i = subface[0]
#                     j = subface[1]
                    
#                     if i == j:
#                         continue
                    
#                     value += np.linalg.norm(S[i] - S[j]) + symmetric_cosine_distance(i, j)

#                 if value != math.inf and value < max_radius:
#                     VR_complex[curr_dim].append((simplex, value))
#                     X.add_simplex(simplex, value)

#                 pbar.update(1)
    
    return VR_complex