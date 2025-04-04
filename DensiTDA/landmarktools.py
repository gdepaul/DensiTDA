import numpy as np
from numpy import linalg as LA
from tqdm import tqdm
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat, chain

def K(r): 
    return np.exp(-r ** 2 / 2)
    
def p(x,y,h):
    return K(LA.norm(x - y) / h)
    
def f(x, powers, X_dom, h):
        
    my_sum = 0
        
    for a_i, x_i in zip(powers, X_dom):
        my_sum += a_i*p(x,x_i,h)
            
    return my_sum

def conv_projection(x, powers, X_dom, h):
        
    my_num = 0 * X_dom[0]
    my_den = 0
        
    for a_i, x_i in zip(powers, X_dom):
        curr_val = a_i*p(x,x_i,h)
        my_num += curr_val * x_i
        my_den += curr_val
            
    return my_num / my_den

def batch_f(curr_batch, powers, X_dom, h): 

    batch_result = []

    for x in curr_batch:
        batch_result.append(f(x,powers,X_dom,h))

    return batch_result

def optimal_convex_landmarking(X, A, h, s): 

    S = []
    Y = []
    epsilon = -np.log(s)

    print("Calculating Projections p(x)")
    with tqdm( total = len(X) ) as pbar:
        for x in X: 
            Y.append(conv_projection(x, A, X, h))
            pbar.update(n=1)

    exp_conjugate = []

    print("Calculating tilde f(y)")
    with tqdm( total = len(X) ) as pbar:
        for x, y in zip(X,Y): 
            exp_conjugate.append( f(x, A, X, h) * np.exp( (LA.norm(x - y) ** 2) / (2 * h ** 2) ) )
            pbar.update(n=1)

    chosen_landmarks = []
    new_powers = []
    print("Greedy algorithm")
    with tqdm( total = len(X) ) as pbar:
        for k in np.argsort(exp_conjugate):
            flag = 1
            for y in chosen_landmarks:
                if ( LA.norm(Y[k] - y) ** 2 ) < 2 * h ** 2 * epsilon:
                    flag = 0
            if flag == 1: 
                chosen_landmarks.append(Y[k])
                new_powers.append(exp_conjugate[k])
            pbar.update(n=1)

    return chosen_landmarks, new_powers

def cosine_similarity(x,y,t):
    r = np.dot(x,y)
    
    if r > 0: 
        return r ** t 
    else: 
        return 0 

def k_fun(x, powers, X_dom, t):
        
    my_sum = 0
        
    for a_i, x_i in zip(powers, X_dom):
        my_sum += a_i*cosine_similarity(x,x_i,t)
            
    return my_sum

def phi_x(x,y):

    return y / (np.dot(x,y))

def sphere_projection(x, powers, X_dom, t):
        
    my_sum = 0
        
    for a_i, x_i in zip(powers, X_dom):
        my_sum += a_i*cosine_similarity(x,x_i,t) * phi_x(x, x_i)
            
    return my_sum

def sphere_correction(v):

    return v / LA.norm(v)

def psi_conjugate(x,y,t): 

    r = np.dot(x,y)
    
    if r > 0: 
        return -t * np.log(r)
    else: 
        return np.inf 

def optimal_spherical_landmarking(X, A, t, s, a):
    S = []
    Y = []
    epsilon = -np.log(s)

    print("Calculating Projections p(x)")
    with tqdm( total = len(X) ) as pbar:
        for x in X: 
            Y.append( sphere_correction( sphere_projection(x, A, X, t) / k_fun(x, A, X, t) ) )
            pbar.update(n=1)

    exp_conjugate = []

    print("Calculating tilde f(y)")
    with tqdm( total = len(X) ) as pbar:
        for x, y in zip(X,Y): 
            exp_conjugate.append( - np.log( k_fun(x, A, X, t) )  -  psi_conjugate(x,y,t) )
            pbar.update(n=1)

    a = ( max(exp_conjugate) + min(exp_conjugate) )/2 
    print(a)

    chosen_landmarks = []
    new_powers = []
    print("Greedy algorithm")
    with tqdm( total = len(X) ) as pbar:
        for k in np.argsort(exp_conjugate):
            flag = 1
            for y in chosen_landmarks:
                if exp_conjugate[k] > a or cosine_similarity(y,Y[k],t) > s: # or  :
                    flag = 0
            if flag == 1: 
                chosen_landmarks.append(Y[k])
                new_powers.append(exp_conjugate[k])
            pbar.update(n=1)

    return chosen_landmarks, new_powers

def max_of_gaussians_landmarking_helper(X, A, candidate_landmarks, h, s):
    
    def GaussFit(y, c):
        
        #c = f(y)
            
        z = 0
        for a_i, x_i in zip(A, X):
            z += a_i * p(y,x_i,h) * x_i
            
        z /= c 
        
        b = c / p(z, y,h)
        
        return b, lambda x : b * p(z, x,h) #b, z

    f_y = np.array([0.0 for x in candidate_landmarks])

    f_x = []

    print("Initializing Distrbution over Candidate Landmark Points:")

    batch_size = max(10, int(len(candidate_landmarks) / 2000))
    
    batches = []
    curr_index = 0
    while curr_index + batch_size < len(candidate_landmarks): 
        batches.append(candidate_landmarks[range(curr_index, curr_index + batch_size)])
        curr_index += batch_size
    batches.append(candidate_landmarks[range(curr_index, len(candidate_landmarks)),:])

    with ProcessPoolExecutor(max_workers=12) as pool:
        result = list(tqdm(pool.map(batch_f, batches, repeat(A), repeat(X), repeat(h)), total = len(batches)))

    f_x = []
    for a_result in result: 
        f_x += a_result
    # with tqdm( total = len(candidate_landmarks) ) as pbar:
    #     for k, x in enumerate(candidate_landmarks):
    #         f_x.append(f(x, A, X))

    #         pbar.update(1)
            
    f_x = np.array(f_x)

    chosen_landmark_indices = []
    chosen_landmarks = []
    total_gaussians = []
    B = []

    print("Maximizing Gaussians over Landmark Points:")
    # with tqdm( total = len(candidate_landmarks) ) as pbar:
        
        # while np.max(f_x - f_y) > 0: 
            
        #     k = np.argmax(f_x - f_y)
        #     y_k = X[k]

        #     g_k = GaussFit(y_k, f_x[k])
        #     # b_k, z_k = GaussFit(y_k, f_x[k])
        #     # g_k = lambda x : b_k * p(z_k, x)
            
        #     chosen_landmark_indices.append(k)
        #     chosen_landmarks.append(y_k)
        #     total_gaussians.append(g_k)
        #     #B.append(b_k)
                
        #     # penalize values 
        #     count_satistied = 0
        #     for i, x in enumerate(candidate_landmarks):
        #         # update with newly chosen landmark point's contributed gaussian 
        #         f_y[i] = max(g_k(x), f_y[i])
        #         # eliminate landmarks that 
        #         if s * f_x[i] <= f_y[i]:
        #             f_y[i] = f_x[i]
        #             count_satistied += 1

        # inc = count_satistied - pbar.n
        # pbar.update(n=inc)
    
        #print(len(chosen_landmarks), count_satistied)

    with tqdm( total = len(candidate_landmarks) ) as pbar:
    
        while np.max(f_x - f_y) > 0: 
            
            k = np.argmax(f_x - f_y)
            y_k = X[k]
    
            b_k, g_k = GaussFit(y_k, f_x[k])

            chosen_landmark_indices.append(k)
            chosen_landmarks.append(y_k)
            total_gaussians.append(g_k)
            B.append(b_k)
                    
            # penalize values         
            count_satistied = 0
            for i, x in enumerate(candidate_landmarks):
                # update with newly chosen landmark point's contributed gaussian 
                f_y[i] = max(g_k(x), f_y[i])
                # eliminate landmarks that 
                if s * f_x[i] <= f_y[i]:
                    f_y[i] = f_x[i]
                    count_satistied += 1
    
            inc = count_satistied - pbar.n
            pbar.update(n=inc)

    # return alpha scheme 

    return chosen_landmarks, total_gaussians, B

def max_of_gaussians_landmarking(X, A, candidate_landmarks, h, s):

    chosen_landmarks, total_gaussians, powers = max_of_gaussians_landmarking_helper(X, A, candidate_landmarks, h, s)
    
    return chosen_landmarks, powers

def max_of_gaussians_2D_plot(X, A, candidate_landmarks, h, s, cut_off = 0.05):

    chosen_landmarks, total_gaussians, powers = max_of_gaussians_landmarking_helper(X, A, candidate_landmarks, h, s)

    X_min = min(X[:,0])
    X_max = max(X[:,0])
    Y_min = min(X[:,1])
    Y_max = max(X[:,1])

    my_X = np.linspace(X_min, X_max, 100)
    my_Y = np.linspace(Y_min, Y_max, 100)

    X_, Y_ = np.meshgrid(my_X, my_Y)

    def combined_gaussians(set_of_gaussians, x):
        
        max_val = 0
        for g in set_of_gaussians:
            max_val = max(max_val, g(x))
            
        return max_val

    def surface_function(x,y): 

        return combined_gaussians(total_gaussians, [x,y])

    vfunc = np.vectorize(surface_function)

    Z_ = vfunc(X_, Y_)
    
    plt.contourf(X_, Y_, Z_, levels=np.linspace(cut_off, np.max(Z_), 30), cmap = 'plasma')
