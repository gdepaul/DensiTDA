import numpy as np
from numpy import linalg as LA
from tqdm import tqdm

def max_of_gaussians_landmarking(X, A, candidate_landmarks, h, s, metric = 'euclidean'):

    def K(r): 
        return np.exp(-r ** 2 / 2)
    
    def p(x,y):
        return K(LA.norm(x - y) / h)
    
    def f(x):
        
        my_sum = 0
        
        for a_i, x_i in zip(A, X):
            my_sum += a_i*p(x,x_i)
            
        return my_sum
    
    def GaussFit(y, c):
        
        #c = f(y)
            
        z = 0
        for a_i, x_i in zip(A, X):
            z += a_i * p(y,x_i) * x_i
            
        z /= c 
        
        b = c / p(z, y)
        
        return lambda x : b * p(z, x)
    
    def combined_gaussians(set_of_gaussians, x):
        
        max_val = 0
        for g in set_of_gaussians:
            max_val = max(max_val, g(x))
            
        return max_val

    f_y = np.array([0.0 for x in candidate_landmarks])

    f_x = []

    print("Initializing Distrbution over Candidate Landmark Points:")
    with tqdm( total = len(candidate_landmarks) ) as pbar:
        for k, x in enumerate(candidate_landmarks):
            f_x.append(f(x))

            pbar.update(1)
            
    f_x = np.array(f_x)

    chosen_landmark_indices = []
    chosen_landmarks = []
    total_gaussians = []

    print("Maximizing Gaussians over Landmark Points:")
    with tqdm( total = len(candidate_landmarks) ) as pbar:
        
        while np.max(f_x - f_y) > 0: 
            
            k = np.argmax(f_x - f_y)
            y_k = X[k]
            
            g_k =  GaussFit(y_k, f_x[k])
            
            chosen_landmark_indices.append(k)
            chosen_landmarks.append(y_k)
            total_gaussians.append(g_k)
                
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
    
        #print(len(chosen_landmarks), count_satistied)

    return chosen_landmarks

    