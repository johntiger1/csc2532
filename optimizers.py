import matplotlib
import autograd.numpy as np  
from autograd import jacobian 
import matplotlib.pyplot as plt
import scipy
import scipy.optimize



##################################################
# Newton's method
##################################################
""" 
args:
  - n: Max iterations
  - f: function
  - dfdx
  - hessian
"""
def NewtonMethod (n, f, dfdx, hessian, x_0):
    xn = np.empty((n + 1, 2)) 
    xn[0] = x_0

    for i in range(n):
        # Get gradient at start location (df/dx or grad(f))
        gn = dfdx(xn[i])
        H = hessian(xn[i])

        # Compute search direction and magnitude (dx)
        #  with dx = -inv(H) * grad
        #delta_xn = np.empty((1, 2))
        try:
            delta_xn = -np.linalg.solve(H, gn)
        except np.linalg.LinAlgError:
            print("WARNING: Newton's Method Singular Matrix ERROR")
            for j in range(i + 1, n):
                xn[j] = xn[i]
            break
        xn[i+1] = xn[i] + delta_xn

    return xn

##################################################
# Steepest descent method
##################################################
def gd(n,f,dfdx,x_start,GD_alpha):
    # Initialize xs
    xs = np.empty((n + 1, 2)) 
    xs[0] = x_start
    # Get gradient at start location (df/dx or grad(f))
    for i in range(n):
        gs = dfdx(xs[i])
        # Compute search direction and magnitude (dx)
        #  with dx = - grad but no line searching
        xs[i + 1] = xs[i] - np.dot(GD_alpha, dfdx(xs[i]))

    return xs



##################################################
# Conjugate gradient method
##################################################
def cgd(n, f, dfdx, x_start, CGD_alpha):
    neg = np.array([[-1.0, 0.0], [0.0, -1.0]])
    # Initialize xc
    xc = np.empty((n + 1, 2)) 
    xc[0] = x_start
    # Initialize delta_gc
    delta_cg = np.zeros((n + 1, 2))
    # Initialize gc
    gc = np.zeros((n + 1, 2))
    # Get gradient at start location (df/dx or grad(f))
    for i in range(n):
        gc[i] = dfdx(xc[i])
        # Compute search direction and magnitude (dx)
        #  with dx = - grad but no line searching
        if i == 0:
            beta = 0
            delta_cg[i] = - np.dot(CGD_alpha, dfdx(xc[i]))
        else:
            beta = np.dot(gc[i], gc[i]) / np.dot(gc[i - 1], gc[i - 1])
            delta_cg[i] = CGD_alpha * np.dot(neg, dfdx(xc[i])) + beta * delta_cg[i - 1]
        xc[i + 1] = xc[i] + delta_cg[i]

    return xc


##################################################
# Quasi-Newton method
# I really don't trust this code or its behaviour
##################################################
def alleged_BFGS(n, f, dfdx,x_start, TEMP_B0, BFGS_alpha):
    # Initialize delta_xq and gamma
    delta_xq = np.zeros((2, 1))
    gamma = np.zeros((2, 1))
    part1 = np.zeros((2, 2))
    part2 = np.zeros((2, 2))
    part3 = np.zeros((2, 2))
    part4 = np.zeros((2, 2))
    part5 = np.zeros((2, 2))
    part6 = np.zeros((2, 1))
    part7 = np.zeros((1, 1))
    part8 = np.zeros((2, 2))
    part9 = np.zeros((2, 2))
    # Initialize xq
    xq = np.empty((n + 1, 2)) 
    xq[0] = x_start
    # Initialize gradient storage
    g = np.zeros((n + 1, 2))
    g[0] = dfdx(xq[0])
    # Initialize hessian storage
    h = np.zeros((n + 1, 2, 2))
    h[0] = TEMP_B0
    for i in range(n):

        search_dirn = np.linalg.solve(h[i], g[i])
        # Compute search direction and magnitude (dx)
        #  with dx = -alpha * inv(h) * grad
        delta_xq = -np.dot(BFGS_alpha[i], np.linalg.solve(h[i], g[i]))
        # delta_xq = - np.linalg.solve(h[i], g[i])

        xq[i + 1] = xq[i] + delta_xq

        # Get gradient update for next step
        g[i + 1] = dfdx(xq[i + 1])

        # Get hessian update for next step
        gamma = g[i + 1] - g[i]
        part1 = np.outer(gamma, gamma)
        part2 = np.outer(gamma, delta_xq)
        part3 = np.dot(np.linalg.pinv(part2), part1)

        part4 = np.outer(delta_xq, delta_xq)
        part5 = np.dot(h[i], part4)
        part6 = np.dot(part5, h[i])
        part7 = np.dot(delta_xq, h[i])
        part8 = np.dot(part7, delta_xq)
        part9 = np.dot(part6, 1 / part8)

        h[i + 1] = h[i] + part3 - part9

    return xq


'''

Recall den => in ml4h stuff!
General QN Rank 1 method as described in Broyden 1973
We can also verify the subsequence of Hessian updates also converges

===

Given an initialization, and a function, will apply QN update rule iteratively to optimize the function.
We DO NOT require access to the initial Hessian!

But we do require access to the gradient

args: 
  - k: the number of QN steps to take
  - f: the function to evaluate
  - gradient: the gradient function that we can evaluate
  - c: ["broyden's first", "davidon's method"] the method to use
  - x0: the initial guess for the iterates
'''
# we have H\delta = grad_x => solving for delta. But B approximates the hessian, not the hessian inverse
def general_rank_1_QN(k,f,gradient,c,x_0, B_0):
    # B_0 is our HESSIAN approximation (NOT hessian inverse). This is very critical!
    counter = 0
    x_k = x_0
    B_k = B_0
    cond = True

    # Initalize the plots
    x_iterates = np.zeros((k + 1, 2)) 
    x_iterates[0] = x_0



    while cond:

        # new iterates
        search_direction = np.linalg.solve(B_k, gradient(x_k))
        x_k_and_1  = x_k - search_direction #equiv to finding B^{-1} * grad. equiv again to solving B\delta = grad; for \delta
        # compute k+1 quantities
        y_k = gradient(x_k_and_1) - gradient(x_k)

        s_k = x_k_and_1 - x_k

        # Terminate if we have converged in finite steps
        if not np.any(s_k):
            # Fix rest of iterates to the converged value
            for j in range(counter, k+1):
                x_iterates[j] = x_k
            break

        #noise = np.random.uniform(500, 1000)
        #noise = np.random.uniform((0,1), size=(2,2,))
        #print("noise added:")
        #print(noise)
        c = y_k
        #print(c)
        #c = noise@y_k # fix to a fixed method
        #print(c)
        # compute the next B_{k+1} iteration
        B_k_and_1 = B_k + np.outer(y_k - np.matmul(B_k,s_k), np.transpose(c)/np.dot(c, s_k))

        # add the noise. that is bounded within some quantity.

        # update the matrix:
        B_k = B_k_and_1
        x_k = x_k_and_1

        # logic for checking whether to terminate or not
        not_done = True
        counter += 1
        cond = counter < k and not_done
        x_iterates[counter] = x_k

    return x_k, x_iterates



def rank_1_H_update(H,s,y,d):
    return H + np.outer((s-np.matmul(H,y)), d)/np.inner(d,y)

"""
args
    -k: Max iterations
    -f: Function to optimize
    -gradient: gradient of f
    -d: "broyden2" for d=y; or "mccormick" for d=s 
    -x_0: startint x
    -H_0: starting H
    -noise: Function from step size to noise, default is no noise
"""
def general_rank_1_QN_H(k,f,gradient,d,x_0, H_0, noise=lambda s:0):
    counter = 0
    x_k = x_0
    H_k = H_0

    if d == "broyden2":
        def update(H,s,y):
            return rank_1_H_update(H,s,y,y+noise(s))
    else:
        assert(d == "mccormick")
        def update(H,s,y):
            return rank_1_H_update(H,s,y,s+noise(s))

    # Initalize the plots
    x_iterates = np.zeros((k + 1, 2)) 
    x_iterates[0] = x_0

    while counter < k:
        x_k_and_1  = x_k - np.matmul(H_k, gradient(x_k))
        y_k = gradient(x_k_and_1) - gradient(x_k)
        s_k = x_k_and_1 - x_k

        # Terminate if we have converged in finite steps
        if not np.any(s_k):
            # Fix rest of iterates to the converged value
            for j in range(counter, k+1):
                x_iterates[j] = x_k
            break

        # update the matrix:
        H_k = update(H_k, s_k, y_k)
        x_k = x_k_and_1

        counter += 1
        x_iterates[counter] = x_k

    return x_k, x_iterates


# TODO: Can probably increase efficiency by moving division before outer products
def rank_2_H_update(H,s,y,d):
    Hy = np.matmul(H,y)
    temp = np.outer(s-Hy,d)
    ddT = np.outer(d,d)
    dTy = np.inner(d,y)

    if dTy**2 == 0:
        raise ZeroDivisionError

    return H + (temp + np.transpose(temp))/dTy - np.inner(y,s-Hy)*ddT/(dTy**2)

"""
args
    -k: Max iterations
    -f: Function to optimize
    -gradient: gradient of f
    -d: "greenstadt" for d=y; or "BFGS" for d=s 
    -x_0: startint x
    -H_0: starting H
    -noise: Function from step size to noise, default is no noise
"""
def general_rank_2_QN_H(k,f,gradient,d,x_0, H_0, noise=lambda s:0):
    counter = 0
    x_k = x_0
    H_k = H_0

    bernoulli = np.random.binomial(1,0.5)

    def update_greenstadt(H, s, y):
        return rank_2_H_update(H, s, y, y + noise(s))

    def update_BFGS(H, s, y):
        return rank_2_H_update(H, s, y, s + noise(s))

    if d == "greenstadt":
        f.update = update_greenstadt
    else:
        f.update = update_BFGS



    # Initalize the plots
    x_iterates = np.zeros((k + 1, 2)) 
    x_iterates[0] = x_0

    while counter < k:
        try:

            x_k_and_1  = x_k - np.matmul(H_k, gradient(x_k))
            y_k = gradient(x_k_and_1) - gradient(x_k)
            s_k = x_k_and_1 - x_k

            # Terminate if we have converged in finite steps
            if not np.any(s_k):
                # Fix rest of iterates to the converged value
                for j in range(counter, k+1):
                    x_iterates[j] = x_k
                break

            # update the matrix:
            bernoulli = np.random.binomial(1, 0.5)
            # print(bernoulli)
            if bernoulli > 0.5:
                f.update = update_greenstadt
            else:
                f.update = update_BFGS
            H_k = f.update(H_k, s_k, y_k)
            x_k = x_k_and_1

            counter += 1
            x_iterates[counter] = x_k
        except ZeroDivisionError: # Terminate if update undefined (therefore converged)
            for i in range(counter, k+1):
                x_iterates[i] = x_k
            print("WARNING: rank 2 H terminated due to undefined update")
            break

    return x_k, x_iterates


def rank_2_B_update(B,y,s,c):
    normalizer = np.dot(s,c)
    temp = np.outer(y - np.matmul(B, s), np.transpose(c))
    symmetric_term = (temp + np.transpose(temp)) /normalizer
    residual_term = (np.dot(np.transpose(s), y-np.matmul(B,s) ) * np.outer(c,np.transpose(c)))/np.power(normalizer, 2)
    return B + symmetric_term - residual_term

'''
Code for implementing a general rank 2 update as in Broyden 1973. This is the B formulation. 
'''
def general_rank_2_QN(k,f,gradient,c,x_0, init_b0):

    # B_0 is our HESSIAN approximation (NOT hessian inverse). This is very critical!
    B_0 = init_b0
    counter = 0
    x_k = x_0
    B_k = B_0
    cond = True

    # Initalize the plots
    x_iterates = np.empty((k + 1, 2))
    x_iterates[0] = x_0

    '''
    Inner function which specifies the update rule
    '''
    def update(B_k, y_k, s_k  ):
        return rank_2_B_update(B_k, y_k, s_k, s_k)
        pass


    while cond:

        # new iterates
        search_direction = np.linalg.solve(B_k, gradient(x_k))
        x_k_and_1  = x_k - search_direction #equiv to finding B^{-1} * grad. equiv again to solving B\delta = grad; for \delta
        # compute k+1 quantities
        y_k = gradient(x_k_and_1) - gradient(x_k)

        s_k = x_k_and_1 - x_k

        # Terminate if we have converged in finite steps
        if not np.any(s_k):
            # Fix rest of iterates to the converged value
            for j in range(counter, k+1):
                x_iterates[j] = x_k
            break


        # compute the next B_{k+1} iteration
        B_k_and_1 = update(B_k, y_k, s_k )

        # update the matrix:
        B_k = B_k_and_1
        x_k = x_k_and_1

        # logic for checking whether to terminate or not
        not_done = True
        counter += 1
        cond = counter < k and not_done
        x_iterates[counter] = x_k

    return x_k, x_iterates