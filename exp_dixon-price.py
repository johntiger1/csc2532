import matplotlib
import autograd.numpy as np  
from autograd import jacobian 
import matplotlib.pyplot as plt
import scipy
import scipy.optimize as optimize

'''
code for an experiment with Dixon Price Function
f(x) = (x_1-1)^2 + \sum_i=1 (2x_i^2-...)
https://www.sfu.ca/~ssurjano/dixonpr.html
'''
########## PLOTTING PARAMETERS ##########
ymin = -20
ymax = 20

xmin = -20
xmax = 20

step = 0.1

title = "Dixon Price Function"

#TODO: fix up newton's method or the bfgs?
# we can also find the optimum using scipy instead of hardcoding it!

png_prefix = "dixon-price_"
########## FUNCTION DEFINITION ################
# define objective function
def f(x):
    return (x[0]-1)**2  + 2*(2*(x[1]**2)-x[0])**2

def f_sep(x1,x2):
    return f([x1,x2])

dfdx = lambda x : jacobian(f)(np.array(x).astype(float))

hessian = lambda x : jacobian(jacobian(f))(np.array(x).astype(float))
########## INITIALIZATION PARAMETERS ##########

# SOLUTION POSITION
num_opt_soln = optimize.minimize(fun= f, x0= np.array([2, 2]), method="Nelder-Mead",jac=dfdx, hess=hessian)

print("Numerical optimal solution:")
print(num_opt_soln .x)
# print(opt_soln)

opt_x = np.array([1, 1/np.sqrt(2)])

H = hessian(np.array(opt_x)) # Exact 2nd derivatives (hessian)

# STARTING PARAMS
x_start = np.array([0,0])
TEMP_B0 = H + [[0.5,0.5],[0.5,0.5]] # H_0 is this thing's inverse

# Max iterations
max_iter = 8

# GD ALPHA
GD_alpha = 0.15

# CGD ALPHA
CGD_alpha = 0.15

# BFGS alpha - set to (1,1,max_iter) to force step size 1
#BFGS_alpha = alpha = np.linspace(0.1, 1.0, max_iter)
BFGS_alpha = alpha = np.linspace(1, 1, max_iter)
##############################################


# Design variables at mesh points
i1 = np.arange(xmin, xmax, step)
i2 = np.arange(ymin, ymax, step)
x1_mesh, x2_mesh = np.meshgrid(i1, i2)

# Create a contour plot

fig, ax = plt.subplots()

plt.ylim(ymin, ymax)
plt.xlim(xmin, xmax)
v_func = np.vectorize(f_sep)
ax.contour(x1_mesh, x2_mesh, v_func(x1_mesh, x2_mesh))

# Add some text to the plot
ax.set_title(title)
ax.set_xlabel('x1')
ax.set_ylabel('x2')


##################################################
# Newton's method
##################################################
n = max_iter
xn = np.zeros((n + 1, 2)) + opt_x
xn[0] = x_start

for i in range(n):
    # Get gradient at start location (df/dx or grad(f))
    gn = dfdx(xn[i])
    # Compute search direction and magnitude (dx)
    #  with dx = -inv(H) * grad
    #delta_xn = np.empty((1, 2))
    delta_xn = -np.linalg.solve(H, gn)
    xn[i+1] = xn[i] + delta_xn
ax.plot(xn[:, 0], xn[:, 1], 'k-o', label="Newton")

print("Newton's method returns")
print(xn[-1,:])

##################################################
# Steepest descent method
##################################################
# Number of iterations
n = max_iter
# Initialize xs
xs = np.zeros((n + 1, 2)) + opt_x
xs[0] = x_start
# Get gradient at start location (df/dx or grad(f))
for i in range(n):
    gs = dfdx(xs[i])
    # Compute search direction and magnitude (dx)
    #  with dx = - grad but no line searching
    xs[i + 1] = xs[i] - np.dot(GD_alpha, dfdx(xs[i]))
ax.plot(xs[:, 0], xs[:, 1], 'g-o', label="GD")

print("Gradient Descent returns")
print(xs[-1,:])

##################################################
# Conjugate gradient method
##################################################
# Number of iterations
n = max_iter
neg = np.array([[-1.0, 0.0], [0.0, -1.0]])
# Initialize xc
xc = np.zeros((n + 1, 2)) + opt_x
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
ax.plot(xc[:, 0], xc[:, 1], 'y-o',label="CG")

print("Conjugate Gradient method returns")
print(xc[-1,:])

##################################################
# Quasi-Newton method
##################################################
# Number of iterations
n = max_iter

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
xq = np.zeros((n + 1, 2)) + opt_x
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

ax.plot(xq[:, 0], xq[:, 1], 'r-o',label="QN")
print("BFGS method returns")
print(xq[-1,:])
'''

General QN Rank 1 method as described in Broyden 1973
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
def general_rank_1_QN(k,f,gradient,c,x_0):
    # B_0 is our HESSIAN approximation (NOT hessian inverse). This is very critical!
    B_0 = TEMP_B0
    counter = 0
    x_k = x_0
    B_k = B_0
    cond = True

    # Initalize the plots
    x_iterates = np.zeros((k + 1, 2)) + opt_x
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

qn_soln , qn_iterates = general_rank_1_QN(max_iter,f,dfdx,None,x_start)
print("rank 1 method returns")
print(qn_soln)
ax.plot(qn_iterates[:, 0], qn_iterates[:, 1], 'c-o',label="QN (1973)")
ax.legend()


def rank_2_B_update(B,y,s,c):
    normalizer = np.dot(s,c)
    temp = np.outer(y - np.matmul(B, s), np.transpose(c))
    symmetric_term = (temp + np.transpose(temp)) /normalizer
    residual_term = (np.dot(np.transpose(s), y-np.matmul(B,s) ) * np.outer(c,np.transpose(c)))/np.power(normalizer, 2)
    return B + symmetric_term - residual_term

'''
Code for implementing a general rank 2 update as in Broyden 1973. This is the B formulation. 
'''
def general_rank_2_QN(k,f,gradient,c,x_0, init_b0=TEMP_B0):

    # B_0 is our HESSIAN approximation (NOT hessian inverse). This is very critical!
    B_0 = init_b0
    counter = 0
    x_k = x_0
    B_k = B_0
    cond = True

    # Initalize the plots
    x_iterates = np.zeros((k + 1, 2)) + opt_x
    x_iterates[0] = x_0

    '''
    Inner function which specifies the update rule
    '''
    def update(B_k, y_k, s_k  ):
        return rank_2_B_update(B_k, y_k, s_k, s_k)

    def update(B_k, y_k, s_k  ):
        return rank_2_B_update(B_k, y_k, s_k, y_k)

    while cond:

        # new iterates
        search_direction = np.linalg.solve(B_k, gradient(x_k))
        x_k_and_1  = x_k - search_direction #equiv to finding B^{-1} * grad. equiv again to solving B\delta = grad; for \delta
        # compute k+1 quantities
        y_k = gradient(x_k_and_1) - gradient(x_k)

        s_k = x_k_and_1 - x_k

        # Terminate if we have converged in finite steps
        if not np.any(s_k):
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

qn_2B_soln , qn_2B_iterates = general_rank_2_QN(max_iter,f,dfdx,None,x_start)
print("rank 2 method returns")
print(qn_2B_soln )
ax.plot(qn_2B_iterates [:, 0], qn_2B_iterates [:, 1], 'm-o',label="QN2 B (1973)")



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
def general_rank_1_QN_H(k,f,gradient,d,x_0, H_0 = np.linalg.inv(TEMP_B0), noise=lambda s:0):
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
    x_iterates = np.zeros((k + 1, 2)) + opt_x
    x_iterates[0] = x_0

    while counter < k:
        x_k_and_1  = x_k - np.matmul(H_k, gradient(x_k))
        y_k = gradient(x_k_and_1) - gradient(x_k)
        s_k = x_k_and_1 - x_k

        # Terminate if we have converged in finite steps
        if not np.any(s_k):
            break

        # update the matrix:
        H_k = update(H_k, s_k, y_k)
        x_k = x_k_and_1

        counter += 1
        x_iterates[counter] = x_k

    return x_k, x_iterates



MODE = "mccormick"
qn_H1_soln , qn_H1_iterates = general_rank_1_QN_H(max_iter,f,dfdx,MODE,x_start)
# FOR NOISE USE noise = lambda s: np.random.multivariate_normal([0,0],[[1,0],[0,1]])
print("rank 1 H method \"{}\" returns".format(MODE))
print(qn_H1_soln)
ax.plot(qn_H1_iterates[:, 0], qn_H1_iterates[:, 1], 'm-o',label="QN-H (1973)")

# TODO: Can probably increase efficiency by moving division before outer products
def rank_2_H_update(H,s,y,d):
    Hy = np.matmul(H,y)
    temp = np.outer(s-Hy,d)
    ddT = np.outer(d,d)
    dTy = np.inner(d,y)
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
def general_rank_2_QN_H(k,f,gradient,d,x_0, H_0 = np.linalg.inv(TEMP_B0), noise=lambda s:0):
    counter = 0
    x_k = x_0
    H_k = H_0

    if d == "greenstadt":
        def update(H,s,y):
            return rank_2_H_update(H,s,y,y+noise(s))
    else:
        assert(d == "BFGS")
        def update(H,s,y):
            return rank_2_H_update(H,s,y,s+noise(s))

    # Initalize the plots
    x_iterates = np.zeros((k + 1, 2)) + opt_x
    x_iterates[0] = x_0

    while counter < k:
        x_k_and_1  = x_k - np.matmul(H_k, gradient(x_k))
        y_k = gradient(x_k_and_1) - gradient(x_k)
        s_k = x_k_and_1 - x_k

        # Terminate if we have converged in finite steps
        if not np.any(s_k):
            break

        # update the matrix:
        H_k = update(H_k, s_k, y_k)
        x_k = x_k_and_1

        counter += 1
        x_iterates[counter] = x_k

    return x_k, x_iterates

MODE = "BFGS"
qn_H2_soln , qn_H2_iterates = general_rank_2_QN_H(max_iter,f,dfdx,MODE,x_start)
# FOR NOISE USE noise = lambda s: np.random.multivariate_normal([0,0],[[1,0],[0,1]])
print("rank 2 H method \"{}\" returns".format(MODE))
print(qn_H2_soln)
ax.plot(qn_H2_iterates[:, 0], qn_H2_iterates[:, 1], marker='o', ls='-', color='lime',label="QN-H R2 (1973)")
ax.legend()

contours = ax.contour(x1_mesh, x2_mesh, v_func(x1_mesh, x2_mesh))
ax.clabel(contours , inline=True, fontsize=8)

# Save the figure as a PNG
fig.savefig(png_prefix+'contour.png')
plt.close(fig)



# qn_iterates-
'''
Computes the residuals
'''
def compute_residuals(iterates, opt_x):
    return np.linalg.norm(iterates-opt_x.transpose(), ord=2, axis=1)


fig,ax= plt.subplots()
ax.set_title("L2-norm between iterate and optimal")

ax.plot(np.arange(0, len(qn_iterates)), compute_residuals(qn_iterates, opt_x), label="QN (1973)")
ax.plot(np.arange(0, len(qn_iterates)), compute_residuals(xq, opt_x), label="BFGS")
ax.plot(np.arange(0, len(qn_iterates)), compute_residuals(xc, opt_x), label="CG")
ax.plot(np.arange(0, len(qn_iterates)), compute_residuals(xs, opt_x), label="GD")
ax.plot(np.arange(0, len(xn)), compute_residuals(xn, opt_x), label="Newton's method")
ax.plot(np.arange(0, len(qn_H1_iterates)), compute_residuals(qn_H1_iterates, opt_x), label="QN-H (1973)")
ax.plot(np.arange(0, len(qn_H2_iterates)), compute_residuals(qn_H2_iterates, opt_x), label="QN-H Rank-2 (1973)")

ax.plot(np.arange(0, len(qn_2B_iterates)), compute_residuals(qn_2B_iterates, opt_x), label="QN-B Rank-2 (1973)")


ax.set_xlabel("Iteration")
ax.set_ylabel("L2 norm between current estimate and optimal")
ax.legend()

fig.savefig(png_prefix+"iterate residuals")

fig,ax= plt.subplots()
ax.set_title("L2-norm between iterate and optimal")
ax.set_yscale('log') # Change to log-scale

ax.plot(np.arange(0, len(qn_iterates)), compute_residuals(qn_iterates, opt_x), label="QN (1973)")
ax.plot(np.arange(0, len(qn_iterates)), compute_residuals(xq, opt_x), label="BFGS")
ax.plot(np.arange(0, len(qn_iterates)), compute_residuals(xc, opt_x), label="CG")
ax.plot(np.arange(0, len(qn_iterates)), compute_residuals(xs, opt_x), label="GD")
ax.plot(np.arange(0, len(xn)), compute_residuals(xn, opt_x), label="Newton's method")
ax.plot(np.arange(0, len(qn_H1_iterates)), compute_residuals(qn_H1_iterates, opt_x), label="QN-H (1973)")
ax.plot(np.arange(0, len(qn_H2_iterates)), compute_residuals(qn_H2_iterates, opt_x), label="QN-H Rank-2 (1973)")
ax.plot(np.arange(0, len(qn_2B_iterates)), compute_residuals(qn_2B_iterates, opt_x), label="QN-B Rank-2 (1973)")

ax.set_xlabel("Iteration")
ax.set_ylabel("L2 norm between current estimate and optimal")
ax.legend()

fig.savefig(png_prefix+"log iterate residuals")
fig.show()
