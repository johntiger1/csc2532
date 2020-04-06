## Generate a contour plot
# Import some other libraries that we'll need
# matplotlib and numpy packages must also be installed
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.optimize

NAME = "f(x) = x1^2 - 2*x1*x2 + 4*x2^2"
# define objective function
def f_sep(x1,x2):
    obj = x1 ** 2 - 2.0 * x1 * x2 + 4 * x2 ** 2
    return obj

def f(x):
    x1 = x[0]
    x2 = x[1]
    return f_sep(x1,x2)


# define objective gradient
def dfdx(x):
    x1 = x[0]
    x2 = x[1]
    grad = np.zeros((2))
    grad[0] = 2.0 * x1 - 2.0 * x2
    grad[1] = -2.0 * x1 + 8.0 * x2
    return grad

# Exact 2nd derivatives (hessian)
H = [[2.0, -2.0], [-2.0, 8.0]]

# Start location
x_start = [-3.0, 3.0]
opt_x = np.zeros((2,1))

# Design variables at mesh points
i1 = np.arange(-4.0, 4.0, 0.1)
i2 = np.arange(-4.0, 4.0, 0.1)
x1_mesh, x2_mesh = np.meshgrid(i1, i2)
#f_mesh = x1_mesh ** 2 - 2.0 * x1_mesh * x2_mesh + 4 * x2_mesh ** 2

# Create a contour plot

fig, ax = plt.subplots()
# Specify contour lines
#lines = range(2, 52, 2)
# Plot contours
#CS = ax.contour(x1_mesh, x2_mesh, f_mesh, lines)
# Label contours
#ax.clabel(CS, inline=1, fontsize=10)


v_func = np.vectorize(f_sep)    # major key!
contours = ax.contour(x1_mesh, x2_mesh, v_func(x1_mesh, x2_mesh))
ax.clabel(contours , inline=True, fontsize=8)
# Add some text to the plot
ax.set_title(NAME)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
# Show the plot
# plt.show()

##################################################
# Newton's method
##################################################
xn = np.zeros((2, 2))
xn[0] = x_start
# Get gradient at start location (df/dx or grad(f))
gn = dfdx(xn[0])
# Compute search direction and magnitude (dx)
#  with dx = -inv(H) * grad
delta_xn = np.empty((1, 2))
delta_xn = -np.linalg.solve(H, gn)
xn[1] = xn[0] + delta_xn
ax.plot(xn[:, 0], xn[:, 1], 'k-o', label="Newton")

##################################################
# Steepest descent method.

# ENSURE STEP SIZE IS SMALL, otherwise we may see divergence!
##################################################
# Number of iterations
n = 8
# Use this alpha for every line search
alpha = 0.15
# Initialize xs
xs = np.zeros((n + 1, 2))
xs[0] = x_start
# Get gradient at start location (df/dx or grad(f))
for i in range(n):
    gs = dfdx(xs[i])
    # Compute search direction and magnitude (dx)
    #  with dx = - grad but no line searching
    xs[i + 1] = xs[i] - np.dot(alpha, dfdx(xs[i]))
ax.plot(xs[:, 0], xs[:, 1], 'g-o', label="GD")

##################################################
# Conjugate gradient method

# SIMILARLY, we need ensure the step size is small enough!
##################################################
# Number of iterations
n = 8
# Use this alpha for the first line search
alpha = 0.15
neg = [[-1.0, 0.0], [0.0, -1.0]]
# Initialize xc
xc = np.zeros((n + 1, 2))
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
        delta_cg[i] = - np.dot(alpha, dfdx(xc[i]))
    else:
        beta = np.dot(gc[i], gc[i]) / np.dot(gc[i - 1], gc[i - 1])
        delta_cg[i] = alpha * np.dot(neg, dfdx(xc[i])) + beta * delta_cg[i - 1]
    xc[i + 1] = xc[i] + delta_cg[i]
ax.plot(xc[:, 0], xc[:, 1], 'y-o',label="CG")

##################################################
# Quasi-Newton method
##################################################
# Number of iterations
n = 8
# Use this alpha for every line search
alpha = np.linspace(0.1, 1.0, n)
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
xq = np.zeros((n + 1, 2))
xq[0] = x_start
# Initialize gradient storage
g = np.zeros((n + 1, 2))
g[0] = dfdx(xq[0])
# Initialize hessian storage
h = np.zeros((n + 1, 2, 2))
h[0] = [[1, 0.0], [0.0, 1]]
h[0] *=5
for i in range(n):

    search_dirn = np.linalg.solve(h[i], g[i])
    # Compute search direction and magnitude (dx)
    #  with dx = -alpha * inv(h) * grad
    delta_xq = -np.dot(1, np.linalg.solve(h[i], g[i]))
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
def general_rank_1_QN(k,f,gradient,c,x_0):
    # B_0 is our HESSIAN approximation (NOT hessian inverse). This is very critical!
    B_0 = [[2.3, -2.50], [-2.5, 7]]
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
        noise = np.random.uniform(500, 1000)
        noise = np.random.uniform((0,1), size=(2,2,))
        print("noise added:")
        print(noise)
        c = y_k
        print(c)
        c = noise@y_k # fix to a fixed method
        print(c)
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

qn_soln , qn_iterates = general_rank_1_QN(8,f,dfdx,None,x_start)
print("rank 1 method returns")
print(qn_soln)
ax.plot(qn_iterates[:, 0], qn_iterates[:, 1], 'c-o',label="QN (1973)")
ax.legend()

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
def general_rank_1_QN_H(k,f,gradient,d,x_0, H_0 = np.linalg.inv([[2.3, -2.50], [-2.5, 7]]), noise=lambda s:0):
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

    s_k = [9999999, 99999999] # junk initialization

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


qn_H1_soln , qn_H1_iterates = general_rank_1_QN_H(8,f,dfdx,"mccormick",x_start)
# Sample call with noise
# qn_H1_soln , qn_H1_iterates = general_rank_1_QN_H(8,f,np_dfdx,"mccormick",x_start, noise = lambda s: np.random.multivariate_normal([0,0],[[1,0],[0,1]]))
print("rank 1 H method returns")
print(qn_H1_soln)
ax.plot(qn_H1_iterates[:, 0], qn_H1_iterates[:, 1], 'm-o',label="QN-H (1973)")
ax.legend()

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
def general_rank_2_QN_H(k,f,gradient,d,x_0, H_0 = np.linalg.inv([[2.3, -2.50], [-2.5, 7]]), noise=lambda s:0):
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
    x_iterates = np.zeros((k + 1, 2))
    x_iterates[0] = x_0

    s_k = [9999999, 99999999] # junk initialization

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


qn_H2_soln , qn_H2_iterates = general_rank_2_QN_H(8,f,dfdx,"BFGS",x_start)
# Sample call with noise
#qn_H2_soln , qn_H2_iterates = general_rank_2_QN_H(8,f,np_dfdx,"BFGS",x_start,noise = lambda s: np.random.multivariate_normal([0,0],[[1,0],[0,1]]))
print("rank 2 H method returns")
print(qn_H2_soln)
ax.plot(qn_H2_iterates[:, 0], qn_H2_iterates[:, 1], marker='o', ls='-', color='lime',label="QN-H Rank-2 (1973)")
ax.legend()

# Save the figure as a PNG
fig.savefig('contour.png')

'''examine spectral norm (induced l2 norm)'''
I = np.array([[2.3, -2.50], [-2.5, 9]])
H = np.array(H)
diff = I - H
H_inverse = np.linalg.inv(H)
prod = np.linalg.norm(H_inverse,ord=2) * np.linalg.norm(diff,ord=2)
print("Norm is " + str(prod )) # tight 1/2, not a constant? (but must be less than one
print(diff)
fig.show()
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

ax.set_xlabel("Iteration")
ax.set_ylabel("L2 norm between current estimate and optimal")
ax.legend()

fig.savefig("iterate residuals")

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

ax.set_xlabel("Iteration")
ax.set_ylabel("L2 norm between current estimate and optimal")
ax.legend()

fig.savefig("log iterate residuals")
#fig.show()
