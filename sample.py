## Generate a contour plot
# Import some other libraries that we'll need
# matplotlib and numpy packages must also be installed
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.optimize

# define objective function
def f(x):
    x1 = x[0]
    x2 = x[1]
    obj = x1 ** 2 - 2.0 * x1 * x2 + 4 * x2 ** 2
    return obj


# define objective gradient
def dfdx(x):
    x1 = x[0]
    x2 = x[1]
    grad = []
    grad.append(2.0 * x1 - 2.0 * x2)
    grad.append(-2.0 * x1 + 8.0 * x2)
    return grad


# define objective gradient
def np_dfdx(x):
    x1 = x[0]
    x2 = x[1]
    grad = []
    grad.append(2.0 * x1 - 2.0 * x2)
    grad.append(-2.0 * x1 + 8.0 * x2)
    return np.array(grad)

# Exact 2nd derivatives (hessian)
H = [[2.0, -2.0], [-2.0, 8.0]]

# Start location
x_start = [-3.0, 3.0]

# Design variables at mesh points
i1 = np.arange(-4.0, 4.0, 0.1)
i2 = np.arange(-4.0, 4.0, 0.1)
x1_mesh, x2_mesh = np.meshgrid(i1, i2)
f_mesh = x1_mesh ** 2 - 2.0 * x1_mesh * x2_mesh + 4 * x2_mesh ** 2

# Create a contour plot
plt.figure()
# Specify contour lines
lines = range(2, 52, 2)
# Plot contours
CS = plt.contour(x1_mesh, x2_mesh, f_mesh, lines)
# Label contours
plt.clabel(CS, inline=1, fontsize=10)
# Add some text to the plot
plt.title('f(x) = x1^2 - 2*x1*x2 + 4*x2^2')
plt.xlabel('x1')
plt.ylabel('x2')
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
plt.plot(xn[:, 0], xn[:, 1], 'k-o', label="Newton")

##################################################
# Steepest descent method
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
plt.plot(xs[:, 0], xs[:, 1], 'g-o', label="GD")

##################################################
# Conjugate gradient method
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
plt.plot(xc[:, 0], xc[:, 1], 'y-o',label="CG")

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

plt.plot(xq[:, 0], xq[:, 1], 'r-o',label="QN")
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
        c = s_k # fix to a fixed method


        # compute the next B_{k+1} iteration
        B_k_and_1 = B_k + np.outer(y_k - np.matmul(B_k,s_k), np.transpose(c)/np.dot(c, s_k))

        # update the matrix:
        B_k = B_k_and_1
        x_k = x_k_and_1

        # logic for checking whether to terminate or not
        not_done = True
        counter += 1
        cond = counter < k and not_done
        x_iterates[counter] = x_k

    return x_k, x_iterates

qn_soln , qn_iterates = general_rank_1_QN(8,f,np_dfdx,None,x_start)
print("rank 1 method returns")
print(qn_soln)
plt.plot(qn_iterates[:, 0], qn_iterates[:, 1], 'c-o',label="QN (1973)")
plt.legend()

# Save the figure as a PNG
plt.savefig('contour.png')


'''examine spectral norm (induced l2 norm)'''
I = np.array([[2.3, -2.50], [-2.5, 9]])
H = np.array(H)
diff = I - H
H_inverse = np.linalg.inv(H)
prod = np.linalg.norm(H_inverse,ord=2) * np.linalg.norm(diff,ord=2)
print("Norm is " + str(prod )) # tight 1/2, not a constant? (but must be less than one
print(diff)
plt.show()
