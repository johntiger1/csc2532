import matplotlib
import autograd.numpy as np  
from autograd import jacobian 
import matplotlib.pyplot as plt
import scipy
import scipy.optimize



########## RANDOMNESS ##########
np.random.seed(42)
trials = 5

noise = lambda s : 100 * np.random.rand(2) - 50

def normalize(x):
    return x / np.linalg.norm(x,ord=2)

noise_p=2 #NOTE: p=1 does not behave nicely; guessing I'd say p=2 b/c rosen is degree 4
noise_C=125 #NOTE: C = 250 we begin to see breakdown (presumably neighbourhood is even smaller at this point)
bnd_noise = lambda s : noise_C * normalize(noise(s))*(np.linalg.norm(s,ord=2)**noise_p)


########## PLOTTING PARAMETERS ##########
ymin = 0.998
ymax = 1.001

xmin = 0.999
xmax = 1.0015

step = 0.0001

title = 'Rosenbrock function with optima (1,1)'

png_prefix = "noisy_rosen1_"

########## FUNCTION DEFINITION ################
# define objective function
def f(x):
    return 100*(x[1] - x[0]**2)**2 + (x[0] - 1)**2

def f_sep(x1,x2):
    return f([x1,x2])

dfdx = lambda x : jacobian(f)(np.array(x).astype(float))

hessian = lambda x : jacobian(jacobian(f))(np.array(x).astype(float))
########## INITIALIZATION PARAMETERS ##########

# SOLUTION POSITION
opt_x = np.array([1,1])


H = hessian(np.array(opt_x)) # Exact 2nd derivatives (hessian)

# STARTING PARAMS
x_start = np.array([1.001, 0.999])
TEMP_B0 = H #+ [[0.05,0.05],[0.05,-0.05]] # H_0 is this thing's inverse

# Max iterations
max_iter = 8

# GD ALPHA
GD_alpha = 0.0001

# CGD ALPHA
CGD_alpha = 0.0001

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


# Run bound noise
qn_H2_bnd_noise = np.zeros((trials,max_iter + 1,2)) + opt_x
for i in range(trials):
    _, qn_H2_bnd_noise[i] = general_rank_2_QN_H(max_iter,f,dfdx,MODE,x_start, noise = bnd_noise)


avg_qn_H2_bnd_noise = np.average(qn_H2_bnd_noise, axis=0)
print("avg bounded noise rank 2 H method \"{}\" returns".format(MODE))
print(avg_qn_H2_bnd_noise[-1,:])
ax.plot(avg_qn_H2_bnd_noise[:, 0], avg_qn_H2_bnd_noise[:, 1], marker='o', ls='-', color='r',label="Avg QN-H R2 Bound Noise")
ax.legend()

# Run unbound noise
qn_H2_unbnd_noise = np.zeros((trials,max_iter + 1,2)) + opt_x
for i in range(trials):
    _, qn_H2_unbnd_noise[i] = general_rank_2_QN_H(max_iter,f,dfdx,MODE,x_start, noise = noise)


avg_qn_H2_unbnd_noise = np.average(qn_H2_unbnd_noise, axis=0)
print("avg unbounded noise rank 2 H method \"{}\" returns".format(MODE))
print(avg_qn_H2_unbnd_noise[-1,:])
ax.plot(avg_qn_H2_unbnd_noise[:, 0], avg_qn_H2_unbnd_noise[:, 1], marker='o', ls='-', color='c',label="Avg QN-H R2 Unbound Noise")
ax.legend()

contours = ax.contour(x1_mesh, x2_mesh, v_func(x1_mesh, x2_mesh))
ax.clabel(contours , inline=True, fontsize=8)

# Save the figure as a PNG
fig.savefig(png_prefix+'contour.png')
plt.close(fig)

def plot_trials(data,filename,label):
    fig, ax = plt.subplots()
    plt.ylim(ymin, ymax)
    plt.xlim(xmin, xmax)
    ax.contour(x1_mesh, x2_mesh, v_func(x1_mesh, x2_mesh))
    ax.set_title(title)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')

    for i in range(trials):
        ax.plot(data[i,:, 0], data[i,:, 1], marker='o', ls='-',label="{} {}".format(label, i))
    ax.legend()
    contours = ax.contour(x1_mesh, x2_mesh, v_func(x1_mesh, x2_mesh))
    ax.clabel(contours , inline=True, fontsize=8)
    fig.savefig(png_prefix+filename)
    plt.close(fig)

plot_trials(qn_H2_bnd_noise,'_bnd_nse_contour.png',"Bound Noise Trial")
plot_trials(qn_H2_unbnd_noise,'_unbd_nse_contour.png',"Unbound Noise Trial")

# qn_iterates-
'''
Computes the residuals
'''
def compute_residuals(iterates, opt_x):
    return np.linalg.norm(iterates-opt_x.transpose(), ord=2, axis=1)


fig,ax= plt.subplots()
ax.set_title("L2-norm between iterate and optimal")

ax.plot(np.arange(0, len(xc)), compute_residuals(xc, opt_x), label="CG")
ax.plot(np.arange(0, len(xn)), compute_residuals(xn, opt_x), label="Newton's method")
ax.plot(np.arange(0, len(qn_H2_iterates)), compute_residuals(qn_H2_iterates, opt_x), label="QN-H Rank-2 (1973)")
ax.plot(np.arange(0, len(avg_qn_H2_bnd_noise)), compute_residuals(avg_qn_H2_bnd_noise, opt_x), label="Avg QN-H Rank-2 Bnd Noise (1973)")
ax.plot(np.arange(0, len(avg_qn_H2_unbnd_noise)), compute_residuals(avg_qn_H2_unbnd_noise, opt_x), label="Avg QN-H Rank-2 Unbnd Noise (1973)")

ax.set_xlabel("Iteration")
ax.set_ylabel("L2 norm between current estimate and optimal")
ax.legend()

fig.savefig(png_prefix+"iterate residuals")

fig,ax= plt.subplots()
ax.set_title("L2-norm between iterate and optimal")
ax.set_yscale('log') # Change to log-scale

ax.plot(np.arange(0, len(xc)), compute_residuals(xc, opt_x), label="CG")
ax.plot(np.arange(0, len(xn)), compute_residuals(xn, opt_x), label="Newton's method")
ax.plot(np.arange(0, len(qn_H2_iterates)), compute_residuals(qn_H2_iterates, opt_x), label="QN-H Rank-2 (1973)")
ax.plot(np.arange(0, len(avg_qn_H2_bnd_noise)), compute_residuals(avg_qn_H2_bnd_noise, opt_x), label="Avg QN-H Rank-2 Bnd Noise (1973)")
ax.plot(np.arange(0, len(avg_qn_H2_unbnd_noise)), compute_residuals(avg_qn_H2_unbnd_noise, opt_x), label="Avg QN-H Rank-2 Unbnd Noise (1973)")

ax.set_xlabel("Iteration")
ax.set_ylabel("L2 norm between current estimate and optimal")
ax.legend()

fig.savefig(png_prefix+"log iterate residuals")
#fig.show()
plt.close(fig)

def plot_residual_trials(data,filename,label):
    fig,ax= plt.subplots()
    ax.set_title("L2-norm between iterate and optimal")
    ax.set_yscale('log')


    for i in range(trials):
        #ax.plot(data[i,:, 0], data[i,:, 1], marker='o', ls='-',label="{} {}".format(label, i))
        ax.plot(np.arange(0, max_iter+1), compute_residuals(data[i,:,:], opt_x), label="{} {}".format(label, i))
    ax.legend()
    fig.savefig(png_prefix+filename)
    plt.close(fig)

plot_residual_trials(qn_H2_unbnd_noise,'_unbd_nse_iter.png',"Unbound Noise Trial")