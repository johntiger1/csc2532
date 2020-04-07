import matplotlib
import autograd.numpy as np  
from autograd import jacobian 
import matplotlib.pyplot as plt
import scipy
import scipy.optimize

from optimizers import *

########## RANDOMNESS ##########
np.random.seed(42)
trials = 100

noise = lambda s : 0
#noise = lambda s: 10000 * np.random.rand(2) - 5000
#noise = lambda s : 5000+10000 * np.random.rand(2) - 5000

def normalize(x):
    return x / np.linalg.norm(x,ord=2)

noise_p=2
noise_C=1
bnd_noise = lambda s : 0


########## PLOTTING PARAMETERS ##########
ymin = -0.25
ymax = 0.25

xmin = -0.25
xmax = 0.25

step = 0.01

title = 'f(x) = x1^2 - 2*x1*x2 + 4*x2^2'

png_prefix = "rand_quad1_"

########## FUNCTION DEFINITION ################
# define objective function
def f(x):
    return x[0] ** 2 - 2.0 * x[0] * x[1] + 4 * x[1] ** 2

def f_sep(x1,x2):
    return f([x1,x2])

dfdx = lambda x : jacobian(f)(np.array(x).astype(float))

hessian = lambda x : jacobian(jacobian(f))(np.array(x).astype(float))
########## INITIALIZATION PARAMETERS ##########

# Note: start [1000,1000], start [[0.5,0.5],[0.5,-0.5]], p=0.5, C=1 is OK

# SOLUTION POSITION
opt_x = np.array([0,0])


H = hessian(np.array(opt_x)) # Exact 2nd derivatives (hessian)

# STARTING PARAMS
x_start = np.array([0.25, 0.25]) # For p = 2, C = 1:
                               #[0.5,0.5] seems to work or be on edge; very poor rate
                               #[12.5,12.5] only just doesn't work on 10 trials
                               #[1,1] breaks on around 100 trials
TEMP_B0 = H + [[0.05,0.05],[0.05,-0.05]] # H_0 is this thing's inverse

# Max iterations
max_iter = 15

# GD ALPHA
GD_alpha = 0.1

# CGD ALPHA
CGD_alpha = 0.1

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
xn = NewtonMethod (max_iter, f, dfdx, hessian, x_start) 

ax.plot(xn[:, 0], xn[:, 1], 'k-o', label="Newton")

print("Newton's method returns")
print(xn[-1,:])


##################################################
# Conjugate gradient method
##################################################
xc = cgd(max_iter, f, dfdx, x_start, CGD_alpha)
ax.plot(xc[:, 0], xc[:, 1], marker='o', ls='-', label="CG")

print("Conjugate Gradient method returns")
print(xc[-1,:])


MODE = "BFGS"
qn_H2_soln , qn_H2_iterates = general_rank_2_QN_H(max_iter,f,dfdx,MODE,x_start,np.linalg.inv(TEMP_B0))
# FOR NOISE USE noise = lambda s: np.random.multivariate_normal([0,0],[[1,0],[0,1]])
print("rank 2 H method \"{}\" returns".format(MODE))
print(qn_H2_soln)
ax.plot(qn_H2_iterates[:, 0], qn_H2_iterates[:, 1], marker='o', ls='-', label="QN-H R2 (1973)")
ax.legend()


# Run bound noise
qn_H2_bnd_noise = np.zeros((trials,max_iter + 1,2)) + opt_x
for i in range(trials):
    _, qn_H2_bnd_noise[i] = general_rank_2_QN_H(max_iter,f,dfdx,MODE,x_start,np.linalg.inv(TEMP_B0), noise = bnd_noise)


avg_qn_H2_bnd_noise = np.average(qn_H2_bnd_noise, axis=0)
print("avg bounded noise rank 2 H method \"{}\" returns".format(MODE))
print(avg_qn_H2_bnd_noise[-1,:])
ax.plot(avg_qn_H2_bnd_noise[:, 0], avg_qn_H2_bnd_noise[:, 1], marker='o', ls='-',label="Avg QN-H R2 Bound Noise")
ax.legend()

# Run unbound noise
qn_H2_unbnd_noise = np.zeros((trials,max_iter + 1,2)) + opt_x
for i in range(trials):
    _, qn_H2_unbnd_noise[i] = general_rank_2_QN_H(max_iter,f,dfdx,MODE,x_start,np.linalg.inv(TEMP_B0), noise = noise)


avg_qn_H2_unbnd_noise = np.average(qn_H2_unbnd_noise, axis=0)
print("avg unbounded noise rank 2 H method \"{}\" returns".format(MODE))
print(avg_qn_H2_unbnd_noise[-1,:])
ax.plot(avg_qn_H2_unbnd_noise[:, 0], avg_qn_H2_unbnd_noise[:, 1], marker='o', ls='-',label="Avg QN-H R2 Unbound Noise")
ax.legend()

contours = ax.contour(x1_mesh, x2_mesh, v_func(x1_mesh, x2_mesh))
ax.clabel(contours , inline=True, fontsize=8)

# Save the figure as a PNG
fig.savefig(png_prefix+'contour.png')
plt.close(fig)

def plot_trials(data,filename,title):
    fig, ax = plt.subplots()
    plt.ylim(ymin, ymax)
    plt.xlim(xmin, xmax)
    ax.contour(x1_mesh, x2_mesh, v_func(x1_mesh, x2_mesh))
    ax.set_title(title)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')

    for i in range(trials):
        ax.plot(data[i,:, 0], data[i,:, 1], marker='o', ls='-')
    #ax.legend()
    contours = ax.contour(x1_mesh, x2_mesh, v_func(x1_mesh, x2_mesh))
    ax.clabel(contours , inline=True, fontsize=8)
    fig.savefig(png_prefix+filename)
    plt.close(fig)

plot_trials(qn_H2_bnd_noise,'_bnd_nse_contour.png',title+" with bounded noise trials")
plot_trials(qn_H2_unbnd_noise,'_unbd_nse_contour.png',title+" with unbound noise trials")

# qn_iterates-
'''
Computes the residuals
'''
def compute_residuals(iterates, opt_x):
    return np.linalg.norm(iterates-opt_x.transpose(), ord=2, axis=1)


fig,ax= plt.subplots()
ax.set_title("L2-norm between iterate and optimal")

ax.plot(np.arange(0, len(xn)), compute_residuals(xn, opt_x), label="Newton's method", color='k')
ax.plot(np.arange(0, len(xc)), compute_residuals(xc, opt_x), label="CG")
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

ax.plot(np.arange(0, len(xn)), compute_residuals(xn, opt_x), label="Newton's method", color='k')
ax.plot(np.arange(0, len(xc)), compute_residuals(xc, opt_x), label="CG")
ax.plot(np.arange(0, len(qn_H2_iterates)), compute_residuals(qn_H2_iterates, opt_x), label="QN-H Rank-2 (1973)")
ax.plot(np.arange(0, len(avg_qn_H2_bnd_noise)), compute_residuals(avg_qn_H2_bnd_noise, opt_x), label="Avg QN-H Rank-2 Bnd Noise (1973)")
ax.plot(np.arange(0, len(avg_qn_H2_unbnd_noise)), compute_residuals(avg_qn_H2_unbnd_noise, opt_x), label="Avg QN-H Rank-2 Unbnd Noise (1973)")

ax.set_xlabel("Iteration")
ax.set_ylabel("L2 norm between current estimate and optimal")
ax.legend()

fig.savefig(png_prefix+"log iterate residuals")
#fig.show()
plt.close(fig)

def plot_residual_trials(data,filename,title):
    fig,ax= plt.subplots()
    ax.set_title(title)
    ax.set_yscale('log')


    for i in range(trials):
        #ax.plot(data[i,:, 0], data[i,:, 1], marker='o', ls='-',label="{} {}".format(label, i))
        ax.plot(np.arange(0, max_iter+1), compute_residuals(data[i,:,:], opt_x))
    #ax.legend()
    fig.savefig(png_prefix+filename)
    plt.close(fig)

plot_residual_trials(qn_H2_unbnd_noise,'_unbd_nse_iter.png',"L2-norm between unbounded noise iterate and optimal")
plot_residual_trials(qn_H2_bnd_noise,'_bd_nse_iter.png',"L2-norm between bounded noise iterate and optimal")
