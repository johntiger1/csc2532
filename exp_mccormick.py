import matplotlib
import autograd.numpy as np  
from autograd import jacobian 
import matplotlib.pyplot as plt
import scipy
import scipy.optimize

from optimizers import *

########## PLOTTING PARAMETERS ##########
ymin = -2
ymax = 0.1

xmin = -0.8
xmax = 0.2

step = 0.1

title = "McCormick's Function"

png_prefix = "McCormick_"
########## FUNCTION DEFINITION ################
# define objective function
def f(x):
    return np.sin(x[0] + x[1]) + (x[0]-x[1])**2 - 1.5*x[0] + 2.5*x[1] +1

def f_sep(x1,x2):
    return f([x1,x2])

dfdx = lambda x : jacobian(f)(np.array(x).astype(float))

hessian = lambda x : jacobian(jacobian(f))(np.array(x).astype(float))
########## INITIALIZATION PARAMETERS ##########
# starting from 2,2 [2.5943947  1.59439441]
# weird: starting from closer: [-0.54719758 -1.54719758]
num_opt_soln = scipy.optimize.minimize(fun= f, x0= np.array([-0.54719, -1.54719]), method="dogleg",jac=dfdx, hess=hessian)

print("Numerical optimal solution:")
print(num_opt_soln .x)

# SOLUTION POSITION
opt_x = num_opt_soln.x


H = hessian(np.array(opt_x)) # Exact 2nd derivatives (hessian)

# STARTING PARAMS
x_start = np.array([0, 0])
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
xn = NewtonMethod (max_iter, f, dfdx, hessian, x_start)

ax.plot(xn[:, 0], xn[:, 1], 'k-o', label="Newton")

print("Newton's method returns")
print(xn[-1,:])

##################################################
# Steepest descent method
##################################################
xs = gd(max_iter,f,dfdx,x_start,GD_alpha)
ax.plot(xs[:, 0], xs[:, 1], marker='o', ls='-',  label="GD")

print("Gradient Descent returns")
print(xs[-1,:])

##################################################
# Conjugate gradient method
##################################################
xc = cgd(max_iter, f, dfdx, x_start, CGD_alpha)
ax.plot(xc[:, 0], xc[:, 1], marker='o', ls='-', label="CG")

print("Conjugate Gradient method returns")
print(xc[-1,:])

##################################################
# Quasi-Newton method
##################################################
xq = alleged_BFGS(max_iter, f, dfdx,x_start, TEMP_B0, BFGS_alpha)

ax.plot(xq[:, 0], xq[:, 1], marker='o', ls='-', label="QN")
print("BFGS method returns")
print(xq[-1,:])





qn_soln , qn_iterates = general_rank_1_QN(max_iter,f,dfdx,None,x_start,TEMP_B0)
print("rank 1 method returns")
print(qn_soln)
ax.plot(qn_iterates[:, 0], qn_iterates[:, 1], marker='o', ls='-', label="QN (1973)")
ax.legend()


qn_2B_soln , qn_2B_iterates = general_rank_2_QN(max_iter,f,dfdx,None,x_start, TEMP_B0)
print("rank 2 method returns")
print(qn_2B_soln )
ax.plot(qn_2B_iterates [:, 0], qn_2B_iterates [:, 1], marker='o', ls='-', label="QN2 B (1973)")

MODE = "mccormick"
qn_H1_soln , qn_H1_iterates = general_rank_1_QN_H(max_iter,f,dfdx,MODE,x_start,np.linalg.inv(TEMP_B0))
# FOR NOISE USE noise = lambda s: np.random.multivariate_normal([0,0],[[1,0],[0,1]])
print("rank 1 H method \"{}\" returns".format(MODE))
print(qn_H1_soln)
ax.plot(qn_H1_iterates[:, 0], qn_H1_iterates[:, 1], marker='o', ls='-', label="QN-H (1973)")
ax.legend()


MODE = "BFGS"
qn_H2_soln , qn_H2_iterates = general_rank_2_QN_H(max_iter,f,dfdx,MODE,x_start,np.linalg.inv(TEMP_B0))
# FOR NOISE USE noise = lambda s: np.random.multivariate_normal([0,0],[[1,0],[0,1]])
print("rank 2 H method \"{}\" returns".format(MODE))
print(qn_H2_soln)
ax.plot(qn_H2_iterates[:, 0], qn_H2_iterates[:, 1], marker='o', ls='-', label="QN-H R2 (1973)")
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

ax.plot(np.arange(0, len(xn)), compute_residuals(xn, opt_x), label="Newton's method", color="k")
ax.plot(np.arange(0, len(qn_iterates)), compute_residuals(xs, opt_x), label="GD")
ax.plot(np.arange(0, len(qn_iterates)), compute_residuals(xc, opt_x), label="CG")
ax.plot(np.arange(0, len(qn_iterates)), compute_residuals(xq, opt_x), label="BFGS")
ax.plot(np.arange(0, len(qn_iterates)), compute_residuals(qn_iterates, opt_x), label="QN-B (1973)")
ax.plot(np.arange(0, len(qn_2B_iterates)), compute_residuals(qn_2B_iterates, opt_x), label="QN-B Rank-2 (1973)")
ax.plot(np.arange(0, len(qn_H1_iterates)), compute_residuals(qn_H1_iterates, opt_x), label="QN-H (1973)")
ax.plot(np.arange(0, len(qn_H2_iterates)), compute_residuals(qn_H2_iterates, opt_x), label="QN-H Rank-2 (1973)")



ax.set_xlabel("Iteration")
ax.set_ylabel("L2 norm between current estimate and optimal")
ax.legend()

fig.savefig(png_prefix+"iterate residuals")

fig,ax= plt.subplots()
ax.set_title("Logscale L2-norm between iterate and optimal")
ax.set_yscale('log') # Change to log-scale

ax.plot(np.arange(0, len(xn)), compute_residuals(xn, opt_x), label="Newton's method", color="k")
ax.plot(np.arange(0, len(qn_iterates)), compute_residuals(xs, opt_x), label="GD")
ax.plot(np.arange(0, len(qn_iterates)), compute_residuals(xc, opt_x), label="CG")
ax.plot(np.arange(0, len(qn_iterates)), compute_residuals(xq, opt_x), label="BFGS")
ax.plot(np.arange(0, len(qn_iterates)), compute_residuals(qn_iterates, opt_x), label="QN-B (1973)")
ax.plot(np.arange(0, len(qn_2B_iterates)), compute_residuals(qn_2B_iterates, opt_x), label="QN-B Rank-2 (1973)")
ax.plot(np.arange(0, len(qn_H1_iterates)), compute_residuals(qn_H1_iterates, opt_x), label="QN-H (1973)")
ax.plot(np.arange(0, len(qn_H2_iterates)), compute_residuals(qn_H2_iterates, opt_x), label="QN-H Rank-2 (1973)")

ax.set_xlabel("Iteration")
ax.set_ylabel("L2 norm between current estimate and optimal")
ax.legend()

fig.savefig(png_prefix+"log iterate residuals")
#fig.show()
