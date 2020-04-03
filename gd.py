
'''
Generic optimization code.
Input is samples, and then we will find a minimum of it!

Actually, we will assume we have the true gradient and the Hessian too
'''
def optimize(samples, ):
    pass

'''

Sample code for simple GD
'''

def gradient_descent(x0, gradient, steps=100):
    xk = x0
    for i in range(steps):
        xk = xk - gradient(xk)

    return xk

