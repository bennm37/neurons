import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# a = 0.025822
# b= 0.49415
a = 1 
b = 1
parameters = {
    "initial": "uniform",
    "N": 1000,
    "K":1,
    "T": 10,
    "dt": 0.1,
    "a": a,
    "b": b
}
def solve(parameters):
    N = parameters["N"]
    K = parameters["K"]
    T = parameters["T"]
    dt = parameters["dt"]
    a = parameters["a"]
    b = parameters["b"]
    if parameters["initial"] == "uniform":
        theta = np.random.random(N) * 2 * np.pi
        omegas = np.random.uniform(0.9,1.1,N)
    elif parameters["initial"] == "clustered":
        theta = np.random.random(N)*0.2 -0.1
        theta = np.mod(theta,2*np.pi)
        omegas = np.random.uniform(0,2,N)
    elif parameters["initial"] == "no_omega":
        theta = np.random.random(N) * 0.2-0.1
        omegas = np.zeros(N)
    else:
        raise ValueError("Unknown initial condition")
    def F(theta,t):
        return omegas + K/N *np.sum(np.sin(theta[np.newaxis,:] - theta[:,np.newaxis]),axis=1)
    def mexican_hat(deltat,a,b):
        val = 2*a/(np.sqrt(3*b)*np.pi**4)*(1-(deltat/b)**2)*np.exp(-deltat**2/(2*b**2))
        return val
    # integrate using odeint
    # solution = odeint(F, theta, np.arange(0,T,dt))
    theta_results = np.zeros((int(T/dt),N))
    weight_results = np.zeros((int(T/dt),N,N))
    spike_times = np.ones(N)*-1
    weights = np.ones((N,N))*K/N
    for i in range(1,int(T/dt)):
        if i%50 == 0:
            print("Step %d of %d" % (i,int(T/dt)))
            print("Spike times: %s" % spike_times)
        # if np.all(spike_times>0):
        #     deltat = spike_times[np.newaxis,:] - spike_times[:,np.newaxis]
        #     weights += mexican_hat(deltat,a,b)*dt
        theta += dt * F(theta,i*dt)
        spike_times[theta<0] = i*dt
        spike_times[theta>2*np.pi] = i*dt
        weight_results[i,:,:] = weights
        theta_results[i,:] = theta
        theta = np.mod(theta,2*np.pi)
    return theta_results,weight_results

def animate(solution):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    for i in range(solution.shape[0]):
        ax.clear()
        ones = np.ones_like(solution[i,:])
        zeros = np.zeros_like(solution[i,:])
        ax.scatter(solution[i,:],ones,s=1)
        ax.axis('off')
        ax.quiver(zeros,zeros,np.cos(solution[i,:]),np.sin(solution[i,:]),scale=3)
        plt.pause(0.02)
    plt.show()
    
def animate_histogram(solution):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set(xlim=(0,2*np.pi))
    bins = np.linspace(0,2*np.pi,20)
    for i in range(solution.shape[0]):
        ax.clear()
        ax.set(xlim=(0,2*np.pi),ylim=(0,100))
        ax.hist(solution[i,:],bins=bins)
        plt.pause(0.02)
    plt.show()

def animate_weight_matrix(weights):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(weights.shape[0]):
        ax.clear()
        ax.set(xlim=(0,1000),ylim=(0,1000))
        ax.imshow(weights[i,:,:])
        plt.pause(0.02)
    plt.show()

print("Solving...")
theta_results,weight_results = solve(parameters)
print("Animating...")
# animate_weight_matrix(weight_results)
animate(theta_results)



