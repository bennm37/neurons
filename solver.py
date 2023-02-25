import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib import animation
import numpy as np
import os
import matplotlib.cm as cm
# a = 0.025822
# b= 0.49415
a = 112
b = 174
parameters = {
    "initial": "normal",
    "N": 60,
    "K":1,
    "mean_weight":10,
    "std_weight":0,
    "T": 40,
    "dt": 0.1,
    "a": a,
    "b": b
}
def solve(parameters):
    K = parameters["K"]
    N = parameters["N"]
    T = parameters["T"]
    dt = parameters["dt"]
    a = parameters["a"]
    b = parameters["b"]
    mw = parameters["mean_weight"]
    sw = parameters["std_weight"]
    if parameters["initial"] == "uniform":
        theta = np.random.random(N) * 2 * np.pi
        omegas = np.sort(np.random.uniform(0.9,1.1,N))
    if parameters["initial"] == "normal":
        theta = np.random.random(N) * 2 * np.pi
        omegas = np.sort(np.random.normal(0.5,0.5,N))
    elif parameters["initial"] == "clustered":
        theta = np.random.random(N)*0.2 -0.1
        theta = np.mod(theta,2*np.pi)
        omegas = np.sort(np.random.uniform(0,2,N))
    elif parameters["initial"] == "no_omega":
        theta = np.random.random(N) * 0.2-0.1
        omegas = np.sort(np.zeros(N))
    else:
        raise ValueError("Unknown initial condition")
    def F(theta,t):
        return omegas + K/N *np.sum(np.sin(theta[np.newaxis,:] - theta[:,np.newaxis]),axis=1)
    def mexican_hat(deltat,a,b):
        val = 2*a/(np.sqrt(3*b)*np.pi**4)*(1-(deltat/b)**2)*np.exp(-deltat**2/(2*b**2))
        return val
    theta_results = np.zeros((int(T/dt),N))
    weight_results = np.zeros((int(T/dt),N,N))
    weights = np.random.normal(mw,sw,(N,N))
    spike_times = -theta/omegas
    for i in range(0,int(T/dt)):
        if i%50 == 0:
            print("Step %d of %d" % (i,int(T/dt)))
            print("Average spike time: %s" % np.mean(spike_times))
        deltat = spike_times[np.newaxis,:] - spike_times[:,np.newaxis]
        weights += mexican_hat(deltat,a,b)*dt
        theta += dt * F(theta,i*dt)
        spike_times[theta<0] = i*dt
        spike_times[theta>2*np.pi] = i*dt
        weight_results[i,:,:] = weights
        theta = np.mod(theta,2*np.pi)
        theta_results[i,:] = theta
    return omegas,theta_results,weight_results

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

def animate2(solution,figax=None,stride=1):
    if figax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
    else:
        fig,ax = figax
    cmap = cm.get_cmap('rainbow')
    colors = cmap(np.linspace(0,1,solution.shape[1]))
    ones = np.ones_like(solution[0,:])
    zeros = np.zeros_like(solution[0,:])
    # create a scatter plot of solution where every point is colored differently
    scat = ax.scatter(solution[0,:],ones,s=20,c = colors)
    scatred = ax.scatter(solution[0,0],1,s=19)
    quiv = ax.quiver(zeros,zeros,np.cos(solution[0,:]),np.sin(solution[0,:]),scale=3)
    ax.axis('off')
    def update(j):
        i = j*stride
        scat.set_offsets(np.array([solution[i,:],ones]).T)
        scatred.set_offsets([solution[i,0],1])
        quiv.set_UVC(np.cos(solution[i,:]),np.sin(solution[i,:]))
    N_frames = solution.shape[0]//stride
    anim = animation.FuncAnimation(fig,update,N_frames)
    return fig,ax,anim

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
    ax.imshow(weights[0,:,:],vmin=0,vmax=np.max(weights),cmap="coolwarm")
    def update(i):
        ax.clear()
        ax.imshow(weights[i,:,:],vmin=0,vmax=np.max(weights),cmap="coolwarm")
    N_frames = weights.shape[0]
    anim = animation.FuncAnimation(fig,update,N_frames)
    return fig,ax,anim

def mexican_hat(deltat,a,b):
    val = 2*a/(np.sqrt(3*b)*np.pi**4)*(1-(deltat/b)**2)*np.exp(-deltat**2/(2*b**2))
    return val
t = np.linspace(-np.pi,np.pi,100)
plt.plot(t,mexican_hat(t,parameters["a"],parameters["b"]))
plt.show()

print("Solving...")
omega,theta_results,weight_results = solve(parameters)
print("Animating...")
fig,ax,anim =animate_weight_matrix(weight_results)
anim.save("weight_test2.mp4",fps=30)
print("Animating...")
fig,ax,anim =animate2(theta_results)
anim.save("weight_test_circle.mp4",fps=30)


filename = 'sweepbPlasticity'
os.mkdir(filename)
for i in range(10):
    b = np.pi*i/10
    parameters["b"] = b
    print("Solving...")
    omegas,theta_results,weight_results = solve(parameters)
    fig,ax,anim = animate_weight_matrix(solution=theta_results,stride=1)
    anim.save(f"{filename}/test{b}.mp4",fps=30)
    print(f"Saved K={b}")
