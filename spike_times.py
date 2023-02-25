import numpy as np  
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.cm as cm

def mexican_hat(deltat,a,b):
    val = 2*a/(np.sqrt(3*b)*np.pi**4)*(1-(deltat/b)**2)*np.exp(-deltat**2/(2*b**2))
    return val

N = 1000
theta = np.random.random(N) * 2 * np.pi
omegas = np.sort(np.random.normal(0.5,0.5,N))
spike_times = -theta/omegas
a = 100
b = 100
t = np.linspace(-5*b,5*b,1000)
deltat = spike_times[:,np.newaxis] - spike_times[np.newaxis,:]
deltat = deltat.flatten()
# Set up figure
fig,ax = plt.subplots(1,3)
fig.set_size_inches(8,5)

# Mexican hat function
line,= ax[0].plot(t,mexican_hat(t,a,b))
ax[0].set(xlabel="Time",ylabel="Weight",title="Mexican Hat")

# Colored spike times histogram
bins_spike = np.linspace(-100,100,20)
n,bins1,patches = ax[1].hist(spike_times,bins=bins_spike,density=True)
col = mexican_hat(bins_spike,a,b)
cmap = cm.get_cmap("coolwarm")
for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', cmap(c))
ax[1].set(xlabel="Spike Times",ylabel="Count",title="Spike Times Histogram")
# Mexican hat histogram
bins_hat = np.linspace(-0.25,0.75,20)
ax[2].hist(mexican_hat(deltat,a,b),bins=bins_hat,density=True)
# add space for sliders
fig.subplots_adjust(bottom=0.25)
# add an ax for the sliders
axa = fig.add_axes([0.15, 0.12, 0.65, 0.03])
slidera = Slider(axa, 'a', 0.1, 200.0, valinit=a)
def update_a(val):
    a = val
    b = sliderb.val
    print("a = %f" % a)
    # line.set_data(t,mexican_hat(t,a,b))
    ax[0].cla()
    t = np.linspace(-5*b,5*b,100)
    ax[0].plot(t,mexican_hat(t,a,b))
    ax[2].cla()
    ax[2].hist(mexican_hat(deltat,a,b),bins=bins_hat,density=True)
    fig.canvas.draw_idle()
slidera.on_changed(update_a)
# add a slider for b
axb = fig.add_axes([0.15, 0.05, 0.65, 0.03])
sliderb = Slider(axb, 'b', 0.1, 200.0, valinit=b)
def update_b(val):
    a = slidera.val
    b = val
    ax[0].cla()
    ax[1].cla()
    bins_spike = np.linspace(-100,100,20)
    n,bins1,patches = ax[1].hist(spike_times,bins=bins_spike,density=True)
    col = mexican_hat(bins_spike,a,b)
    cmap = cm.get_cmap("coolwarm")
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cmap(c))
    ax[1].set(xlabel="Spike Times",ylabel="Count",title="Spike Times Histogram")
    ax[2].cla()
    t = np.linspace(-5*b,5*b,100)   
    ax[0].plot(t,mexican_hat(t,a,b))
    # line.set_data(t,mexican_hat(t,a,b))
    ax[2].hist(mexican_hat(deltat,a,b),bins=bins_hat,density=True)
    fig.canvas.draw_idle()
sliderb.on_changed(update_b)
# add the sliders below ax[2]
plt.show()