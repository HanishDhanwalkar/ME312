import matplotlib.pyplot as plt
import matplotlib.widgets as wid
from matplotlib.animation import FuncAnimation
import numpy as np

plt.rcParams['animation.ffmpeg_path'] = "/mnt/d/files/ffmpeg/bin/ffmpeg.exe"

class OptimizerPlot:
    def __init__(self, numIter: int, volumeThreshold: float):
        self.n = numIter
        self.volumes = []
        self.comps = []
        self.i = 0
        self.volThresh = volumeThreshold*100

    def init(self):
        plt.ion()
        fig, axes = plt.subplots(1, 2, figsize=(10,5))
        axes = axes.flatten()
        fig.canvas.draw()
        self.figure = fig
        
        self.volumePlot, = axes[0].plot([], [], color = 'b', marker='.')
        axes[0].set_xlim([0, self.n])
        axes[0].set_ylim([self.volThresh-10, 101])
        axes[0].set_title('Volume Fraction', fontweight = 'bold', fontsize=16)
        axes[0].set_xlabel('Iterartion')
        axes[0].set_ylabel('Volume Fraction (%)')
        axes[0].hlines(self.volThresh, 0, self.n, color='r', zorder=2)

        self.compPlot,  = axes[1].plot([], [], color = 'b', marker='.')
        axes[1].set_xlim([0, self.n])
        axes[1].set_ylim(bottom = 0, auto = True)
        axes[1].set_title('Compliance', fontweight = 'bold', fontsize=16)
        axes[1].set_xlabel('Iterartion')
        axes[1].set_ylabel('Compliance')

    def update(self, volume, compliance):
        self.volumes.append(volume*100)
        self.comps.append(compliance)
        self.i += 1
        self.volumePlot.set_data([np.arange(self.i), self.volumes])
        self.compPlot.set_data([np.arange(self.i), self.comps])
        self.figure.canvas.flush_events()

    def stop(self):
        plt.ioff()
        plt.show()
        plt.pause(2)
        plt.close()


def adaptiveMeanThresholding(vec: np.ndarray, elemLocs: np.ndarray, R: float = 5.0, c: float = 0.1):
    dNew = np.zeros_like(vec)
    for i in range(len(vec)):
        mask = (elemLocs[i,0]-R<=elemLocs[:,0]) & (elemLocs[:,0]<=elemLocs[i,0]+R) & (elemLocs[i,1]-R<=elemLocs[:,1]) & (elemLocs[:,1]<=elemLocs[i,1]+R)
        idxs = np.argwhere(mask)
        if vec[i] >= max(0.01, np.mean(vec[idxs])- c):
            dNew[i] = 1
    return dNew

def setThreshold(density: np.ndarray, coords: np.ndarray):
    fig, ax = plt.subplots(figsize=(15,10))
    ax.set_aspect('equal')
    scat = ax.scatter(coords[:,0], coords[:,1], marker='o')
    scat.set_cmap('gray')
    scat.set_array(1-density)
    optDensity = density.copy()

    plt.subplots_adjust(bottom=0.3)
    ax_R = plt.axes([0.25, 0.2, 0.65, 0.03])
    ax_c = plt.axes([0.25, 0.1, 0.65, 0.03])

    RSlider = wid.Slider(ax_R, "R", 0.1, 7, 5)
    cSlider = wid.Slider(ax_c, 'c', 0, 0.5, 0.2)

    def update(val):
        R = RSlider.val
        c = cSlider.val
        scat.set_array(1-adaptiveMeanThresholding(density, coords, R, c))
        fig.canvas.draw_idle()
    
    RSlider.on_changed(update)
    cSlider.on_changed(update)
    
    axBtn = plt.axes([0.8, 0.01, 0.1, 0.04])
    setBtn = wid.Button(axBtn, 'Set', hovercolor='0.975')

    def set(event):
        nonlocal optDensity
        R = RSlider.val
        c = cSlider.val
        optDensity = adaptiveMeanThresholding(density, coords, R, c)
        plt.close()

    setBtn.on_clicked(set)

    plt.show()
    return optDensity

def saveAnimation(history: np.ndarray, coords: np.ndarray):
    fig, ax = plt.subplots(figsize=(15,10))
    ax.set_aspect('equal')
    scat = ax.scatter(coords[:,0], coords[:,1], marker='o')
    scat.set_cmap('gray')
    nFrames = history.shape[0]
    def animate(i):
        nonlocal scat, history
        scat.set_array(1-history[i+1,:])
        return scat,
    
    anim = FuncAnimation(fig, animate, range(nFrames-1), interval=250)
    anim.save('imgs/descent.mp4', writer='ffmpeg')
    print('Animation saved in descent.mp4')