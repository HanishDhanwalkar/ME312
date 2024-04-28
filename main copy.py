from backend import Simulation
from frontend import OptimizerPlot, setThreshold, saveAnimation
import numpy as np


class TopOpt:
    def __init__(self, corners: np.ndarray, 
                 meshDensity: int = 100,
                 E0: float = 190e9,
                 Emin:float = 100,
                 nu: float = 0.3,
                 penalty: int = 4) -> None:
        self.simulation = Simulation(corners, meshDensity)
        self.W = corners[:,0].max() - corners[:,0].min()
        self.L = corners[:,1].max() - corners[:,1].min()
        self.density = np.ones(self.simulation.domain.topology.index_map(2).size_local)
        self.elemLocs = self.simulation.locs
        self.numElems = len(self.density)
        self.simulation.createFunctions()
        self.E = E0
        self.Emin = Emin
        self.nu = nu
        self.penalty = penalty

    def createFixedBoundaries(self, locFunctions: list):
        self.fixedBoundaries = locFunctions
        for locFunction in locFunctions:
            self.simulation.fixedBoundary(locFunction)

    def applyForces(self, forces: dict[tuple, tuple]):
        self.forces = forces
        forceTuples = []
        for loc in forces:
            forceTuples.append((loc, forces[loc]))
        self.simulation.applyForce(forceTuples)

    def objectiveFunction(self):
        self.simulation.density.interpolate(lambda _: self.density)
        self.simulation.constituentEqns(self.Emin, self.E, self.penalty, self.nu)
        self.simulation.solve()
        self.simulation.updateStress()
        comp = self.simulation.compliance()
        return comp
    
    def normalize(self, vec: np.ndarray):
        return (vec-vec.min())/(vec.max()-vec.min())
    
    def percentileMask(self, vec: np.ndarray, p: float = 50, defaultVal: float = 0):
        mask = vec >= np.percentile(vec, p)
        return np.where(mask, vec, defaultVal)
    
    def gradient(self):
        C = self.simulation.complianceArr.vector.array 
        C = C * self.simulation.domain.h(2,np.arange(len(C), dtype='int32'))
        num = -self.penalty*(self.E-self.Emin)*self.density**(self.penalty-1)*C 
        denom = self.Emin + self.density**self.penalty*(self.E-self.Emin)
        grad = self.normalize(num/denom)**300
        return self.percentileMask(grad)
    
    def gaussianFilter(self, vec: np.ndarray, R: float = 0.3):
        sigma = R/3
        def filter(i):
            mask = (self.elemLocs[i,0]-R<=self.elemLocs[:,0]) & (self.elemLocs[:,0]<=self.elemLocs[i,0]+R) & (self.elemLocs[i,1]-R<=self.elemLocs[:,1]) & (self.elemLocs[:,1]<=self.elemLocs[i,1]+R)
            y = np.where(mask, vec, 0)
            weight = np.exp(-((self.elemLocs[:,0]-self.elemLocs[i,0])**2+(self.elemLocs[:,1]-self.elemLocs[i,1])**2)/(2*sigma**2))
            weight = weight/weight.sum()
            return (y*weight).sum()
        filter = np.vectorize(filter)
        return filter(np.arange(self.numElems))
    
    def optimize(self, numIter: int = 50, 
                       targetVol: float = 0.5, 
                       saveResult: bool = True, 
                       animate: bool = False):
        vPrev = 2
        history = []
        plotter = OptimizerPlot(numIter, targetVol)
        plotter.init()        
        for i in range(numIter):
            comp = self.objectiveFunction()
            vol = self.density.mean()
            plotter.update(vol, comp)
            history.append(self.density)
            print(f"Iteration: {i+1}  Volume Fraction: {vol}, Compliance: {comp}")
            if vol <= targetVol or abs(vol-vPrev)<0.0001:
                break
            vPrev = vol
            grad = self.gradient()
            self.density = np.maximum(0.01, self.density-0.1*grad)
            self.density = self.gaussianFilter(self.density)
            self.density = self.normalize(self.density)
            self.density = self.percentileMask(self.density, 10, 0.01)

        plotter.stop()
        self.density = setThreshold(self.density, self.elemLocs)
        
        print(f'Optimization Completed. \nFinal Compliance = {self.objectiveFunction()}')
        self.simulation.displayDistribution()
        self.simulation.displayResult()

        if saveResult:
            result = np.empty((3, self.numElems))
            result[:2, :] = self.elemLocs[:, :2].T
            result[2, :] = self.density
            np.save('result', result)

        if animate:
            saveAnimation(np.array(history), self.elemLocs)


corners = np.array([[0, 0],
           [3, 0],
           [9, 5],
           [15, 4],
           [16, 6],
           [8, 8],
           [0, 4]])
# topBoundary = lambda x: np.isclose(x[1], 15)
fixedBoundary1 = lambda x: np.isclose(x[1], 0) #& (x >= 0) & (x <=3) 
forces = {(0, 4): (1e3, 0), (15,4): (0, -1e3)}
opt = TopOpt(corners, meshDensity=70)
opt.createFixedBoundaries([fixedBoundary1])
opt.applyForces(forces)
opt.optimize(targetVol=0.3,animate=False, numIter= 10)