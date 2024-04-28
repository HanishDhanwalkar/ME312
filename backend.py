"""
312

Author: Vighnesh Nayak
Date: 07 Mar 2024
Github: https://github.com/Kugelblitz25
"""
from mpi4py import MPI
import dolfinx
import dolfin
import pyvista
from dolfinx.fem import (VectorFunctionSpace, 
                         FunctionSpace, 
                         Function, 
                         Constant,
                         locate_dofs_topological,
                         dirichletbc, Expression,
                         assemble_scalar, form)
from dolfinx.mesh import locate_entities_boundary, compute_midpoints
from ufl import (TrialFunction,
                 TestFunction, Measure, dot,
                 sym, grad, dx, inner, tr,
                 nabla_div, Identity, sqrt)
from mshr import Polygon, generate_mesh
from dolfinx.fem.petsc import LinearProblem
import numpy as np

class Simulation:
    def __init__(self, corners, meshDensity = 15) -> None:
        self.corners = corners
        self.domain = self.createPolygonalMesh(corners, meshDensity)
        self.V = VectorFunctionSpace(self.domain, ("Lagrange", 1))
        self.U = FunctionSpace(self.domain, ("DG", 0))
        self.bcs = []

    def createPolygonalMesh(self, corners, meshDensity=15):
        corners = list(map(dolfin.Point, corners))
        domain = Polygon(corners)
        domain = generate_mesh(domain, meshDensity)

        with dolfin.XDMFFile(MPI.COMM_WORLD, "mesh.xdmf") as file:
            file.write(domain)

        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as xdmf:
            domain = xdmf.read_mesh()

        self.locs = compute_midpoints(domain, 2, np.arange(domain.topology.index_map(2).size_local, dtype='int32'))
        return domain
    
    def createFunctions(self):
        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)
        self.B = Function(self.V, name="Body Force")
        self.T = Constant(self.domain, dolfinx.default_scalar_type((0, 0)))
        self.density = Function(self.U, name="Density")
        self.vm = Function(self.U, name="Von Mises Stress")
        self.complianceArr = Function(self.U, name="Compliance")

    def fixedBoundary(self, f):
        fdim = self.domain.topology.dim - 1
        boundary_facets = locate_entities_boundary(self.domain, fdim, f)

        u_D = np.array([0, 0], dtype=dolfinx.default_scalar_type)
        dofs = locate_dofs_topological(self.V, fdim, boundary_facets)
        self.bcs.append(dirichletbc(u_D, dofs, self.V))

    def applyForce(self, forces):
        def bodyForce(x):
            T = np.zeros_like(x[:2,:])
            for loc, force in forces:
                c = np.isclose(x[0], loc[0], rtol=0.01) & np.isclose(x[1], loc[1], rtol=0.01)
                T = T + np.stack([force[0]*c, force[1]*c], axis=0)
            return T

        self.B.interpolate(bodyForce)

    def constituentEqns(self, Emin, E0, penal, nu):
        E = Emin + self.density**penal*(E0-Emin)
        lambda_ = E*nu/(1+nu)/(1-2*nu)
        mu = E/(2*(1+nu))

        self.eps = lambda u: sym(grad(u))
        self.sigma = lambda u: lambda_ * nabla_div(u) * Identity(len(u)) + 2 * mu * self.eps(u)
        s = lambda u: self.sigma(u) - 1. / 3 * tr(self.sigma(u)) * Identity(len(u))
        self.von_Mises = lambda u: sqrt(3. / 2 * inner(s(u), s(u)))
    
    def solve(self):
        ds = Measure('ds', domain=self.domain)
        a = inner(self.sigma(self.u), self.eps(self.v)) * dx
        L = dot(self.B, self.v) * dx + dot(self.T, self.v) * ds

        problem = LinearProblem(a, L, bcs=self.bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        self.disp = problem.solve()

    def updateStress(self):
        stress_expr = Expression(self.von_Mises(self.disp), self.U.element.interpolation_points())
        comp_expr = Expression(inner(self.eps(self.disp), self.sigma(self.disp)), self.U.element.interpolation_points())
        self.complianceArr.interpolate(comp_expr)
        self.vm.interpolate(stress_expr)

    def compliance(self):
        comp = MPI.COMM_WORLD.allreduce(assemble_scalar(form(inner(self.eps(self.disp), self.sigma(self.disp)) * dx)),op=MPI.SUM)
        return comp
    
    def displayResult(self, showUndeformed = False):
        if pyvista.OFF_SCREEN:
            pyvista.start_xvfb(wait=0.1)

        p = pyvista.Plotter()

        topology, cell_types, geometry = dolfinx.plot.vtk_mesh(self.domain, self.domain.topology.dim)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

        if showUndeformed:
            p.add_mesh(grid)

        grid["u"] = self.disp.x.array.reshape((geometry.shape[0], 2))
        z = np.zeros(len(grid['u']))
        grid['u'] = np.c_[grid["u"], z]
        warped = grid.warp_by_vector("u", factor=1)

        warped.cell_data["VonMises"] = self.vm.vector.array
        warped.set_active_scalars("VonMises")
        p.add_mesh(warped,clim=[0, 0.3*max(self.vm.vector.array)], cmap='jet')
        p.add_camera_orientation_widget()

        if not pyvista.OFF_SCREEN:
            p.show()
            p.screenshot('imgs/stress.png')
        else:
            print("Unable to show plot.")

    def displayDistribution(self):
        if pyvista.OFF_SCREEN:
            pyvista.start_xvfb(wait=0.1)

        p = pyvista.Plotter()

        topology, cell_types, geometry = dolfinx.plot.vtk_mesh(self.domain, self.domain.topology.dim)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

        grid.cell_data["Density"] = 1-self.density.vector.array
        grid.set_active_scalars("Density")
        p.add_mesh(grid, cmap='gray', clim=[0, 1])
        p.add_camera_orientation_widget()

        if not pyvista.OFF_SCREEN:
            p.show()
            p.screenshot('imgs/density.png')
        else:
            print("Unable to show plot.")