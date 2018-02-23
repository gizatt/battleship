from __future__ import absolute_import, division, print_function

import os
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import sys

import battleship_utils_py as bscpp

from pydrake.solvers import mathematicalprogram as mp
import pydrake.symbolic as sym
from pydrake.solvers.gurobi import GurobiSolver
        
def tfmat(x, y, theta):
    return np.array([[math.cos(theta), -math.sin(theta), x],
                     [math.sin(theta), math.cos(theta), y],
                     [0, 0, 1]])

# Line segment vw, point p
# Ref https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
def project_onto_line_segment(v, w, p):
    l2 = np.linalg.norm(w - v)
    if l2 == 0:
        return v
    # Project p onto vw without considering its bounds, scaled by that line segment length
    # and bound to 0, 1
    t = max(0, min(1, np.dot(p - v, w - v) / l2))
    return v + t * (w - v)

def get_sign_of_distance_to_line_segment(v, w, p):
    return np.sign((w-v)[0]*p[1] - (w-v)[1]*p[0])

def LogicalAndAsVariable(prog, invars):
    tempvar = prog.NewBinaryVariables(1, "temp_logical_and_var")[0]
    prog.AddLinearConstraint(tempvar >= np.sum(invars) - (len(invars)-1))
    for var in invars:
        prog.AddLinearConstraint(tempvar <= var)
    return tempvar

class Board():
    ''' Simulates a board of the battleship game.
        
        Represents the board state as a list
        of the 3DOF (planar) poses of each
        ship. Each ship class has an integer length
        (the class is represented as an integer, which
        is this length). The ship extends in its local
        +x direction by this extent minus epsilon. (So
        two 1-length ships may be immediately neighboring.)

        Board states are expected to conform to
        the restrictions of the board:
            - Yaw must be an integer multiple of pi/2.
            - X and Y must be integer.
            - No ships may intersect.
            - No ship extent may protrude outside of the
              board boundaries.
    '''

    def __init__(self, width = 10, height = 10):
        self.width = width
        self.height = height

        self.ships = []

    def spawn_N_ships(self, N, max_length=5):
        # Spawn ships randomly, and then project to the nearest feasible
        # configuration.
        color_generator = iter(plt.cm.rainbow(np.linspace(0, 1, N)))
        for i in range(N):
            new_ship = bscpp.Ship(random.randrange(1, max_length+1),
                            random.uniform(0., self.width),
                            random.uniform(0., self.height),
                            random.uniform(0., math.pi*2.),
                            next(color_generator))
            self.ships.append(new_ship)

    def draw(self, ax):
        ax.axis('equal')
        ax.axis('on')
        ax.set_aspect('equal', 'datalim')
        ax.set_xticks(range(0, self.width+1))
        ax.set_xticks(np.arange(-0.5, self.width, 1.0), minor=True)
        ax.set_yticks(range(0, self.height+1))
        ax.set_yticks(np.arange(-0.5, self.height, 1.0), minor=True)
        ax.grid(which="major", color="b", linestyle="--")
        ax.grid(which="minor", color="b", linestyle="-")


        for ship in self.ships:
            ship_points = ship.GetPointsInWorldFrame(side_length=0.9, spacing=1.0)
            ax.fill(ship_points[0, :], ship_points[1, :],
                        edgecolor='k',
                        facecolor=ship.get_color(),
                        closed=True)

        plt.pause(0.00001)



    def project_to_feasibility_nlp(self, ships):
        # Represent finding a board configuration as
        #  a resolution of nonlinear nonpenetration
        #  constraints (entirely point-vs-plane, for
        #  points sampled sufficiently densely
        #  on the surface of each ship), against
        #  an objective with local minima at integer
        #  solutions. (The "nominal" input configuration
        #  does *not* enter other than as the solver's
        #  initial guess... I wonder if I can do any better
        #  than that...)

        # Aim to have nonpenetration enter as a constraint
        # of the form phi(q) >= 0 elementwise
        # (Where each element of phi will be the collision
        # distance from each point on each body

        all_points = []
        for i, ship in enumerate(ships):
            all_points.append(ship.GetPointsInWorldFrame(spacing=0.5, side_length=1.0))
        npts = np.sum( [pts.shape[1] for pts in all_points] )

        print("Npts %d" % npts)

        phi = np.zeros(npts)
        dphi_dq = np.zeros((npts, len(ships)*3))

        dphi_dqall = dphi_dq
        for i, ship in enumerate(ships):
            for k in range(all_points[i].shape[1]):
                point = all_points[i][:, k]
                for other_ship in ships:
                    if other_ship is not ship:
                        phi[(i*4):(i*4+4)], dphi_dq[(i*4):(i*4+4), (k*3):(k*3+3)] = other_ship.getSignedDistanceToPoint(point)


        return ships




        


    def project_to_feasibility_mip(self, ships):
        prog = mp.MathematicalProgram()

        # Represent board as an occupancy grid
        board_variables = prog.NewContinuousVariables(self.width, self.height, "board_occupancy")
        for h in range(self.height):
            for w in range(self.width):
                prog.AddLinearConstraint(board_variables[w, h] >= 0.0)
                prog.AddLinearConstraint(board_variables[w, h] <= 1.0)
        prog.AddLinearCost(np.sum(board_variables))

        board_variable_contributions = board_variables * 0.0

        ship_pose_variables_by_ship = []
        for i, ship in enumerate(ships):
            ship_pose_variables = {}

            # Approximate each of our pose variables with
            # a sum of binary variables, with only one allowed
            # to be active

            # TODO(gizatt) replace with integer-constrained variables
            # when those become available...
            ship_pose_variables["x"] = prog.NewContinuousVariables(1, "ship_%d_x" % i)[0]
            prog.AddLinearConstraint(ship_pose_variables["x"] >= 0.0)
            prog.AddLinearConstraint(ship_pose_variables["x"] <= self.width - 1.0)
            ship_pose_variables["y"] = prog.NewContinuousVariables(1, "ship_%d_x" % i)[0]
            prog.AddLinearConstraint(ship_pose_variables["y"] >= 0.0)
            prog.AddLinearConstraint(ship_pose_variables["y"] <= self.height - 1.0)
            ship_pose_variables["t"] = prog.NewContinuousVariables(1, "ship_%d_x" % i)[0]
            prog.AddLinearConstraint(ship_pose_variables["t"] >= 0.0)
            prog.AddLinearConstraint(ship_pose_variables["t"] <= 4*math.pi/3.)
            ship_pose_variables["x_bins"] = prog.NewBinaryVariables(self.width, "ship_%d_x_bins" % i)
            ship_pose_variables["y_bins"] = prog.NewBinaryVariables(self.height, "ship_%d_y_bins" % i)
            ship_pose_variables["t_bins"] = prog.NewBinaryVariables(4, "ship_%d_t_bins" % i) # four cardinal directions from +x ccw
            prog.AddLinearConstraint(np.sum(ship_pose_variables["x_bins"]) == 1)
            prog.AddLinearConstraint(np.sum(ship_pose_variables["y_bins"]) == 1)
            prog.AddLinearConstraint(np.sum(ship_pose_variables["t_bins"]) == 1)
    
            prog.AddLinearConstraint(np.sum(
                ship_pose_variables["x_bins"]*np.arange(0., self.width))
                    == ship_pose_variables["x"])
            prog.AddLinearConstraint(np.sum(
                ship_pose_variables["y_bins"]*np.arange(0., self.height)) 
                    == ship_pose_variables["y"])
            prog.AddLinearConstraint(np.sum(
                ship_pose_variables["t_bins"]*np.arange(0., math.pi*2, math.pi/2.)) 
                    == ship_pose_variables["t"])

            # That's an awful lot of work that we wouldn't have to do if we had
            # integer constraints...
            prog.AddQuadraticCost( (ship_pose_variables["x"]-ship.x)**2. )
            prog.AddQuadraticCost( (ship_pose_variables["y"]-ship.y)**2. )
            prog.AddQuadraticCost( (ship_pose_variables["t"]-ship.theta)**2. )

            # Using those, we can calculate (as a linear relationship) the
            # contribution of each ship to each cell in the occupancy grid
            for h in range(self.height):
                for w in range(self.width):
                    # At the origin of the ship, it's simple -- occupancy
                    # if the ship is here
                    board_variable_contributions[w, h] += LogicalAndAsVariable(prog, 
                                             [ship_pose_variables["x_bins"][w],
                                              ship_pose_variables["y_bins"][h]])

                    for l in range(1, ship.length):
                        # At l along the ship length, we could cause occupancy by being
                        # in any of the cardinal directions by length, depending on
                        # the value of theta

                        # Theta = 0 --> ship extends +x, so look w-l for origin
                        if w - l >= 0:
                            board_variable_contributions[w, h] += LogicalAndAsVariable(prog,
                                [ship_pose_variables["x_bins"][w-l], 
                                 ship_pose_variables["y_bins"][h],
                                 ship_pose_variables["t_bins"][0]])
                        else: # This combination of variables violates bounds
                            prog.AddLinearConstraint(ship_pose_variables["x_bins"][w]
                                                   + ship_pose_variables["t_bins"][2] <= 1)

                        # Theta = 1 --> ship extends +y
                        if h - l >= 0:
                            board_variable_contributions[w, h] += LogicalAndAsVariable(prog,
                                [ship_pose_variables["x_bins"][w],
                                 ship_pose_variables["y_bins"][h-l],
                                 ship_pose_variables["t_bins"][1]])
                        else: # This combination of variables violates bounds
                            prog.AddLinearConstraint(ship_pose_variables["y_bins"][h]
                                                   + ship_pose_variables["t_bins"][3] <= 1)
                        # Theta = 2 --> ship extends -x
                        if w + l < self.width:
                            board_variable_contributions[w, h] += LogicalAndAsVariable(prog,
                                [ship_pose_variables["x_bins"][w+l],
                                 ship_pose_variables["y_bins"][h],
                                 ship_pose_variables["t_bins"][2]])
                        else: # This combination of variables violates bounds
                            prog.AddLinearConstraint(ship_pose_variables["x_bins"][w]
                                                   + ship_pose_variables["t_bins"][0] <= 1)
                        # Theta = 3 --> ship extends -y
                        if h + l < self.height:
                            board_variable_contributions[w, h] += LogicalAndAsVariable(prog,
                                [ship_pose_variables["x_bins"][w],
                                 ship_pose_variables["y_bins"][h+l],
                                 ship_pose_variables["t_bins"][3]])
                        else: # This combination of variables violates bounds
                            prog.AddLinearConstraint(ship_pose_variables["y_bins"][h]
                                                   + ship_pose_variables["t_bins"][1] <= 1)

            ship_pose_variables_by_ship.append(ship_pose_variables)


        for h in range(self.height):
            for w in range(self.width):
                prog.AddLinearConstraint(board_variables[w, h] >= board_variable_contributions[w, h])


        solver = GurobiSolver()
        if not solver.available():
            print("Couldn't set up Gurobi :(")
            exit(1)
        prog.SetSolverOption(solver.solver_type(), "OutputFlag", 1)
        prog.SetSolverOption(solver.solver_type(), "LogToConsole", 1)
        prog.SetSolverOption(solver.solver_type(), "MIPGap", 0.05)
        result = solver.Solve(prog)
        
        print(prog.GetSolution(board_variables))

        out_ships = []
        for i, ship in enumerate(ships):
            length = ship.length
            x = prog.GetSolution(ship_pose_variables_by_ship[i]["x"])
            y = prog.GetSolution(ship_pose_variables_by_ship[i]["y"])
            t = prog.GetSolution(ship_pose_variables_by_ship[i]["t"])
            out_ships.append(Ship(length, x, y, t, ship.color))
            print("Ship %d: %dx(%f,%f,%f)" % (i, length, x, y, t))
        return out_ships

if __name__ == "__main__":
    ship = bscpp.Ship(5, 2.3, 1.0, 0.2, [1., 0., 0.])

    board = Board(10, 10)
    board.spawn_N_ships(20, max_length=3)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    board.draw(ax1)
    fig.show()

    board.ships = board.project_to_feasibility_nlp(board.ships)
    board.draw(ax2)
    plt.show()
