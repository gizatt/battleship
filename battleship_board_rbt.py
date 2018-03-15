from __future__ import absolute_import, division, print_function

import os
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import sys

import battleship_utils_py as bscpp

from pydrake.all import (AutoDiffXd,
                         GurobiSolver,
                         RigidBodyTree,
                         RigidBody)
from pydrake.solvers import ik
from pydrake.multibody.joints import PrismaticJoint, RevoluteJoint
from pydrake.multibody.shapes import Box, VisualElement
from pydrake.multibody.collision import CollisionElement

from underactuated import PlanarRigidBodyVisualizer

class BattleshipBoardVisualizer(PlanarRigidBodyVisualizer):
    def __init__(self, width, height, *args, **kwargs):
        PlanarRigidBodyVisualizer.__init__(self, *args, **kwargs)

        self.width = width
        self.height = height

        
        self.ax.autoscale(enable=False, axis='both')
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.axis('on')
        #self.ax.set_aspect('equal', 'datalim')
        self.ax.set_xticks(range(0, self.width+1))
        self.ax.set_xticks(np.arange(-0.5, self.width, 1.0), minor=True)
        self.ax.set_yticks(range(0, self.height+1))
        self.ax.set_yticks(np.arange(-0.5, self.height, 1.0), minor=True)
        self.ax.grid(which="major", color="b", linestyle="--")
        self.ax.grid(which="minor", color="b", linestyle="-")

        self.ax.set_xlim([0, self.width])
        self.ax.set_ylim([0, self.height])


    def draw(self, context):
        PlanarRigidBodyVisualizer.draw(self, context)

def draw_board_state(ax, q, board_width, board_height):
    Tview = np.array([[1., 0., 0., 0.],
                      [0., 1., 0., 0.],
                      [0., 0., 0., 0.],
                      [0., 0., 0., 1.]])
    viz = BattleshipBoardVisualizer(board_width, board_height,
        rbt, Tview, xlim=[0., board_width],
        ylim=[0., board_height], ax = ax)
    viz.draw(q)

def spawn_rbt(width, height, max_length, N):
    rbt = RigidBodyTree()
    q0 = np.zeros(N*3)
    world_body = rbt.get_body(0)


    color_generator = iter(plt.cm.rainbow(np.linspace(0, 1, N)))
    for i in range(N):
        q0[i*3+0] = random.uniform(0., width)
        q0[i*3+1] = random.uniform(0., height)
        q0[i*3+2] = random.uniform(0., math.pi*2.)
        color = next(color_generator)
        length = random.randrange(1, max_length + 1)

        joint_link_x = RigidBody()
        joint_link_x.set_name("ship_%d_joint_link_x" % i)
        joint_link_x.add_prismatic_joint(
            world_body, PrismaticJoint("x", np.eye(4), np.array([1., 0., 0.])))
        rbt.add_rigid_body(joint_link_x)

        joint_link_y = RigidBody()
        joint_link_y.set_name("ship_%d_joint_link_y" % i)
        joint_link_y.add_prismatic_joint(
            joint_link_x, PrismaticJoint("y", np.eye(4), np.array([0., 1., 0.])))
        rbt.add_rigid_body(joint_link_y)

        ship_link = RigidBody()
        ship_link.set_name("ship_%d_ship_link" % i)
        ship_link.add_revolute_joint(
            joint_link_y, RevoluteJoint("theta", np.eye(4), np.array([0., 0., 1.])))
        boxElement = Box([1.0, length, 1.0])
        boxVisualElement = VisualElement(boxElement, np.eye(4), color)
        ship_link.AddVisualElement(boxVisualElement)
        # necessary so this last link isn't pruned by the rbt.compile() call
        ship_link.set_spatial_inertia(np.eye(6)) 
        # get welded
        rbt.add_rigid_body(ship_link)

        boxCollisionElement = CollisionElement(boxElement, np.eye(4))
        boxCollisionElement.set_body(ship_link)
        rbt.addCollisionElement(boxCollisionElement, ship_link, "default")


    rbt.compile()
    return rbt, q0

def projectToFeasibilityWithIK(rbt, q0, board_width, board_height):        
    constraints = []

    constraints.append(ik.MinDistanceConstraint(
        model=rbt, min_distance=0.01))

    for body_i in range(rbt.get_num_bodies()-1):
        # All corners on body must be inside of the
        # bounds of the board
        body = rbt.get_body(body_i+1)
        visual_elements = body.get_visual_elements()
        if len(visual_elements) > 0:
            points = visual_elements[0].getGeometry().getPoints()
            lb = np.tile(np.array([0., 0., -100.]), (points.shape[1], 1)).T
            ub = np.tile(np.array([board_width, board_height, 100.]), (points.shape[1], 1)).T

            constraints.append(ik.WorldPositionConstraint(
                rbt, body_i+1, points, lb, ub))


    options = ik.IKoptions(rbt)
    options.setDebug(True)
    options.setMajorIterationsLimit(10000)
    options.setIterationsLimit(100000)
    results = ik.InverseKin(
        rbt, q0, q0, constraints, options)
    return results.q_sol[0], results.info

def projectToFeasibilityWithNLP(rbt, q0, board_width, board_height):        
    # More generic than above... instead of using IK to quickly
    # assembly the nlp solve that goes to snopt, build it ourselves.
    # (Gives us lower-level control at the cost of complexity.)

    print("TODO, I think this requires a few new drake bindings"
          " for generic nonlinear constraints")


if __name__ == "__main__":


    fig, (ax1, ax2) = plt.subplots(1, 2)

    board_width = 10
    board_height = 10

    info_histogram = {}

    for i in range(10000):
        ax1.clear()
        ax2.clear()

        rbt, q0 = spawn_rbt(board_width, board_height, 5, 5)

        draw_board_state(ax1, q0, board_width, board_height)

        for j in range(10):
            noise = np.random.normal(q0.shape)*0.1
            q_sol, info = projectToFeasibilityWithIK(rbt, q0+noise, board_width, board_height)
            print(q_sol, info)

            if info[0] not in info_histogram.keys():
                info_histogram[info[0]] = 1
            else:
                info_histogram[info[0]] += 1

            print(info_histogram)

            draw_board_state(ax2, q_sol, board_height, board_width)

            plt.draw()

            plt.pause(1e-6)

        plt.pause(1)