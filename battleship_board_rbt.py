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

def nullspace(A, atol=1e-13, rtol=0):
    """Compute an approximate basis for the nullspace of A.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length k will be treated
        as a 2-D with shape (1, k)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    ns : ndarray
        If `A` is an array with shape (m, k), then `ns` will be an array
        with shape (k, n), where n is the estimated dimension of the
        nullspace of `A`.  The columns of `ns` are a basis for the
        nullspace; each element in numpy.dot(A, ns) will be approximately
        zero.
    """
    A = np.atleast_2d(A)
    u, s, vh = np.linalg.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns

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


    qf = results.q_sol[0]
    info = results.info[0]
    dqf_dq0 = np.zeros(qf.shape[0])
    if info == 1:
        # We've solved an NLP of the form:
        # qf = argmin_q || q - q_0 ||
        #        s.t. phi(q) >= 0
        #
        # which projects q_0 into the feasible set $phi(q) >= 0$.
        # We want to return the gradient of qf w.r.t. q_0.

        # We'll tackle an approximation of this (which isn't perfect,
        # but is a start):
        # We'll build a linear approximation of the active set at
        # the optimal value, and project the incremental dq_0 into
        # the null space of this active set.

        # These vectors all point in directions that would
        # bring q off of the constraint surfaces.
        constraint_violation_directions = []

        cache = rbt.doKinematics(qf)
        for i, constraint in enumerate(constraints):
            c, dc = constraint.eval(0, cache)
            lb, ub = constraint.bounds(0)

            phi_lb = c - lb
            phi_ub = ub - c
            for k in range(c.shape[0]):
                if phi_lb[k] < -1E-6 or phi_ub[k] < -1E-6:
                    print("Bounds violation detected, solution wasn't feasible")
                    print("%f <= %f <= %f" % (phi_lb[k], c[k], phi_ub[k]))
                    return qf, info, dqf_dq0

                if phi_lb[k] < 1E-6:
                    # Not allowed to go down
                    constraint_violation_directions.append(-dc[k, :])

                # If ub = lb and ub is active, then lb is also active,
                # and we don't need to double-add this vector.
                if phi_ub[k] < 1E-6 and phi_ub[k] != phi_lb[k]:
                    # Not allowed to go up
                    constraint_violation_directions.append(dc[k, :])

        # Build a full matrix C(q0_new - qstar) = 0
        # that approximates the feasible set.
        if len(constraint_violation_directions) > 0:
            C = np.vstack(constraint_violation_directions)
            ns = nullspace(C)

            dqf_dq0 = np.eye(qf.shape[0])
            dqf_dq0 = np.dot(np.dot(dqf_dq0, ns), ns.T)
        else:
            # No null space so movements
            dqf_dq0 = np.eye(qf.shape[0])

    return qf, info, dqf_dq0

def projectToFeasibilityWithNLP(rbt, q0, board_width, board_height):
    # More generic than above... instead of using IK to quickly
    # assembly the nlp solve that goes to snopt, build it ourselves.
    # (Gives us lower-level control at the cost of complexity.)

    print("TODO, I think this requires a few new drake bindings"
          " for generic nonlinear constraints")


if __name__ == "__main__":

    np.set_printoptions(precision=4, suppress=True)

    os.system("mkdir -p figs")
    fig, (ax1, ax2) = plt.subplots(1, 2)
    scatter_fig, scatter_ax = plt.subplots(1, 1)

    board_width = 10
    board_height = 10


    for i in range(20):
        info_0 = -1
        while info_0 != 1:
            ax1.clear()
            ax2.clear()
            rbt, q0 = spawn_rbt(board_width, board_height, 5, 5)
            draw_board_state(ax1, q0, board_width, board_height)
            q_sol_0, info_0, dqf_dq0_0 = \
                projectToFeasibilityWithIK(rbt, q0, board_width, board_height)

        error_pairs = []
        for j in range(50):
            info = -1
            while info != 1:
                noise = np.random.normal(loc=0.0, scale=0.1/(j+1), size=q0.shape)
                q_sol, info, dqf_dq0 = \
                    projectToFeasibilityWithIK(rbt, q0+noise, board_width, board_height)

            # Did our linearization predict this solution very well?
            expected_q_sol_new = np.dot(dqf_dq0_0, noise) + q_sol_0
            est_error = np.linalg.norm(expected_q_sol_new - q_sol)
            ini_error = np.linalg.norm(noise)
            print("\nError in estimate: ", est_error)
            print("Error in initial: ", ini_error)
            error_pairs.append([ini_error, est_error])

            draw_board_state(ax2, q_sol, board_height, board_width)
            plt.draw()
            plt.pause(1e-6)

        fig.savefig('figs/plot_run_%d_ik.png' % i)

        all_error_pairs = np.vstack(error_pairs).T
        scatter_ax.clear()
        scatter_ax.scatter(all_error_pairs[0, :], all_error_pairs[1, :])
        scatter_ax.plot([-10.0, 10.0], [-10.0, 10.0], '--')
        scatter_ax.set_xlim([0., 1.1*np.max(all_error_pairs[0, :])])
        scatter_ax.set_ylim([0., 1.1*np.max(all_error_pairs[1, :])])
        scatter_ax.set_xlabel("Norm difference to q0_new")
        scatter_ax.set_ylabel("Prediction error of qf_new")
        scatter_ax.grid(True)
        scatter_fig.savefig('figs/plot_run_%d_prediction_error_of_lin.png' % i)

        plt.pause(0.1)

    plt.show()