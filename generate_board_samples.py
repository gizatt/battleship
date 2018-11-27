import os
import matplotlib.pyplot as plt
import numpy as np
import battleship_board_rbt as bsrbt

'''

Rejection-samples feasible configurations of non-penetrating
blocks. Generates a uniform distribution over
feasible configurations.

'''

output_file = "data/uniform_feasible_2d.csv"
draw = False
n_samples = 1000000
board_width = board_height = 10
max_length = 1
n_ships = 2

num_rejected = 0
num_accepted = 0

file = open(output_file, 'w')

try:
    def update_print_str(a, r):
        print "\rRejected %d, Accepted %d, Rate %f" % \
            (r, a, float(a) / max(1, (a + r))),

    if draw:
        fig, ax = plt.subplots(1, 1)
        plt.show(block=False)
        rbt, q0 = bsrbt.spawn_rbt(board_width, board_height,
                                  max_length, n_ships)
        viz = bsrbt.draw_board_state(ax, rbt, q0, board_width, board_height)

    for k in range(n_samples):
        has_no_collision = False
        while has_no_collision is False:
            rbt, q0 = bsrbt.spawn_rbt(board_width, board_height,
                                      max_length, n_ships)
            #q0[3] = q0[0] + 1.
            # Check collision distances
            kinsol = rbt.doKinematics(q0)
            ptpairs = rbt.ComputeMaximumDepthCollisionPoints(kinsol)
            if draw:
                viz.draw(q0)
                plt.draw()
                plt.pause(1e-6)

            update_print_str(num_accepted, num_rejected)

            if len(ptpairs) == 0:
                has_no_collision = True
                num_accepted += 1
            else:
                num_rejected += 1
        file.write(",".join([str(x) for x in q0.tolist()]) + "\n")

    update_print_str(num_accepted, num_rejected)
except Exception as e:
    print "Exception: ", e
    file.close()
