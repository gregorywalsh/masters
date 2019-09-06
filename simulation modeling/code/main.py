import simulator as sim
import simulator_cont as sim_cont
import simulator_tri as sim_tri
import integrator as integ
import analytic as an
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

from matplotlib import pyplot as plt, rcParams

rcParams['font.sans-serif'] = ['Linux Biolinum', 'Tahoma', 'DejaVu Sans',
                               'Lucida Grande', 'Verdana']

# This part of the code was written before I realised that alpha could be calculated analytically
def get_alpha(n, Eh, El, a):
    def get_payoff_extremum(extremum_function):
        payoff_extremum = float('inf') if extremum_function == min else -float('inf')
        for strat in (0, 1):
            for nh in range(0, n):
                E = Eh if strat == 1 else El
                mark = (
                               E
                               + (nh * Eh)
                               + ((n - nh) * El)
                       ) / n
                payoff = mark - (a * E)
                payoff_extremum = extremum_function(payoff_extremum, payoff)
        return payoff_extremum

    payoff_max = get_payoff_extremum(max)
    payoff_min = get_payoff_extremum(min)
    alpha = 1 / (payoff_max - payoff_min)
    return alpha

# General Constants
t_max = 3
runs = 50

n = 4
x0 = 0.5
x0s = [0, 0.25, 0.5, 0.75, 1]
y0 = 0.6
Eh = 1
El = 0
Em = 0.5
E_diffs = [0, 0.5, 1, 1.5]
a = 0.5
As = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
alpha = get_alpha(n=n, Eh=Eh, El=El, a=a)

# Simulator Constants
num_groups = 2

# Integrator Constants
h = 1
hs = [0.1, 1, 5]

# Runtime Variables
run_step_by_step = True
generate_x_vs_t = False
generate_x_vs_h = False
generate_x_vs_a = False
generate_x_vs_x0 = False
generate_ibar = False
generate_xyz_vs_t = False

# Get a simulator, an integrator, and a analytic
simulator = sim.Simulator(n=n, x0=x0, El=El, Eh=Eh, a=a, alpha=alpha, num_groups=num_groups)
integrator = integ.integrator(n=n, x0=x0, Eh=Eh, El=El, a=a, alpha=alpha, h=h)
analytic = an.analytic(n=n, x0=x0, Eh=Eh, El=El, a=a, alpha=alpha, h=h)
simulator_cont = sim_cont.Simulator(n=n, x0=x0, min_effort=El, max_effort=Eh, a=a, alpha=alpha, num_groups=num_groups)
simulator_tri = sim_tri.Simulator(n=n, x0=x0, y0=y0, El=El, Em=Em, Eh=Eh, a=a, alpha=alpha, num_groups=num_groups)

# RUN STEP BY STEP
if run_step_by_step:

    print('Simulator:{:10.4f}'.format(simulator.get_hardworking_prop()))
    print(simulator)
    print()
    for t in range(t_max):
        print('t:', t+1)
        print('Simulator:{:10.3f}'.format(simulator.advance()))
        print(simulator)
        # result = integrator.advance(verbose=True)
        # print('x: {:.3f}'.format(result[0]))
        # print('K1: {:.3f}'.format(result[1]))
        # print('K2: {:.3f}'.format(result[2]))
        # print('K3: {:.3f}'.format(result[3]))
        # print('K4: {:.3f}'.format(result[4]))
        # print('Integrator:{:10.3f}'.format(integrator.advance()))
        # print('Analytic:{:10.3f}'.format(analytic.advance()))
        print()


# GENERATE X vs T
if generate_x_vs_t:
    sim_xyz = np.zeros(shape=(t_max + 1, runs))
    analytic_x = np.zeros(shape=t_max + 1)
    integ_x = np.zeros(shape=t_max + 1)

    sim_xyz[0, :] = x0
    analytic_x[0] = x0
    integ_x[0] = x0

    for run in range(runs):
        simulator = sim.Simulator(n=n, x0=x0, El=El, Eh=Eh, a=a, alpha=alpha, num_groups=num_groups)
        for t in range(t_max):
            sim_xyz[t + 1, run] = simulator.advance()

    for t in range(t_max):
        integ_x[t + 1] = integrator.advance()
        analytic_x[t + 1] = analytic.advance()


    # Find mean simulation and std
    sim_x_mean = np.mean(a=sim_xyz, axis=1)
    sim_x_sigma = np.std(a=sim_xyz, axis=1)
    t = range(t_max + 1)

    # Prepare vector field
    T, X = np.meshgrid(np.arange(0, t_max + 1, 3), np.arange(0, 1.1, .1))
    slopes = 30 * analytic.slope(X)
    inv_tan = np.arctan(slopes)
    U = np.cos(inv_tan)
    V = np.sin(inv_tan)

    fig = plt.figure(figsize=(3.45, 3.45), dpi=220)

    Q = plt.quiver(T, X, U, V, units='width', color='#dbdbdb', label='Trajectory')

    plt.scatter(range(t_max + 1), integ_x, label='Numerical ($\delta$=1)', color='#3d3d3d', marker='.', lw=2, s=15, zorder=2)
    plt.scatter(range(t_max + 1), sim_x_mean, label='Simulated', color='#00b2ff', lw=1, marker='+', s=60, zorder=3)
    plt.plot(range(t_max + 1), analytic_x, label='Analytical', color='#999999', zorder=1, lw=1)
    plt.fill(np.concatenate([t, t[::-1]]),
            np.concatenate([sim_x_mean - (2.576 * sim_x_sigma),
                           (sim_x_mean + 2.576 * sim_x_sigma)[::-1]]),
            alpha=.2, fc='#00b2ff', ec='None', label='Simulated CI (99%)', zorder=0)
    plt.xlabel('t (number of semesters)', fontweight='bold', fontsize=9)
    plt.ylabel('x(t)', fontweight='bold', fontsize=9)
    plt.tick_params(labelsize=8)

    ax = fig.axes[0]
    handles, labels = ax.get_legend_handles_labels()
    handles = [handles[0], handles[3], handles[4], handles[1]]
    labels = [labels[0], labels[3], labels[4], labels[1]]
    legend = plt.legend(frameon=True, handles=handles, labels=labels, loc=1, framealpha=1)
    legend.get_frame().set_facecolor('#ffffff')
    fig.savefig('figures/x_t.pdf', frameon=None)
    plt.show()


# GENERATE X VS H
if generate_x_vs_h:

    integ_x = {}
    num_points = {}
    for i, h in enumerate(hs):
        num_points[h] = int(t_max / h)+1
        integ_x[h] = np.zeros(shape=num_points[h])
        integ_x[h][0] = x0
        integrator = integ.integrator(n=n, x0=x0, Eh=Eh, El=El, a=a, alpha=alpha, h=h)
        for j in range(1, num_points[h]):
            integ_x[h][j] = integrator.advance()

    fig = plt.figure(figsize=(3.45, 3.45), dpi=220)


    plt.plot(np.linspace(start=0, stop=t_max, num=num_points[0.1]), integ_x[0.1], label='$\delta$=0.1', color='#999999', zorder=1, lw=1)
    plt.scatter(np.linspace(start=0, stop=t_max, num=num_points[1]), integ_x[1], label='$\delta$=1', color='#3d3d3d', marker='.', lw=2, s=15, zorder=2)
    plt.scatter(np.linspace(start=0, stop=t_max, num=num_points[5]), integ_x[5], label='$\delta$=5', color='#00b2ff', marker='+', lw=1, s=60, zorder=3)
    plt.legend(frameon=False)
    plt.tick_params(labelsize=8)
    plt.xlabel('t (number of semesters)', fontweight='bold', fontsize=9)
    plt.ylabel('x(t)', fontweight='bold', fontsize=9)

    fig.savefig(fname='figures/x_h.pdf',
                frameon=None)
    plt.show()


# GENERATE X VS a
if generate_x_vs_a:
    a_count = len(As)
    sim_xyz = np.zeros(shape=(t_max + 1, a_count, runs))
    analytic_x = np.zeros(shape=(t_max + 1, a_count))
    integ_x = np.zeros(shape=(t_max + 1, a_count))

    sim_xyz[0, :, :] = x0
    analytic_x[0, :] = x0
    integ_x[0, :] = x0

    for i, a in enumerate(As):
        alpha = get_alpha(n=n, Eh=Eh, El=El, a=a)
        for run in range(runs):
            simulator = sim.Simulator(n=n, x0=x0, El=El, Eh=Eh, a=a, alpha=alpha, num_groups=num_groups)
            for t in range(t_max):
                sim_xyz[t + 1, i, run] = simulator.advance()

        integrator = integ.integrator(n=n, x0=x0, Eh=Eh, El=El, a=a, alpha=alpha, h=h)
        analytic = an.analytic(n=n, x0=x0, Eh=Eh, El=El, a=a, alpha=alpha, h=h)
        for t in range(t_max):
            integ_x[t + 1, i] = integrator.advance()
            analytic_x[t + 1, i] = analytic.advance()

        # Find mean simulation and std
        sim_x_mean = np.mean(a=sim_xyz, axis=2)
        sim_x_sigma = np.std(a=sim_xyz, axis=2)

    fig = plt.figure(figsize=(3.45, 3.45), dpi=220)

    plt.scatter(As, integ_x[t_max, :], label='Numerical', color='#3d3d3d', lw=2, marker='.', s=15, zorder=3)
    plt.scatter(As, analytic_x[t_max, :], label='Analytic', color='#00b2ff', lw=1, marker='+', s=60, zorder=2)
    plt.errorbar(As, sim_x_mean[t_max, :], yerr=2.576 * sim_x_sigma[t_max, :],
                 linestyle='None', elinewidth=1, capsize=2,
                 label='Simulated', color='#999999', mew=1, marker='x', ms=5, zorder=1)
    plt.xlabel('a (coefficient of effort)', fontweight='bold', fontsize=9)
    plt.ylabel('x('+str(t_max)+')', fontweight='bold', fontsize=9)
    plt.tick_params(labelsize=8)

    ax = fig.axes[0]
    handles, labels = ax.get_legend_handles_labels()
    # handles = [handles[0], handles[2], handles[3], handles[1]]
    # labels = [labels[0], labels[2], labels[3], labels[1]]
    plt.legend(frameon=False, handles=handles, labels=labels)

    fig.savefig('figures/x_a.pdf', frameon=None)
    plt.show()

# GENERATE X VS X0
if generate_x_vs_x0:
    x0_count = len(x0s)
    sim_xyz = np.zeros(shape=(t_max + 1, x0_count, runs))
    analytic_x = np.zeros(shape=(t_max + 1, x0_count))
    integ_x = np.zeros(shape=(t_max + 1, x0_count))

    for i, a in enumerate(x0s):
        sim_xyz[0, i, :] = x0
        analytic_x[0, i] = x0
        integ_x[0, i] = x0
        for run in range(runs):
            simulator = sim.Simulator(n=n, x0=x0, El=El, Eh=Eh, a=a, alpha=alpha, num_groups=num_groups)
            for t in range(t_max):
                sim_xyz[t + 1, i, run] = simulator.advance()

        integrator = integ.integrator(n=n, x0=x0, Eh=Eh, El=El, a=a, alpha=alpha, h=h)
        analytic = an.analytic(n=n, x0=x0, Eh=Eh, El=El, a=a, alpha=alpha, h=h)
        for t in range(t_max):
            integ_x[t + 1, i] = integrator.advance()
            analytic_x[t + 1, i] = analytic.advance()

        # Find mean simulation and std
        sim_x_mean = np.mean(a=sim_xyz, axis=2)
        sim_x_sigma = np.std(a=sim_xyz, axis=2)

    fig = plt.figure(figsize=(3.45, 3.45), dpi=220)

    plt.scatter(x0s, integ_x[t_max, :], label='Numerical', color='#3d3d3d', lw=2, marker='.', s=15, zorder=3)
    plt.scatter(x0s, analytic_x[t_max, :], label='Analytic', color='#00b2ff', lw=1, marker='+', s=60, zorder=2)
    plt.errorbar(x0s, sim_x_mean[t_max, :], yerr=2.576 * sim_x_sigma[t_max, :],
                 linestyle='None', elinewidth=1, capsize=2,
                 label='Simulated', color='#999999', mew=1, marker='x', ms=5, zorder=1)
    plt.xlabel('x_0', fontweight='bold', fontsize=9)
    plt.ylabel('x('+str(t_max)+')', fontweight='bold', fontsize=9)
    plt.tick_params(labelsize=8)

    ax = fig.axes[0]
    handles, labels = ax.get_legend_handles_labels()
    # handles = [handles[0], handles[2], handles[3], handles[1]]
    # labels = [labels[0], labels[2], labels[3], labels[1]]
    plt.legend(frameon=False, handles=handles, labels=labels)

    fig.savefig('figures/x_x0.pdf', frameon=None)
    plt.show()

# GENERATE ibar vs T
if generate_ibar:
    sim_efforts = np.zeros(shape=(t_max + 1, n * num_groups, runs))

    for run in range(runs):
        simulator_cont = sim_cont.Simulator(n=n, x0=x0, min_effort=El, max_effort=Eh, a=a, alpha=alpha, num_groups=num_groups)
        sim_efforts[0, :, run] = simulator_cont.current_strats
        print(run)
        for t in range(t_max):
            sim_efforts[t + 1, :, run] = simulator_cont.advance()

    # Find mean simulation and std
    sim_efforts_reshaped = np.reshape(a=sim_efforts, newshape=(t_max + 1, n * num_groups * runs))

    fig = plt.figure(figsize=(3.45, 3.45), dpi=220)

    for t in [0, 20, 50, 150, 500, 1000, 2000]:
        # density = gaussian_kde(sim_efforts_reshaped[t])
        # density.covariance_factor = lambda: .25
        # density._compute_covariance()
        # xs = np.linspace(start=0, stop=1, num=101)
        # plt.fill(xs, density(xs), label='t = ' + str(t), alpha = 0.5)
        plt.hist(sim_efforts_reshaped[t], bins=np.linspace(start=0, stop=1, num=101), label='t = ' + str(t), density=True, alpha = 0.4)
    plt.xlabel('Effort', fontweight='bold', fontsize=9)
    plt.ylabel('Frequency', fontweight='bold', fontsize=9)
    plt.tick_params(labelsize=8)
    plt.legend(frameon=False)

    fig.savefig('figures/sim_cont.pdf', frameon=None)
    plt.show()


# GENERATE XYZ vs T
if generate_xyz_vs_t:

    fig = plt.figure(figsize=(3.45, 3.45), dpi=220)
    for Em in [0.20, 0.8]:
        sim_xyz = np.zeros(shape=(t_max + 1, 3, runs))

        for run in range(runs):
            simulator_tri = sim_tri.Simulator(n=n, x0=x0, y0=y0, El=El, Em=Em, Eh=Eh, a=a, alpha=alpha, num_groups=num_groups)
            sim_xyz[0, :, run] = x0, y0, 1 - x0 - y0
            print(run)
            for t in range(t_max):
                sim_xyz[t + 1, :, run] = simulator_tri.advance()
                # print(simulator_tri)

        # Find mean simulation and std
        sim_x_mean = np.mean(a=sim_xyz, axis=2)
        sim_x_sigma = np.std(a=sim_xyz, axis=2)
        t = range(t_max + 1)

        if Em == 0.2:
            plt.plot(range(t_max + 1), sim_x_mean[:, 0], label='Hard, $E_m=' + str(Em) + '$', color='#008ECC', ls='dotted', lw=1)
            plt.plot(range(t_max + 1), sim_x_mean[:, 2], label='Medium, $E_m=' + str(Em) + '$', color='#008ECC', ls='dashed', lw=1)
            plt.plot(range(t_max + 1), sim_x_mean[:, 1], label='Lazy, $E_m=' + str(Em) + '$', color='#008ECC', lw=1)
        else:
            plt.plot(range(t_max + 1), sim_x_mean[:, 0], label='Hard, $E_m=' + str(Em) + '$', color='#C5004C', ls='dotted', lw=1)
            plt.plot(range(t_max + 1), sim_x_mean[:, 2], label='Medium, $E_m=' + str(Em) + '$', color='#C5004C', ls='dashed', lw=1)
            plt.plot(range(t_max + 1), sim_x_mean[:, 1], label='Lazy, $E_m=' + str(Em) + '$', color='#C5004C', lw=1)


    plt.xlabel('t (number of semesters)', fontweight='bold', fontsize=9)
    plt.ylabel('Proportion', fontweight='bold', fontsize=9)
    plt.tick_params(labelsize=8)

    ax = fig.axes[0]



    legend_elements = [
        Line2D([0], [0], color='#DDDDDD', lw=1, ls='dotted', label='Hardworking'),
        Line2D([0], [0], color='#AAAAAA', lw=1, ls='dashed', label='Medium'),
        Line2D([0], [0], color='#444444', lw=1, label='Lazy'),
        Patch(facecolor='#008ECC',
              label='$E_m=0.2$'),
        Patch(facecolor='#C5004C',
              label='$E_m=0.8$')
    ]

    legend = plt.legend(handles=legend_elements, frameon=False, loc=0, fontsize=8)


    # legend.get_frame().set_facecolor('#ffffff')
    fig.savefig('figures/xyz_t.pdf', frameon=None)
    plt.show()