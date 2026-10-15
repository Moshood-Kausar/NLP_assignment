import time
import numpy as np
import matplotlib.pyplot as plt


# Parameters
n = 2                      
P = np.array([[ 0.9432,  0.0587, -0.1813],
              [ 0.0587,  0.8450, -0.1013],
              [-0.1813, -0.1013,  0.5519]])    # P from section 7.1 of the paper we used from

ELL_Q1 = 10                #context size for Q1
NUM_ELL_Q2 = 100           #context sizes for Q2 trial
ELL_START_Q2 = 10          #first context size in Q2
ELL_STEP_Q2 = 10           #increment between context sizes
ELL_Q2 = [ELL_START_Q2 + ELL_STEP_Q2 * k for k in range(NUM_ELL_Q2)]   

DT = 0.05                  #rk4 step size
MAX_STEPS = 5000           # to stop if not converged after this many steps
TOL = 1e-4                 

SEED_Q1 = 0
SEED_Q2 = 1


# Functions used by both Q1 and Q2
def p_norms(Y):
    return np.sqrt(np.einsum('il,ij,jl->l', Y, P, Y))


def random_tokens(ell, rng):
    X = rng.standard_normal((n + 1, ell))
    return X / p_norms(X)


def rhs(Y):
    G = Y.T @ P @ Y                         
    A = np.exp(G)
    A = A / (np.sqrt(n + 1) * A.sum(axis=1, keepdims=True))
    return Y @ A.T - Y * np.sum(A * G, axis=1)


def rk4_step(Y, dt):
    k1 = rhs(Y)
    k2 = rhs(Y + 0.5 * dt * k1)
    k3 = rhs(Y + 0.5 * dt * k2)
    k4 = rhs(Y + dt * k3)
    return Y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def consensus_gap(Y):
    return 1.0 - (Y.T @ P @ Y).min()


def potential(Y):
    return -0.5 * np.exp(Y.T @ P @ Y).sum()


def simulate(Y0, dt=DT, max_steps=MAX_STEPS, tol=TOL, store=False):
    Y = Y0.copy()
    traj = [Y.copy()]
    gaps = [consensus_gap(Y)]
    for k in range(1, max_steps + 1):
        Y = rk4_step(Y, dt)
        gaps.append(consensus_gap(Y))
        if store:
            traj.append(Y.copy())
        if gaps[-1] < tol:
            return Y, k, np.array(traj), np.array(gaps)
    return Y, None, np.array(traj), np.array(gaps)



# Q1
# print("Q1")
rng1 = np.random.default_rng(SEED_Q1)
 
#1a for the 10 random tokens on the ellipsoid
Y0 = random_tokens(ELL_Q1, rng1)
print("y^T P y at t=0:", np.round(p_norms(Y0) ** 2, 10))
print("consensus gap at t=0: %.4f" % consensus_gap(Y0))
 
#used RK4 to solve
_, steps, _, _ = simulate(Y0)
print("converged after %d steps (t = %.2f)" % (steps, steps * DT))
 
#had to run 50% longer than needed so the plots show the settled state
n_plot = int(1.5 * steps)
_, _, traj, _ = simulate(Y0, max_steps=n_plot, tol=-1.0, store=True)
t = DT * np.arange(traj.shape[0])
 
#distance from the ellipsoid over the whole run
on_ell = np.einsum('tim,ij,tjm->tm', traj, P, traj)
print("max |y_i^T P y_i - 1| over all tokens and steps: %.1e" % np.abs(on_ell - 1).max())
 
#to see where they end up
Yf = traj[-1]
spread = max(np.linalg.norm(Yf[:, i] - Yf[:, j])
             for i in range(ELL_Q1) for j in range(ELL_Q1))
print("max distance between final tokens: %.1e" % spread)
print("common final point:", np.round(Yf.mean(axis=1), 4))
 
#to replicate figure 1 of the paper
fig = plt.figure(figsize=(13, 5.5))
 
ax = fig.add_subplot(1, 2, 1, projection='3d')
w, vecs = np.linalg.eigh(P)
P_inv_sqrt = vecs @ np.diag(1 / np.sqrt(w)) @ vecs.T          # maps unit sphere to the ellipsoid
u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 40), np.linspace(0, np.pi, 20))
S = np.stack([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)])
E = np.einsum('ij,jab->iab', P_inv_sqrt, S)
ax.plot_wireframe(E[0], E[1], E[2], color='lightgray', linewidth=0.4, alpha=0.6)
for i in range(ELL_Q1):
    ax.plot(traj[:, 0, i], traj[:, 1, i], traj[:, 2, i], color='tab:blue', lw=1.5)
ax.scatter(*traj[0], s=50, c='white', edgecolors='k', depthshade=False, label='initial')
ax.scatter(*traj[-1], s=50, c='gray', edgecolors='k', depthshade=False, label='final')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')
ax.set_title('%d tokens on the ellipsoid $y^T P y = 1$' % ELL_Q1)
ax.legend(loc='upper left')
ax.set_box_aspect((1, 1, 1))
 
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(t, [potential(Y) for Y in traj], color='tab:blue')
ax2.set_xlabel('t')
ax2.set_ylabel('$V_P$')
ax2.set_title('Potential $V_P$')
ax2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('q1_figure1.png', dpi=150)
 
fig_c, ax_c = plt.subplots(figsize=(6.5, 4))
ax_c.semilogy(t, np.abs(on_ell - 1) + 1e-18)
ax_c.set_xlabel('t')
ax_c.set_ylabel('$|y_i^T P y_i - 1|$')
ax_c.set_title('Distance from the ellipsoid (one line per token)')
ax_c.grid(alpha=0.3)
fig_c.tight_layout()
fig_c.savefig('q1_on_ellipsoid.png', dpi=150)
print("saved q1_figure1.png and q1_on_ellipsoid.png")

# # Q2
# print()
# print("Q2 (tol = %g, dt = %g)" % (TOL, DT))
# rng2 = np.random.default_rng(SEED_Q2)
# results = []
# gap_curves = []
# t_start = time.time()
# print("%6s %10s %7s %8s" % ("ell", "converged", "steps", "t"))
# for ell in ELL_Q2:
#     Y0_q2 = random_tokens(ell, rng2)
#     _, steps_q2, _, gaps_q2 = simulate(Y0_q2)
#     gap_curves.append(gaps_q2)
#     ok = steps_q2 is not None
#     results.append((ell, int(ok), steps_q2 if ok else -1))
#     t_conv = steps_q2 * DT if ok else float('nan')
#     print("%6d %10s %7s %8.2f" % (ell, ok, steps_q2, t_conv), flush=True)
# print("Q2 run time: %.0f s" % (time.time() - t_start))
 
# res = np.array(results)
# np.savetxt('q2_results.csv', res, delimiter=',', fmt='%d',
#            header='ell,converged,steps', comments='')
 
# conv = res[:, 1] == 1
# fig2, ax3 = plt.subplots(figsize=(7, 4))
# ax3.plot(res[conv, 0], res[conv, 2], 'o-', ms=3)
# if (~conv).any():
#     ax3.plot(res[~conv, 0], np.full((~conv).sum(), MAX_STEPS), 'rx', label='not converged')
#     ax3.legend()
# ax3.set_xlabel('context size $\\ell$')
# ax3.set_ylabel('RK4 steps to convergence')
# ax3.set_title('Steps to consensus (dt = %g, tol = %g)' % (DT, TOL))
# ax3.grid(alpha=0.3)
# fig2.tight_layout()
# fig2.savefig('q2_steps_vs_ell.png', dpi=150)
# fig3, ax4 = plt.subplots(figsize=(7, 4.5))
# cmap = plt.get_cmap('viridis')
# for ell, g in zip(ELL_Q2, gap_curves):
#     ax4.semilogy(np.arange(len(g)), g, color=cmap((ell - ELL_Q2[0]) / max(ELL_Q2[-1] - ELL_Q2[0], 1)), lw=0.8)
# ax4.axhline(TOL, color='k', ls='--', lw=1, label='tol = %g' % TOL)
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(ELL_Q2[0], ELL_Q2[-1]))
# fig3.colorbar(sm, ax=ax4, label='context size $\\ell$')
# ax4.set_xlabel('RK4 step')
# ax4.set_ylabel('consensus gap  $1 - \\min_{ij} y_i^T P y_j$')
# ax4.set_title('Convergence of all %d runs' % len(ELL_Q2))
# ax4.legend(loc='lower left')
# ax4.grid(alpha=0.3)
# fig3.tight_layout()
# fig3.savefig('q2_gap_curves.png', dpi=150)
# print("saved q2_results.csv, q2_steps_vs_ell.png and q2_gap_curves.png")
 
 
plt.show()
 