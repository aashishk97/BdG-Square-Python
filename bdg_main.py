import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

L        = 12      
N        = L * L  
t        = 1.0   

U_VALUES = [4.0]
n_target = 0.875

DISORDER_LIST = [0.25, 1.0, 2.0, 3.0]

N_DIS    = 20
MAX_ITER = 1000
TOL      = 2e-5
MIXING   = 0.3

_xs = np.tile(np.arange(L), L)      # x-coord of site i  (periodic)
_ys = np.repeat(np.arange(L), L)

def idx(x, y):
    return int(x % L) + int(y % L) * L

_nn = np.array([[idx(_xs[i]+dx, _ys[i]+dy)
                  for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]]
                 for i in range(N)]) 

def build_H0(V_dis, mu, n_hartree, U):
    diag  = V_dis - mu - U * n_hartree / 2.0   # Hartree: mu_tilde = mu + |U|n/2
    H0    = np.diag(diag).astype(float)
    rows  = np.repeat(np.arange(N), 4)
    # print(rows)
    cols  = _nn.ravel()
    H0[rows, cols] = -t
    return H0

def solve_bdg(V_dis, U, verbose=False, seed=None):
    rng   = np.random.default_rng(seed)
    Delta = rng.uniform(0.05, 0.3, N)     
    n_i   = np.full(N, n_target)          
    mu    = -2.0                          

    E = Uvec = None

    for it in range(MAX_ITER):
        H0    = build_H0(V_dis, mu, n_i, U)
        Dm    = np.diag(Delta)
        HBdG  = np.block([[H0, Dm], [Dm, -H0]])

        u   = Uvec[:N, :]  
        v   = Uvec[N:, :]  
        pos = E > 0        

        Delta_new = U   * (u[:, pos] * v[:, pos]).sum(axis=1)
        n_new     = 2.0 * (v[:, pos]**2).sum(axis=1)

        n_avg  = n_new.mean()
        mu    += 1.5 * (n_target - n_avg)

        diff   = np.abs(Delta_new - Delta).max()
        Delta  = (1 - MIXING) * Delta + MIXING * Delta_new
        n_i    = (1 - MIXING) * n_i   + MIXING * n_new

        if verbose and it % 50 == 0:
            print(f"    iter {it:3d}  max|δΔ|={diff:.2e}  "
                  f"<n>={n_avg:.4f}  μ={mu:.4f}")
        if diff < TOL:
            if verbose:
                print(f"    Converged ({it} iters)  μ={mu:.4f}  <n>={n_avg:.4f}")
            break
        # print(it, diff)

    return Delta, n_i, E, Uvec, mu

def compute_dos(E, eta=0.08, n_pts=1000):
    w   = np.linspace(-10., 10., n_pts)

    u_sq = (Uvec[:N, :]**2).sum(axis=0)  

    dos = ((eta / np.pi) * u_sq / ((w[:, None] - E[None, :])**2 + eta**2)).sum(axis=1) / N

    dos = dos * 2.
    return dos, w


def spectral_gap(E):
    pos = E[E > 1e-10]
    return float(pos.min()) if len(pos) else 0.0


def order_parameter(Delta):
    return float(np.mean(np.abs(Delta)))

def superfluid_stiffness(E, Uvec):
    u, v   = Uvec[:N, :], Uvec[N:, :]
    pos    = E > 0
    neg    = E < 0

    j_x = np.zeros((N, N))
    for i in range(N):
        ip = idx(_xs[i]+1, _ys[i])
        im = idx(_xs[i]-1, _ys[i])
        j_x[i, ip] = +t     
        j_x[i, im] = -t     
                            

    v_p   = v[:, pos]        
    G_up  = v_p @ v_p.T      
    K_x   = np.zeros((N, N))
    for i in range(N):
        ip = idx(_xs[i]+1, _ys[i])
        K_x[i, ip] = -t;  K_x[ip, i] = -t
    Tx_up = np.sum(K_x * G_up)           
    diag_term = -2.0 * Tx_up / N         

    u_p, v_p2 = u[:, pos], v[:, pos]
    u_n, v_n  = u[:, neg], v[:, neg]

    A  = u_p.T @ j_x @ u_n      
    B  = v_p2.T @ j_x @ v_n     
    M  = A + B                  

    dE = E[pos, None] - E[None, neg]  
    Lambda_xx = 2.0 * (M**2 / dE).sum() / N

    return diag_term - Lambda_xx

np.random.seed(42)

all_results = {}

for U in U_VALUES:
    print(f"\n{'='*62}")
    print(f"  |U|/t = {U},   L = {L}×{L},   <n> = {n_target}")
    print(f"{'='*62}")

    for W in DISORDER_LIST:
        print(f"\n  V/t = {W}  ({N_DIS} realisations)")

        dos_list     = []
        delta_pool   = []     # all Delta_i values → histogram P(Delta)
        egap_list    = []
        dop_list     = []
        ds_list      = []
        delta_map_last = None

        # if W == 0:
        #     N_DIS = 1
        # else:
        #     N_DIS = N_DIS

        # print(N_DIS)

        for r in range(N_DIS):
            V_dis = W * np.random.uniform(-1.0, 1.0, N)
            # if r == 0:
            #     V_dis = 0.0 * np.random.uniform(-1.0, 1.0, N)
            # else:
            #     V_dis = W * np.random.uniform(-1.0, 1.0, N)

            # print(V_dis)
            Delta, n_i, E, Uvec, mu = solve_bdg(
                V_dis, U, verbose=False, seed=r * 1000 + int(W * 100))

            dos, w      = compute_dos(E)
            dos_list.append(dos)
            delta_pool.extend(np.abs(Delta).tolist())   
            egap_list.append(spectral_gap(E))
            dop_list.append(order_parameter(Delta))
            ds_list.append(superfluid_stiffness(E, Uvec))
            delta_map_last = np.abs(Delta).reshape(L, L)
        # print(delta_map_last)
        # print(dop_list)

        res = dict(
            dos       = np.mean(dos_list, axis=0),
            w         = w,
            delta_pool = np.array(delta_pool),
            E_gap     = np.mean(egap_list),
            DeltaOP   = np.mean(dop_list),
            Ds        = np.mean(ds_list),
            Delta_map = delta_map_last,
        )
        all_results[(U, W)] = res
        print(f"    E_gap = {res['E_gap']:.4f}t   "
              f"Δ_OP = {res['DeltaOP']:.4f}t   "
              f"D0_s/π = {res['Ds']:.4f}t")

U = U_VALUES[0]
W_vals = DISORDER_LIST
colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(W_vals)))

fig = plt.figure(figsize=(18, 11))
fig.suptitle(r"$|U|/t$" + f"= {U},  $L={L}$,  $\\langle n\\rangle={n_target}$",
    fontsize=13, y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

ax1 = fig.add_subplot(gs[0, 0])
for W, col in zip(W_vals, colors):
    d    = all_results[(U, W)]['delta_pool']
    dmax = max(d.max(), 0.5)
    bins = np.linspace(0, dmax, 30)
    ax1.hist(d, bins=bins, density=True, histtype='step',
             color=col, lw=2, label=f'$V$={W}$t$')

ax1.set_xlabel(r'$\Delta(r_i)/t$', fontsize=11)
ax1.set_ylabel(r'$P(\Delta)$',      fontsize=11)
ax1.set_title('Fig. 1 — Distribution of local $\\Delta$', fontsize=10)
ax1.legend(fontsize=8, framealpha=0.7)
ax1.set_xlim(left=0)
# ax1.set_ylim(0.0,2)

plt.savefig("various_disorder_distribution.png", bbox_inches='tight', dpi=500)

fig, ax2 = plt.subplots(figsize=(7, 5))

for W, col in zip(W_vals, colors):
    r   = all_results[(U, W)]
    dos = r['dos']
    ax2.plot(r['w'], dos, color=col, lw=1.8, label=f'$V$={W}$t$')
ax2.axvline(0, color='k', lw=0.6, ls='--', alpha=0.5)
ax2.set_xlabel(r'$\omega/t$',  fontsize=11)
ax2.set_ylabel(r'$N(\omega)$', fontsize=11)
ax2.set_title('Fig. 2 — DOS: persistent spectral gap', fontsize=10)
ax2.legend(fontsize=8, framealpha=0.7)
ax2.set_xlim(-8, 8)

plt.savefig("various_disorder_N(w).png", bbox_inches='tight', dpi=500)

fig, ax3a = plt.subplots(figsize=(7, 5))

E_gaps = [all_results[(U, W)]['E_gap']   for W in W_vals]
D_ops  = [all_results[(U, W)]['DeltaOP'] for W in W_vals]
ax3a.plot(W_vals, E_gaps, 'o-',  color='royalblue',  lw=2, ms=7,
          label=r'$E_{\rm gap}/t$')
ax3a.plot(W_vals, D_ops,  's--', color='tomato',     lw=2, ms=7,
          label=r'$\Delta_{\rm OP}/t$')
ax3a.set_xlabel(r'$V/t$',    fontsize=11)
ax3a.set_ylabel(r'Energy$/t$', fontsize=11)
ax3a.set_title('Fig. 3a — Gap & order parameter', fontsize=10)
ax3a.legend(fontsize=9)
ax3a.set_xlim(-0.1, max(W_vals)+0.1)
ax3a.set_ylim(bottom=0)

plt.savefig("Eg_phi_vs_disorder.png", bbox_inches='tight', dpi=500)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

for ax, W_map in zip(axes.flatten(), [0.25, 1.0, 2.0, 3.0]):
    dmap = all_results[(U, W_map)]['Delta_map']
    vmax = all_results[(U, 0.25)]['delta_pool'].max() * 1.1

    im = ax.imshow(dmap, origin='lower', cmap='inferno', vmin=0, vmax=vmax)

    ax.set_title(f'$\\Delta(r_i)$ map, $V={W_map}t$', fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])

    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.04)
    cbar.set_label(r'$\Delta/t$', fontsize=10)

plt.tight_layout()

plt.savefig("phi_real_space_map_various_disorder.png", bbox_inches='tight', dpi=500)

fig, ax4 = plt.subplots(figsize=(7, 5))


Ds_vals = [all_results[(U, W)]['Ds'] for W in W_vals]
ax4.plot(W_vals, E_gaps,  'o-',  color='royalblue',  lw=2, ms=7,
         label=r'$E_{\rm gap}/t$')
ax4.plot(W_vals, Ds_vals, '^--', color='darkorange', lw=2, ms=7,
         label=r'$D^0_s/\pi t$')
ax4.set_xlabel(r'$V/t$',    fontsize=11)
ax4.set_ylabel(r'Energy$/t$', fontsize=11)
ax4.set_title('Fig. 4 — Superfluid stiffness vs disorder', fontsize=10)
ax4.legend(fontsize=9)
ax4.set_xlim(-0.1, max(W_vals)+0.1)
ax4.set_ylim(bottom=0)

plt.savefig("Eg_superfluid_vs_disorder.png", bbox_inches='tight', dpi=500)

plt.show()

