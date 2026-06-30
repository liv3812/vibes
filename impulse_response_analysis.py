#!/usr/bin/env python3
"""
Análise da Resposta ao Impulso — Suspensão Baja SAE (1 GDL Subamortecido)
Parâmetros lidos de modelagemSistema1GDL.ipynb
"""

import sys
import json
import re
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.stdout.reconfigure(encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# 1. EXTRAÇÃO DE PARÂMETROS DO NOTEBOOK
# ══════════════════════════════════════════════════════════════════════════════

def extrair_params_notebook(caminho: str) -> dict:
    """
    Percorre as células de código do .ipynb em ordem, avaliando
    atribuições simples (var = expr numérica) para recriar o namespace.
    """
    with open(caminho, encoding="utf-8") as f:
        nb = json.load(f)

    ns = {"np": np, "__builtins__": {}}

    # Prefixos de linha que devemos ignorar (evita executar código pesado/inválido)
    SKIP_STARTS = (
        "print", "plt", "from ", "import ", "def ", "class ",
        "sol", "xt", "vt", "at", "ct", "wd", "theta", "envelope",
        "indices", "idx", "x_pico", "t_pico", "delta_exp", "zeta_exp",
        "v0", "x0", "a0", "K_suav", "sgn_", "suspensao",
    )

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        for raw_line in src.splitlines():
            line = raw_line.split("#")[0].strip()
            if not line or "=" not in line:
                continue
            if any(line.startswith(s) for s in SKIP_STARTS):
                continue
            var, _, expr = line.partition("=")
            var  = var.strip()
            expr = expr.strip()
            # Só identificadores simples; pula t (array de tempo)
            if not re.match(r"^[a-zA-Z_]\w*$", var) or var == "t":
                continue
            try:
                val = eval(expr, {"__builtins__": {}, "np": np}, ns)
                if isinstance(val, (int, float)):
                    ns[var] = float(val)
            except Exception:
                pass

    return ns


CAMINHO_NB = "modelagemSistema1GDL.ipynb"
params = extrair_params_notebook(CAMINHO_NB)

# ── Parâmetros dianteira (baseline) ──────────────────────────────────────────
wn1_nb = params["wn1"]   # rad/s  (identificado pelo sensor)
xi1    = params["xi1"]   # adimensional
k1     = params["k1"]    # N/m   (k = wn² · m_notebook)
c1     = params["c1"]    # N·s/m (c = ξ · 2m · wn)

# ── Parâmetros traseira ───────────────────────────────────────────────────────
wn2_nb = params["wn2"]
xi2    = params["xi2"]
k2     = params["k2"]
c2     = params["c2"]

# ── Massa experimental medida em ensaio ───────────────────────────────────────
M_EXP = 36.0   # kg  (valor medido na balança — roda dianteira)

print("─" * 62)
print("PARÂMETROS EXTRAÍDOS DO NOTEBOOK")
print(f"  Dianteira: wn={wn1_nb:.3f} rad/s | ξ={xi1:.4f} | "
      f"k={k1:.1f} N/m | c={c1:.2f} N·s/m")
print(f"  Traseira:  wn={wn2_nb:.3f} rad/s | ξ={xi2:.4f} | "
      f"k={k2:.1f} N/m | c={c2:.2f} N·s/m")
print(f"  m experimental (dianteira): {M_EXP:.1f} kg")
print("─" * 62)


# ══════════════════════════════════════════════════════════════════════════════
# 2. DERIVAÇÃO ANALÍTICA COM SYMPY
# ══════════════════════════════════════════════════════════════════════════════

print("\n── Derivação simbólica de h(t) ──────────────────────────────────────")

t_s, m_s, wn_s, xi_s, wd_s = sp.symbols("t m omega_n xi omega_d", positive=True)

h_sym = (1 / (m_s * wd_s)) * sp.exp(-xi_s * wn_s * t_s) * sp.sin(wd_s * t_s)
dh_sym = sp.diff(h_sym, t_s)

# Fatorando: dh/dt = [exp(−ξωₙt)/(m·ωd)] · [ωd·cos(ωd·t) − ξωₙ·sin(ωd·t)]
# Zerando a parte oscilatória:
#   ωd·cos(ωd·t*) = ξωₙ·sin(ωd·t*)
#   tan(ωd·t*) = ωd / (ξωₙ)
#   t* = (1/ωd) · arctan(ωd / (ξωₙ))

t_star_expr = sp.atan(wd_s / (xi_s * wn_s)) / wd_s

print("h(t)   =  (1 / (m · ωd)) · exp(−ξωₙt) · sin(ωd·t)")
print("dh/dt  =  [exp(−ξωₙt) / (m·ωd)] · [ωd·cos(ωd·t) − ξωₙ·sin(ωd·t)]")
print("\nCondição dh/dt = 0  →  tan(ωd·t*) = ωd / (ξωₙ)")
print(f"Solução analítica:  t* = (1/ωd) · arctan(ωd / (ξωₙ))")


# ══════════════════════════════════════════════════════════════════════════════
# 3. FUNÇÕES AUXILIARES
# ══════════════════════════════════════════════════════════════════════════════

def freq_amortecida(wn, xi):
    return wn * np.sqrt(1.0 - xi**2)

def h_t(t_arr, m, wn, xi):
    wd = freq_amortecida(wn, xi)
    return (1.0 / (m * wd)) * np.exp(-xi * wn * t_arr) * np.sin(wd * t_arr)

def tempo_pico_analitico(wn, xi):
    wd = freq_amortecida(wn, xi)
    return np.arctan(wd / (xi * wn)) / wd

def h_pico(m, wn, xi):
    ts = tempo_pico_analitico(wn, xi)
    return float(h_t(ts, m, wn, xi))


# ══════════════════════════════════════════════════════════════════════════════
# 4. PARÂMETROS BASELINE E CENÁRIOS
# ══════════════════════════════════════════════════════════════════════════════

# Baseline: k e c do notebook + massa experimental
M0  = M_EXP
K0  = k1
C0  = c1
WN0 = np.sqrt(K0 / M0)
XI0 = C0 / (2.0 * M0 * WN0)
WD0 = freq_amortecida(WN0, XI0)

print(f"\n── Parâmetros baseline (dianteira + m experimental) ─────────────────")
print(f"   m={M0:.1f} kg | k={K0:.1f} N/m | c={C0:.2f} N·s/m")
print(f"   ωn={WN0:.3f} rad/s | fn={WN0/(2*np.pi):.3f} Hz | ξ={XI0:.4f}")
print(f"   t* = {tempo_pico_analitico(WN0, XI0)*1000:.2f} ms | "
      f"h(t*) = {h_pico(M0, WN0, XI0):.6f} 1/kg")


def _cen(label, m, k, c, baseline=False):
    wn = np.sqrt(k / m)
    xi = c / (2.0 * m * wn)
    return dict(label=label, m=m, k=k, c=c, wn=wn, xi=xi, baseline=baseline)


cenarios_k = [
    _cen("k −20%",      M0, K0*0.80, C0),
    _cen("k −10%",      M0, K0*0.90, C0),
    _cen("k  baseline", M0, K0,      C0, baseline=True),
    _cen("k +10%",      M0, K0*1.10, C0),
    _cen("k +20%",      M0, K0*1.20, C0),
]

cenarios_m = [
    _cen("m −20%",      M0*0.80, K0, C0),
    _cen("m −15%",      M0*0.85, K0, C0),
    _cen("m −10%",      M0*0.90, K0, C0),
    _cen("m  baseline", M0,      K0, C0, baseline=True),
]

cenarios_c = [
    _cen("c  baseline", M0, K0, C0,    baseline=True),
    _cen("c ×2",        M0, K0, C0*2),
    _cen("c ×3",        M0, K0, C0*3),
]

grupos = [
    ("Variação de Rigidez  k",        cenarios_k),
    ("Variação de Massa  m",           cenarios_m),
    ("Variação de Amortecimento  c",   cenarios_c),
]

PALETTES = [
    ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c", "#9467bd"],  # k
    ["#e377c2", "#ff7f0e", "#2ca02c", "#1f77b4"],             # m
    ["#1f77b4", "#ff7f0e", "#d62728"],                        # c
]


# ══════════════════════════════════════════════════════════════════════════════
# 5. FIGURAS
# ══════════════════════════════════════════════════════════════════════════════

t_vec  = np.linspace(0.0, 1.0, 5000)
h_base = h_pico(M0, WN0, XI0)

fig = plt.figure(figsize=(18, 14))
fig.suptitle(
    "Análise da Resposta ao Impulso — Suspensão Baja SAE (1 GDL Subamortecido)\n"
    f"Baseline: m = {M0:.0f} kg  |  k = {K0:.0f} N/m  |  c = {C0:.1f} N·s/m  |"
    f"  ωₙ = {WN0:.2f} rad/s  |  ξ = {XI0:.4f}",
    fontsize=12, fontweight="bold",
)

gs = GridSpec(2, 2, figure=fig, hspace=0.50, wspace=0.38)
ax_k   = fig.add_subplot(gs[0, 0])
ax_m   = fig.add_subplot(gs[0, 1])
ax_c   = fig.add_subplot(gs[1, 0])
ax_bar = fig.add_subplot(gs[1, 1])
axes_tempo = [ax_k, ax_m, ax_c]

todos = []  # coleta todos os cenários para gráfico de barras

for idx, (titulo, cenarios) in enumerate(grupos):
    ax  = axes_tempo[idx]
    pal = PALETTES[idx]

    for i, cen in enumerate(cenarios):
        ht = h_t(t_vec, cen["m"], cen["wn"], cen["xi"])
        ts = tempo_pico_analitico(cen["wn"], cen["xi"])
        hs = h_pico(cen["m"], cen["wn"], cen["xi"])

        lw = 2.6 if cen["baseline"] else 1.5
        ls = "-"  if cen["baseline"] else "--"

        ax.plot(t_vec, ht, color=pal[i], lw=lw, ls=ls, label=cen["label"])
        ax.plot(ts, hs, "o", color=pal[i], ms=6.5,
                markeredgecolor="black", markeredgewidth=0.8, zorder=5)

        todos.append(dict(
            label=cen["label"], ts=ts, hs=hs,
            wn=cen["wn"], xi=cen["xi"], baseline=cen["baseline"],
        ))

    ax.set_title(titulo, fontsize=11, fontweight="bold")
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("h(t)  [1/kg]")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(True, alpha=0.35)
    ax.set_xlim(0.0, 1.0)

# ── Gráfico de barras ─────────────────────────────────────────────────────────
labels_bar = [c["label"] for c in todos]
h_stars    = [c["hs"]    for c in todos]
cores_bar  = [
    "#2ca02c" if c["baseline"] else
    ("#d62728" if c["hs"] < h_base else "#4e9de3")
    for c in todos
]

x_pos = np.arange(len(todos))
ax_bar.bar(x_pos, h_stars, color=cores_bar, edgecolor="black", lw=0.5, alpha=0.88)
ax_bar.axhline(h_base, color="green", ls="--", lw=1.6,
               label=f"Baseline = {h_base:.5f} 1/kg")

for xi_pos, hs in zip(x_pos, h_stars):
    delta = (hs - h_base) / h_base * 100
    ax_bar.text(xi_pos, hs + h_base * 0.008, f"{delta:+.1f}%",
                ha="center", va="bottom", fontsize=6.5, rotation=90)

ax_bar.set_xticks(x_pos)
ax_bar.set_xticklabels(labels_bar, rotation=60, ha="right", fontsize=7)
ax_bar.set_ylabel("h(t*)  [1/kg]")
ax_bar.set_title("Comparação do Pico h(t*) — Todos os Cenários",
                 fontsize=11, fontweight="bold")
ax_bar.legend(fontsize=8.5)
ax_bar.grid(True, axis="y", alpha=0.35)

plt.savefig("impulse_response_analysis.png", dpi=150, bbox_inches="tight")
plt.show(block=False)
plt.pause(0.5)
print("\nFigura salva em: impulse_response_analysis.png")


# ══════════════════════════════════════════════════════════════════════════════
# 6. TABELA RESUMO
# ══════════════════════════════════════════════════════════════════════════════

W = 82
print("\n" + "═" * W)
print(f"{'TABELA RESUMO — RESPOSTA AO IMPULSO (RODA DIANTEIRA)':^{W}}")
print("═" * W)
print(f"{'Cenário':<22} {'t* (ms)':>8} {'h(t*)':>12} {'Δh (%)':>10} "
      f"{'fn (Hz)':>9} {'ξ':>8}")
print("─" * W)

for cen in todos:
    fn_hz  = cen["wn"] / (2.0 * np.pi)
    delta  = (cen["hs"] - h_base) / h_base * 100
    mark   = "  ◄ baseline" if cen["baseline"] else ""
    print(
        f"{cen['label']:<22}"
        f"{cen['ts']*1000:>8.2f} "
        f"{cen['hs']:>12.6f} "
        f"{delta:>+10.2f} "
        f"{fn_hz:>9.3f} "
        f"{cen['xi']:>8.4f}"
        f"{mark}"
    )

print("═" * W)
ts0 = tempo_pico_analitico(WN0, XI0)
print(f"\nBaseline  →  m={M0:.1f} kg | k={K0:.1f} N/m | c={C0:.2f} N·s/m")
print(f"             ωn={WN0:.3f} rad/s | fn={WN0/(2*np.pi):.3f} Hz | ξ={XI0:.4f}")
print(f"             t* = {ts0*1000:.2f} ms | h(t*) = {h_base:.6f} 1/kg")

# ══════════════════════════════════════════════════════════════════════════════
# ANÁLISE COMBINADA: PERMUTAÇÃO ENTRE MASSA E RIGIDEZ
# ══════════════════════════════════════════════════════════════════════════════

# Variações desejadas
var_m = np.array([-0.30, -0.25, -0.20, -0.15, -0.10, 0.00, 0.10, 0.20, 0.30])
var_k = np.array([-0.30, -0.25, -0.20, -0.15, -0.10, 0.00, 0.10, 0.20])

resultados = []

for dm in var_m:
    for dk in var_k:
        m = M0 * (1 + dm)
        k = K0 * (1 + dk)
        c = C0

        wn = np.sqrt(k / m)
        xi = c / (2 * m * wn)
        hp = h_pico(m, wn, xi)
        ts = tempo_pico_analitico(wn, xi)

        resultados.append({
            "dm": dm,
            "dk": dk,
            "m": m,
            "k": k,
            "c": c,
            "wn": wn,
            "fn": wn / (2*np.pi),
            "xi": xi,
            "hp": hp,
            "ts": ts,
            "delta_hp": (hp - h_base) / h_base * 100
        })

# Ordena pelos menores picos
resultados_ordenados = sorted(resultados, key=lambda x: x["hp"])

print("\n" + "═"*95)
print("MELHORES COMBINAÇÕES DE MASSA E RIGIDEZ — MENOR PICO h(t*)")
print("═"*95)
print(f"{'m (%)':>8} {'k (%)':>8} {'m (kg)':>10} {'k (N/m)':>12} {'ξ':>8} {'fn (Hz)':>9} {'h(t*)':>12} {'Δh (%)':>10}")
print("─"*95)

for r in resultados_ordenados[:12]:
    print(
        f"{r['dm']*100:>+8.0f} "
        f"{r['dk']*100:>+8.0f} "
        f"{r['m']:>10.2f} "
        f"{r['k']:>12.1f} "
        f"{r['xi']:>8.4f} "
        f"{r['fn']:>9.3f} "
        f"{r['hp']:>12.6f} "
        f"{r['delta_hp']:>+10.2f}"
    )

print("═"*95)


# ══════════════════════════════════════════════════════════════════════════════
# MATRIZES PARA MAPAS DE CALOR
# ══════════════════════════════════════════════════════════════════════════════

M_grid = np.zeros((len(var_m), len(var_k)))
XI_grid = np.zeros((len(var_m), len(var_k)))
HP_grid = np.zeros((len(var_m), len(var_k)))
DELTA_grid = np.zeros((len(var_m), len(var_k)))

for i, dm in enumerate(var_m):
    for j, dk in enumerate(var_k):
        m = M0 * (1 + dm)
        k = K0 * (1 + dk)
        c = C0

        wn = np.sqrt(k / m)
        xi = c / (2 * m * wn)
        hp = h_pico(m, wn, xi)

        XI_grid[i, j] = xi
        HP_grid[i, j] = hp
        DELTA_grid[i, j] = (hp - h_base) / h_base * 100


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA: MASSA x RIGIDEZ
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

x_labels = [f"{dk*100:+.0f}%" for dk in var_k]
y_labels = [f"{dm*100:+.0f}%" for dm in var_m]

# ---------------------------------------------------------------------------
# Mapa 1: pico da resposta
# ---------------------------------------------------------------------------
im0 = axes[0].imshow(DELTA_grid, cmap="RdYlGn_r", aspect="auto")

axes[0].set_title("Redução do Pico h(t*)\nPermutação Massa × Rigidez", fontweight="bold")
axes[0].set_xlabel("Variação de rigidez k")
axes[0].set_ylabel("Variação de massa m")
axes[0].set_xticks(np.arange(len(var_k)))
axes[0].set_yticks(np.arange(len(var_m)))
axes[0].set_xticklabels(x_labels)
axes[0].set_yticklabels(y_labels)

for i in range(len(var_m)):
    for j in range(len(var_k)):
        axes[0].text(
            j, i,
            f"{DELTA_grid[i, j]:+.1f}%",
            ha="center", va="center",
            fontsize=8, color="black"
        )

cbar0 = fig.colorbar(im0, ax=axes[0])
cbar0.set_label("Variação do pico em relação ao baseline (%)")


# ---------------------------------------------------------------------------
# Mapa 2: fator de amortecimento xi
# ---------------------------------------------------------------------------
im1 = axes[1].imshow(XI_grid, cmap="viridis", aspect="auto")

axes[1].set_title("Fator de Amortecimento ξ\nPermutação Massa × Rigidez", fontweight="bold")
axes[1].set_xlabel("Variação de rigidez k")
axes[1].set_ylabel("Variação de massa m")
axes[1].set_xticks(np.arange(len(var_k)))
axes[1].set_yticks(np.arange(len(var_m)))
axes[1].set_xticklabels(x_labels)
axes[1].set_yticklabels(y_labels)

for i in range(len(var_m)):
    for j in range(len(var_k)):
        axes[1].text(
            j, i,
            f"{XI_grid[i, j]:.3f}",
            ha="center", va="center",
            fontsize=8, color="white"
        )

cbar1 = fig.colorbar(im1, ax=axes[1])
cbar1.set_label("Fator de amortecimento ξ")

plt.suptitle(
    "Análise Combinada de Massa e Rigidez — Suspensão Baja SAE",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout()
plt.savefig("permutacao_massa_rigidez.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nFigura salva em: permutacao_massa_rigidez.png")
