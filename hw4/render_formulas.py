"""Render the four TODO formulas to PNG via MiKTeX so the README does not
depend on GitHub's MathJax (which choked on the bmatrix row separator)."""
import os
os.environ['PATH'] = r'D:\MiKTeX\miktex\bin\x64;' + os.environ['PATH']

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = (
    r'\usepackage{amsmath}'
    r'\usepackage{amssymb}'
    r'\usepackage{bm}'
)
import matplotlib.pyplot as plt


# NOTE: bmatrix row separator is '\\' (two characters); written as r'\\' here.
formulas = [
    ('todo1_covariance',
     r'$\Sigma = R\,S\,S^{T}\,R^{T}$',
     0.75),
    ('todo2_jacobian',
     r'$J = \begin{bmatrix} f_x/Z & 0 & -f_x X / Z^{2} \\ 0 & f_y/Z & -f_y Y / Z^{2} \end{bmatrix}$',
     1.6),
    ('todo3_gaussian',
     r'$f(\mathbf{x}) = \dfrac{1}{2\pi\sqrt{|\Sigma|}} \exp\!\left(-\tfrac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^{\!\top}\, \Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$',
     1.1),
    ('todo4_alpha',
     r'$w_i(\mathbf{x}) = \alpha_i \cdot T_i, \qquad T_i = \prod_{j<i}(1-\alpha_j)$',
     0.85),
]

# Sanity print: confirm '\\' is exactly 2 chars in the bmatrix formula
print('todo2 backslashes before bmatrix break:',
      formulas[1][1].count('\\\\'))  # expect 1 occurrence of '\\'

for name, formula, h in formulas:
    fig = plt.figure(figsize=(10, h))
    fig.text(0.5, 0.5, formula, ha='center', va='center', fontsize=22)
    plt.axis('off')
    out = f'assets/formulas/{name}.png'
    plt.savefig(out, dpi=140, bbox_inches='tight',
                pad_inches=0.20, facecolor='white')
    plt.close()
    print(f'saved {out}')
