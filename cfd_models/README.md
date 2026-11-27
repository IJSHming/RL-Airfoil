# 2D Airfoil Geometry Models for CFD Aerodynamic Validation (RL-Airfoil)

This directory contains 100% watertight (closed 2-manifold) extruded 2D airfoil STL surface meshes corresponding to the exact **Representative Median Runs** plotted in the paper (`figures/llm_malibo/llm_malibo_airfoil_shapes.pdf`), ready for high-fidelity RANS CFD validation.

## Coordinate System & Flow Conditions
- **X-axis**: Chordwise coordinate $x \in [0.0, 1.0]\,\text{m}$ (unit chord $c = 1.0\,\text{m}$).
- **Y-axis**: Airfoil camber and thickness normal direction (preserving exact Kulfan CST finite trailing-edge thickness $t_{\text{TE}}$).
- **Z-axis**: Quasi-2D spanwise extrusion $z \in [0.0, 0.1]\,\text{m}$ (extrusion span $b = 0.1\,\text{m} = 0.1c$, directly compatible with OpenFOAM 1-cell-thick 2D RANS / ANSYS Fluent / STAR-CCM+ symmetry-plane setups).
- **Operating Condition**: Incompressible subsonic flow at $Re = 3.5e+06$, angle of attack $\alpha = 3.5^\circ$ (e.g., sea-level air $\rho = 1.225\,\text{kg/m}^3$, $\mu = 1.789 \times 10^{-5}\,\text{Pa}\cdot\text{s}$, $U_\infty \approx 51.1\,\text{m/s}$ for $c = 1.0\,\text{m}$).
- **Chordwise Discretization**: 200 cosine-clustered nodes per surface (leading-edge & trailing-edge refinement) with analytical strip-triangulated end-caps (guaranteed zero self-intersection even for high-camber profiles).

## Model Inventory (Exact Paper Median Runs)

| File Name | Design | Selected Seed | NeuralFoil $L/D$ | $C_L$ | $C_D$ | $C_M$ | $t_{\max}/c$ | $t_{\text{TE}}/c$ | Watertight | Volume ($\text{m}^3$) | Surface Area ($\text{m}^2$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `airfoil_naca0012_baseline.stl` | Baseline NACA 0012 | `N/A` | 65.80 | 0.3901 | 0.00593 | 0.0005 | 12.01% | 0.255% | True | 0.008221 | 0.3686 |
| `airfoil_turbo_median.stl` | TuRBO-1 (Median) | `3407` | 200.14 | 0.8963 | 0.00448 | -0.0972 | 11.47% | 0.222% | True | 0.007311 | 0.3511 |
| `airfoil_rl_meta_turbo_median.stl` | RL-Meta-TuRBO (Median) | `3407` | 202.09 | 0.9109 | 0.00451 | -0.1058 | 13.62% | 0.250% | True | 0.008929 | 0.3840 |
| `airfoil_llm_meta_turbo_median.stl` | LLM-Meta-TuRBO (Median) | `9999` | 226.37 | 0.9599 | 0.00424 | -0.1341 | 13.02% | 0.282% | True | 0.008939 | 0.3840 |
| `airfoil_malibo_median.stl` | MALIBO (Median) | `42` | 181.06 | 0.8385 | 0.00463 | -0.1080 | 15.63% | 0.327% | True | 0.010788 | 0.4218 |

## CFD Meshing Recommendations
- **Turbulence Model**: SST $k$-$\omega$ (with $\gamma$-$Re_\theta$ transition model recommended to match NeuralFoil/XFoil natural transition physics at $Re = 3.5 \times 10^6$).
- **First Cell Wall Thickness**: For $Re = 3.5 \times 10^6$ and $c = 1.0\,\text{m}$, target $y^+ \le 1$ with $y_1 \approx 6.5 \times 10^{-6}\,\text{m}$ ($0.0065\,\text{mm}$), growth ratio $1.15\text{--}1.20$, 35–45 inflation layers.
- **Far-field Domain**: C-grid or O-grid with radius $\ge 20c$ ($20\,\text{m}$) and spanwise `symmetry` / `empty` boundary conditions on $z = 0$ and $z = 0.1\,\text{m}$.
