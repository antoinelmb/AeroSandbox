import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib.pyplot as plt
import csv
import os

span_base = 2.785
S_Cl = 4

span_range = np.linspace(span_base, 2 * span_base, 10)

output_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(output_dir, "batch_results_twist.csv")
twist_plot_path = os.path.join(output_dir, "twist_distributions.png")

# Collect all results across spans
results = []

# Plot setup for twist distributions
fig_twist, ax_twist = plt.subplots(figsize=(10, 6))

for span in span_range:
    print(f"span = {span:.3f} m")
    opti = asb.Opti()  # Initialize an optimization environment.

    N = 60  # Number of chord sections to optimize

    section_y = np.linspace(0, span, N)

    twist_root = opti.variable(init_guess=5)  # Twist at root (y=0) in degrees
    twist_slope = opti.variable(init_guess=-1)  # Linear gradient (deg per unit span)

    # Compute twists as a linear function of spanwise position
    twists = twist_root + twist_slope * section_y
    chord = opti.variable(init_guess=1, lower_bound=0.1, upper_bound=5)  # Chord length (constant across span)

    section = asb.Airfoil(coordinates= r'C:\Users\antoine.vittet\Desktop\MC2_60_6101_1.dat')

    wing = asb.Wing(
    symmetric=True,
    xsecs=[
        asb.WingXSec(
            xyz_le=[
                0,
                section_y[i],  # Our (known) span locations for each section.
                0,
            ],
            chord=chord,
            twist=twists[i], #0 OR twists[i]
            airfoil = section
        )
        for i in range(N)
    ],
    )

    airplane = asb.Airplane(  # Make an airplane object containing only this wing.
        wings=[wing]
    )

    alpha = 3 # We could chose to set alpha to a different value for example to the typical leeway angle at certain navigation case

    op_point = asb.OperatingPoint(
        velocity=1,  # Some fixed velocity; doesn't matter since we're working nondimensionally.
        alpha=alpha,
    )

    vlm = asb.VortexLatticeMethod(
        airplane=airplane,
        op_point=op_point,
        spanwise_resolution=1,
        chordwise_resolution=8,
    )

    aero = vlm.run()

    # CD_friction = 0.0021 * wing.area() # coeff provenant de simus xfoil

    opti.subject_to(
    [  # Constraints for linear twist distribution
        twist_root > 0,  # Root twist lower bound
        twist_root < 20,  # Root twist upper bound
        twist_root + twist_slope * section_y[-1] > -20,  # Ensure tip twist > 0
        twist_root + twist_slope * section_y[-1] < 20,  # Ensure tip twist < 20
        twist_slope <= 0,  # Negative slope for washout (decreasing twist)
        aero["CL"] > 0.1,
        aero["CL"] * wing.area() == S_Cl,  # Lift coefficient constraint
        aero["CD"] > 0,  # Drag must be positive (trivial, but helps the optimizer)
    ]
    )

    L_over_D = aero["CL"] / aero["CD"]

    opti.minimize(-L_over_D)

    sol = opti.solve()

    twist_res = sol(twists)
    twist_slope_res = sol(twist_slope)
    root_twist_res = sol(twist_root)
    tip_twist_res = float(twist_res[-1])
    alpha_res = sol(alpha)
    cl_res = sol(aero["CL"])
    cd_res = sol(aero["CD"])
    ld_res = float(cl_res) / float(cd_res)
    chord_res = sol(chord)
    surface_area_res = sol(wing.area())
    S_Cl_res = sol(wing.area() * aero["CL"])

    results.append({
        "span_m": float(span),
        "chord_m": float(chord_res),
        "surface_area_m2": float(surface_area_res),
        "twist_root_deg": float(root_twist_res),
        "twist_slope_deg_per_m": float(twist_slope_res),
        "tip_twist_deg": tip_twist_res,
        "alpha_deg": float(alpha_res),
        "CL": float(cl_res),
        "CD": float(cd_res),
        "L_over_D": ld_res,
        "S_CL": float(S_Cl_res),
    })

    # Add twist distribution to the combined plot
    section_y_np = np.linspace(0, float(span), N)
    ax_twist.plot(section_y_np, twist_res, label=f"span={span:.2f} m")

# Write CSV
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
print(f"Results saved to {csv_path}")

# Save twist distribution plot
ax_twist.set_xlabel("Span position y (m)")
ax_twist.set_ylabel("Twist (deg)")
ax_twist.set_title("Optimal twist distribution for different spans")
ax_twist.legend(loc="upper right")
ax_twist.grid(True)
fig_twist.tight_layout()
fig_twist.savefig(twist_plot_path, dpi=150)
plt.close(fig_twist)
print(f"Twist distribution plot saved to {twist_plot_path}")