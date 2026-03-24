import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib.pyplot as plt
import csv
import os

span_base = 2.785
S_Cl = 4

span_range = np.linspace(span_base, 2 * span_base, 10)

output_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(output_dir, "batch_results_no_twist.csv")

# Collect all results across spans
results = []

for span in span_range:
    print(f"span = {span:.3f} m")
    opti = asb.Opti()  # Initialize an optimization environment.

    N = 60  # Number of chord sections to optimize

    section_y = np.linspace(0, span, N)

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
            twist=0, #0 OR twists[i]
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
    [ 
        aero["CL"] > 0.1,
        aero["CL"] * wing.area() == S_Cl,  # Lift coefficient constraint
        aero["CD"] > 0,  # Drag must be positive (trivial, but helps the optimizer)
    ]
    )

    L_over_D = aero["CL"] / aero["CD"]

    opti.minimize(-L_over_D)

    sol = opti.solve()

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
        "alpha_deg": float(alpha_res),
        "CL": float(cl_res),
        "CD": float(cd_res),
        "L_over_D": ld_res,
        "S_CL": float(S_Cl_res),
    })

# Write CSV
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
print(f"Results saved to {csv_path}")