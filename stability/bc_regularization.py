"""Explicit raw BC coefficients for the paired Phase 7 sweep.

These multiply mean squared error in normalized action units. The Phase 6 B3
anchor used 10.0; values <=0.5 are not rescaled against Q magnitude.
"""

BC_WEIGHTS = {"L001": 0.01, "L005": 0.05, "L010": 0.1,
              "L050": 0.5, "E0": 10.0, "E1": 10.0, "E2": 10.0,
              "E3": 10.0, "G1": 10.0, "E1L": 10.0, "E2L": 10.0}
