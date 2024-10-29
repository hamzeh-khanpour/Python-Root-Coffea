import sympy as sp

# Define the symbols
Q2e, ye, yp, Ee, Ep, W = sp.symbols('Q2e ye yp Ee Ep W')

# Define the equation
equation = -Q2e + 2 * ye * yp * Ee * Ep + 2 * yp * Ep * sp.sqrt((ye * Ee) ** 2 + Q2e) * (1 - Q2e / (2 * Ee**2 * (1 - ye))) - W**2

# Solve for Q2e
solution = sp.solve(equation, Q2e)

# Print the solution
print("Analytical solution for Q2e:")
for sol in solution:
    sp.pprint(sol)
