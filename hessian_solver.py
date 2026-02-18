import sympy as sp
import sys

def main():
    print("--- HESSIAN MATRIX SOLVER (INTERACTIVE MODE) ---")
    print("⚠️  SYNTAX GUIDE:")
    print("   - For power use ** (e.g., x**2 for x squared)")
    print("   - For multiplication use * (e.g., 3*x not 3x)")
    print("   - Valid example: x**3 - 3*x*y + y**2")
    print("----------------------------------------------------")

    # 1. Define symbolic variables
    x, y = sp.symbols('x y')

    # 2. ASK USER FOR FUNCTION
    try:
        expr_str = input("\nEnter function f(x, y) > ")
        # sympify converts string to math expression
        f = sp.sympify(expr_str)
    except (sp.SympifyError, SyntaxError):
        print("\n❌ SYNTAX ERROR! Invalid formula.")
        print("Remember: use * for multiplication and ** for power.")
        sys.exit(1)

    print(f"\n✅ Function loaded: {f}")

    # 3. Compute Gradient (First Derivatives)
    print("\n[1] COMPUTING GRADIENT...")
    fx = sp.diff(f, x)
    fy = sp.diff(f, y)
    print(f"   fx = {fx}")
    print(f"   fy = {fy}")

    # 4. Find Critical Points
    print("\n[2] FINDING CRITICAL POINTS...")
    try:
        # dict=True returns a list of dictionaries
        critical_points = sp.solve([fx, fy], (x, y), dict=True)
    except NotImplementedError:
        print("❌ System too complex for automatic solver.")
        sys.exit(1)

    if not critical_points:
        print("   No critical points found.")
        sys.exit(0)
    else:
        print(f"   Found {len(critical_points)} candidate(s).")

    # 5. Compute Hessian Matrix (Second Derivatives)
    print("\n[3] HESSIAN ANALYSIS...")
    fxx = sp.diff(fx, x)
    fyy = sp.diff(fy, y)
    fxy = sp.diff(fx, y) # Mixed derivative

    print(f"   General Hessian Matrix: [[{fxx}, {fxy}], [{fxy}, {fyy}]]")

    # 6. Point Classification
    print("\n[4] FINAL VERDICT:")
    
    for i, point in enumerate(critical_points):
        # Extract values
        val_x = point[x]
        val_y = point[y]
        
        # Skip complex numbers
        if not (val_x.is_real and val_y.is_real):
            print(f"\n-> Point {i+1}: ({val_x}, {val_y}) -> IGNORED (Complex Number)")
            continue

        # Substitute values into Hessian components
        h_det = (fxx.subs({x:val_x, y:val_y}) * fyy.subs({x:val_x, y:val_y})) - (fxy.subs({x:val_x, y:val_y})**2)
        val_fxx = fxx.subs({x:val_x, y:val_y})

        print(f"\n-> Point {i+1}: ({val_x}, {val_y})")
        print(f"   Hessian Determinant = {h_det}")

        if h_det > 0:
            if val_fxx > 0:
                print("   🏆 CLASSIFICATION: LOCAL MINIMUM (Smiley Face 🙂)")
            elif val_fxx < 0:
                print("   🏆 CLASSIFICATION: LOCAL MAXIMUM (Sad Face ☹️)")
        elif h_det < 0:
            print("   🐎 CLASSIFICATION: SADDLE POINT")
        else:
            print("   ❓ CLASSIFICATION: INCONCLUSIVE (Zero Determinant)")

if __name__ == "__main__":
    main()
