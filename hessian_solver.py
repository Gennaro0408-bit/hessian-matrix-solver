import sympy as sp
import sys


def main():
    print("--- HESSIAN MATRIX SOLVER (INTERACTIVE MODE) ---")
    print("⚠️  ATTENZIONE ALLA SINTASSI TATTICA:")
    print("   - Per la potenza usa ** (es: x**2 per x al quadrato)")
    print("   - Per la moltiplicazione usa * (es: 3*x non 3x)")
    print("   - Esempio valido: x**3 - 3*x*y + y**2")
    print("----------------------------------------------------")

    # 1. Definisci le variabili simboliche
    x, y = sp.symbols('x y')

    # 2. CHIEDI AL GENERALE LA FUNZIONE (Input da tastiera)
    try:
        expr_str = input("\nInserisci la funzione f(x, y) > ")
        # sympify converte la stringa in matematica vera
        f = sp.sympify(expr_str)
    except (sp.SympifyError, SyntaxError):
        print("\n❌ ERRORE DI SINTASSI! Hai scritto male la formula.")
        print("Ricorda: usa * per moltiplicare e ** per le potenze.")
        sys.exit(1)

    print(f"\n✅ Funzione acquisita: {f}")

    # 3. Calcola il Gradiente (Derivate Prime)
    print("\n[1] CALCOLO GRADIENTE...")
    fx = sp.diff(f, x)
    fy = sp.diff(f, y)
    print(f"   fx = {fx}")
    print(f"   fy = {fy}")

    # 4. Trova i Punti Critici
    print("\n[2] RICERCA PUNTI CRITICI...")
    try:
        critical_points = sp.solve([fx, fy], (x, y), dict=True)
    except NotImplementedError:
        print("❌ Il sistema è troppo complesso per il risolutore automatico.")
        sys.exit(1)

    if not critical_points:
        print("   Nessun punto critico trovato (o impossibile da risolvere).")
        sys.exit(0)
    else:
        print(f"   Trovati {len(critical_points)} candidati.")

    # 5. Calcola l'Hessiano (Derivate Seconde)
    print("\n[3] ANALISI HESSIANO...")
    fxx = sp.diff(fx, x)
    fyy = sp.diff(fy, y)
    fxy = sp.diff(fx, y)

    print(f"   Hessiano Generale: [[{fxx}, {fxy}], [{fxy}, {fyy}]]")

    # 6. Analisi puntuale
    print("\n[4] VERDETTO FINALE:")

    # critical_points con dict=True restituisce una lista di dizionari [{x: 1, y: 0}, ...]
    for i, point in enumerate(critical_points):
        # Estraiamo i valori (a volte possono essere complessi o espressioni)
        val_x = point[x]
        val_y = point[y]

        # Saltiamo i numeri immaginari (complessi) se ci sono
        if not (val_x.is_real and val_y.is_real):
            print(f"\n-> Punto {i + 1}: ({val_x}, {val_y}) -> IGNORATO (Numero Complesso)")
            continue

        # Sostituzione dei valori nelle derivate seconde
        h_det = (fxx.subs({x: val_x, y: val_y}) * fyy.subs({x: val_x, y: val_y})) - (
                    fxy.subs({x: val_x, y: val_y}) ** 2)
        val_fxx = fxx.subs({x: val_x, y: val_y})

        print(f"\n-> Punto {i + 1}: ({val_x}, {val_y})")
        print(f"   Determinante Hessiano = {h_det}")

        if h_det > 0:
            if val_fxx > 0:
                print("   🏆 CLASSIFICAZIONE: MINIMO RELATIVO (Sorriso 🙂)")
            elif val_fxx < 0:
                print("   🏆 CLASSIFICAZIONE: MASSIMO RELATIVO (Triste ☹️)")
        elif h_det < 0:
            print("   🐎 CLASSIFICAZIONE: PUNTO DI SELLA")
        else:
            print("   ❓ CLASSIFICAZIONE: CASO DUBBIO (Hessiano nullo)")


if __name__ == "__main__":
    main()
