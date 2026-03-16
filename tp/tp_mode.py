import math
import numpy as np
import matplotlib.pyplot as plt

# ===== Parámetros del problema =====
g = 9.81
H = 150.0
NP = 112002
delta = NP - 100000

L0 = (0.1/10000 * delta + 0.25) * H
m  = 40/10000 * delta + 50
K1 = 10/10000 * delta + 40
K2 = 1
omega = math.sqrt(K1/m)

# ==== Conversiones ====
TO_KMH = 3.6

# === Reducción de parámetros para visualización ===

FACTOR_REDUCCION = 10000
FACTOR_REDUCCION_REF = 100


# === Pasos para simular por euler y rk4 ===

def euler_step(u, v, h, k1=K1, k2=K2):
  return u + h*v, v + h*accel(u, k1, k2)

def rk4_step(u, v, h, k1=K1, k2=K2):
    q_1u = h * v
    q_1v = h * accel(u, k1, k2)

    q_2u = h * (v + q_1v/2)
    q_2v = h * accel(u + q_1u/2, k1, k2)

    q_3u = h * (v + q_2v/2)
    q_3v = h * accel(u + q_2u/2, k1, k2)

    q_4u = h * (v + q_3v)
    q_4v = h * accel(u + q_3u, k1, k2)

    u_next = u + (q_1u + 2*q_2u + 2*q_3u + q_4u) / 6.0
    v_next = v + (q_1v + 2*q_2v + 2*q_3v + q_4v) / 6.0
    return u_next, v_next

# --- Aceleración común para ambos métodos ---
def accel(y, k1=K1, k2=K2):
  return g if y <= L0 else  g - (k1/m)*((y - L0)**k2)


'''
Dada una función para simular y(t) indicadada por 'step'
devuelve una estimación del punto más bajo de la trayectoria
usando h como paso y t_max como máximo tiempo para simular y
hallar ymin
'''
def calcular_y_min(h, t_max=20.0, step=euler_step):
    u_abs = v_abs = t = u = v = 0.0
    entro = False
    max_error = float("-inf")
    while t < t_max:
        u_next, v_next = step(u,v, h)

        if (not entro) and (u > L0 or u_next > L0):
            entro = True
        if entro and v > 0 and v_next <= 0:
            α = v / (v - v_next)
            return u + α*(u_next - u)# (*)
            #return min(u, u_next)
        t, u, v = t + h, u_next, v_next

    return float('inf')

# === Cálculo de errores relativos para distintos pasos en ymin ===
'''
Estima error relativo en el punto más bajo de la trayecyoria
para el método de euler usando como referencia rungekutta
'''
def rel_err_euler(h,f, t_max):
  y_num= calcular_y_min(h,t_max, f)
  y_min_rk = calcular_y_min(h, t_max, rk4_step)
  return abs(y_num - y_min_rk) / abs(y_min_rk)

'''
Estima error relativo en el punto más bajo de la trayecyoria
para el método de rungekutta usando como referencia rungekutta
pero con un paso mucho más chico reducido por FACTOR_REDUCCION
'''
def rel_err_rk4(h,f, t_max):
  y_num = calcular_y_min(h,t_max, f)
  y_min_rk = calcular_y_min(h/FACTOR_REDUCCION, t_max, rk4_step)
  return abs(y_num - y_min_rk) / abs(y_min_rk)

# === Cálculo del error del método y orden experimental para h1 y h2 = h1/2 ===


'''
Simula un referencia para un método
usando rungekutta con el h_ref dado
'''
def simulate_ref_rk4(h_ref, t_end):
    return simulate_integrator(rk4_step, h_ref, t_end)
'''
Máximo error en y contra una referencia (T_ref, Y_ref).
Interpola la referencia en los tiempos del método a evaluar.
Evita extrapolación clippeando en los bordes.
'''
def max_error_y_against_ref(step_num, h, T_ref, Y_ref, t_end):
    # Simulación a evaluar
    T, Y, V, A = simulate_integrator(step_num, h, t_end)

    # Recorta referencia al rango común y evita extrapolación
    mask = (T_ref >= T[0]) & (T_ref <= T[-1])
    T_r = T_ref[mask]; Y_r = Y_ref[mask]
    if len(T_r) < 2:
        raise ValueError("La referencia no cubre el intervalo temporal de la simulación.")

    T_clip = np.clip(T, T_r[0], T_r[-1])
    Y_ref_i = np.interp(T_clip, T_r, Y_r)

    # Máximo error absoluto en y
    return np.max(np.abs(Y - Y_ref_i))

def orden_exp_y(step_num, h, T_ref, Y_ref, t_end):
    """
    Calcula el orden experimental de un método contra una referencia.
    OJO: Para un h lo compara a la mitad con otro h/2 y devuelve orden
    """
    e1 = max_error_y_against_ref(step_num, h,   T_ref, Y_ref, t_end)
    e2 = max_error_y_against_ref(step_num, h/2, T_ref, Y_ref, t_end)
    return math.log(e1/e2, 2), e1, e2

#  === Simuladores generales ===

def simulate_integrator(step_func, h, t_end):
    t=0.0; y=0.0; v=0.0
    T=[]; Y=[]; V=[]; A=[]
    while t <= t_end:
        T.append(t); Y.append(y); V.append(v); A.append(accel(y))
        y, v = step_func(y, v, h)
        t += h
    return np.array(T), np.array(Y), np.array(V), np.array(A)

#  Detectar 4 mínimos (v cruza de + a <= 0 tras tensar cuerda)
def detectar_minimos(T, Y, V):
    mins = []
    entro = False
    for i in range(1, len(T)):
        if (not entro) and (Y[i-1] > L0 or Y[i] > L0):
            entro = True
        if entro and V[i-1] > 0.0 and V[i] <= 0.0:
            dv = V[i-1] - V[i]
            alpha = V[i-1]/dv if dv != 0 else 0.0
            t_star = T[i-1] + alpha*(T[i]-T[i-1])
            y_star = Y[i-1] + alpha*(Y[i]-Y[i-1])
            mins.append((t_star, y_star))
            if len(mins) >= 4:
                break
    return mins


# === Simulador para hallar k1 y k2 óptimos sin restencia del aire ===

def simulate_first_fall(k1, k2, h=0.002, tmax=200.0):
    u, v, t = 0.0, 0.0, 0.0

    while t < tmax:
        u, v = rk4_step(u, v, h, k1, k2)
        t += h

        if  v < 0:
            a = accel(u, k1, k2)
            return {"y_max": u, "t": t, "a": a}

    return None
    
def find_k1_k2_pairs(k1_start, k1_end, k1_amount, k2_start, k2_end, k2_amount):
    pairs = []

    k1_values = np.linspace(k1_start, k1_end, k1_amount)
    k2_values = np.linspace(k2_start, k2_end, k2_amount)

    for k1 in k1_values:
      for k2 in k2_values:
          res = simulate_first_fall(k1, k2)
          if res is None:
              continue

          y_max = res["y_max"]
          a = res["a"]

          if y_max >= 0.9*H and y_max < H and abs(a) <= 2.5 * g:
              pairs.append({
                  "k1": k1,
                  "k2": k2,
                  "y_max": y_max,
                  "a": a,
                  "t": res["t"]
              })

    return pairs

# === Simulador para hallar k1 y k2 óptimos con restencia del aire ===
def viscous_force(v, c1, c2):
    if v > 0:
        return -c1 * (v ** c2)
    elif v < 0:
        return c1 * ((-v) ** c2)
    else:
        return 0.0

def accel_with_viscous_force(u, v, k1, k2, c1, c2):
    g = 9.81
    L0 = 55.503  # Calculado para NP = 112002
    m = 98.008   # Calculado para NP = 112002

    if u <= L0:
        elastic_force = 0
    else:
        elastic_force = k1 * (u - L0)**k2

    viscous_force = viscous_force(v, c1, c2)

    return g - (elastic_force/m) + (viscous_force/m)

def rk4_step_with_viscous_force(u, v, h, k1, k2, c1, c2):
    k1u = h * v
    k1v = h * accel_with_viscous_force(u, v, k1, k2, c1, c2)

    k2u = h * (v + 0.5*k1v)
    k2v = h * accel_with_viscous_force(u + 0.5*k1u, v + 0.5*k1v, k1, k2, c1, c2)

    k3u = h * (v + 0.5*k2v)
    k3v = h * accel_with_viscous_force(u + 0.5*k2u, v + 0.5*k2v, k1, k2, c1, c2)

    k4u = h * (v + k3v)
    k4v = h * accel_with_viscous_force(u + k3u, v + k3v, k1, k2, c1, c2)

    u_next = u + (k1u + 2*k2u + 2*k3u + k4u)/6
    v_next = v + (k1v + 2*k2v + 2*k3v + k4v)/6

    return u_next, v_next

def simulate_first_fall_with_viscous_force(k1, k2, c1, c2, h=0.002, tmax=200.0):
    u, v, t = 0.0, 0.0, 0.0
    positive = False
    i = 0

    while t < tmax:
        u, v = rk4_step_with_viscous_force(u, v, h, k1, k2, c1, c2)
        t += h
        i += 1

        if v > 1e-8:
            positive = True

        if positive and v < 0:
            a = accel_with_viscous_force(u, v, k1, k2, c1, c2)
            return {"y_max": u, "t": t, "a": a}

    return None

def find_k1_k2_pairs_with_viscous_force(k1_start, k1_end, k1_amount, k2_start, k2_end, k2_amount, c1, c2):
    pairs = []

    k1_values = np.linspace(k1_start, k1_end, k1_amount)
    k2_values = np.linspace(k2_start, k2_end, k2_amount)

    for k1 in k1_values:
        for k2 in k2_values:
            res = simulate_first_fall_with_viscous_force(k1, k2, c1, c2)
            if res is None:
                continue

            y_max = res["y_max"]
            a = res["a"]

            if y_max >= 0.9*H and y_max < H and abs(a) <= 2.5 * g:
                pairs.append({
                    "k1": k1,
                    "k2": k2,
                    "y_max": y_max,
                    "a": a,
                    "t": res["t"]
                })

    return pairs

# === Plots === 
def plot_analisis_h(inicio, fin, metodo_nombre, f_error):
    hs = np.logspace(math.log10(inicio), math.log10(fin), num=TAMAÑO_MUESTRA)
    erros = [f_error(h, euler_step, t_max=20.0) for h in hs]
    plt.figure(figsize=(7,5))
    plt.loglog(hs, erros, "o-", label=metodo_nombre, color = 'purple')
    plt.xlabel("Paso h")
    plt.ylabel("Error relativo en el punto mínimo")
    plt.title(f"Error relativo vs tamaño de paso ({metodo_nombre})")
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend()

    plt.axhline(0.001, color="red", linestyle="--", label="Error 0.1%")
    plt.legend()
    plt.show()

def plot_panel_with_ref(title, subtitle, T_ex, Y_ex, T_rk, Y_rk, T_eu, Y_eu, y_label):
    plt.figure(figsize=(9,5))

    plt.plot(T_ex, Y_ex, color="purple", linewidth=2, label=f"Referencia usada (h= {h/FACTOR_REDUCCION_REF})")
    # RK4: línea punteada + algunos puntos

    plt.scatter(T_rk[::5], Y_rk[::5], s=18, color="magenta",label=f"RK4 (h={h})")
    # Euler: puntos más espaciados
    plt.scatter(T_eu[::5], Y_eu[::5], s=22, color="royalblue", label=f"Euler (h={h})")

    plt.title(title, fontsize=17, loc="left", color="#004C99")
    plt.suptitle(subtitle, y=0.97, fontsize=11, color="gray")
    plt.xlabel("Tiempo [s]")
    plt.ylabel(y_label)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


# === Recursos usados en el informe ===
TAMAÑO_MUESTRA = 30
H_MIN = 0.00000119
H_MAX = 4

### Aálisis para Euler Explícito
plot_analisis_h(H_MIN, H_MAX, "Euler Explícito", rel_err_euler)

h0 = 0.003
print(f"Error relativo para h = {h0} es {rel_err_euler(h0, euler_step, 20.0)} (Euler vs RK4 referencia)")

h = 0.001
t_end = 6
T_ref, Y_ref, V_ref, A_ref = simulate_ref_rk4(h/FACTOR_REDUCCION_REF, t_end)


p_euler, e_h, e_h2 = orden_exp_y(euler_step, h, T_ref, Y_ref, t_end)
print(f"Orden Euler ~ {p_euler:.5f}  (e(h)={e_h}, e(h/2)={e_h2}) vs esperado e(h/2)= {e_h/2}")


### Análisis para Runge-Kutta 4

h1 = 0.29
print(f"Error relativo usando h = {h1} es {rel_err_rk4(h1, rk4_step, 20.0)} (RK4 vs RK4 referencia)")

def y_min_analitico():
    return L0 + (m*g + math.sqrt(m*m*g*g + 2*K1*m*g*L0)) / K1

def rel_err_rk42(h,f, t_max): # disclaimer para que no tarde tanto uso ymin aanalítico aunque se puede usar la ref perfectamente
  y_num = y_min_analitico()

  y_min_rk = calcular_y_min(h, t_max, rk4_step)
  return abs(y_num - y_min_rk) / abs(y_min_rk)

plot_analisis_h(H_MIN, H_MAX, "RungeKutta orden 4", rel_err_rk42)

## Para comparar métodos

h = 0.05  # MISMO paso para Euler y RK4 para comparar
T_ref1, Y_ref1, V_ref1, A_ref1 = simulate_integrator(rk4_step,  h/FACTOR_REDUCCION_REF, t_end=43)
mins = detectar_minimos(T_ref1, Y_ref1, V_ref1)

if len(mins) < 4:
    print("Ojo: no se encontraron 4 mínimos, subí t_end o revisá parámetros.")
# tiempo final = 4º mínimo + un margen
t_end = mins[3][0] + 1.0

# ===== 2) Simular los tres métodos hasta t_end =====

T_rk1, Y_rk1, V_rk1, A_rk1 = simulate_integrator(rk4_step,  h, t_end)
T_eu1, Y_eu1, V_eu1, A_eu1 = simulate_integrator(euler_step, h, t_end)

V_ref_kmh1 = V_ref1 * TO_KMH
V_rk_kmh1 = V_rk1 *  TO_KMH
V_eu_kmh1 = V_eu1 *  TO_KMH


plot_panel_with_ref(
    "Comparación con referencia usada",
    "Posición — 4 caídas sucesivas",
    T_ref1, Y_ref1, T_rk1, Y_rk1, T_eu1, Y_eu1,
    "Posición [m]"
)
plot_panel_with_ref(
    "Comparación con referencia usada",
    "Velocidad — 4 caídas sucesivas",
    T_ref1, V_ref_kmh1, T_rk1, V_rk_kmh1, T_eu1, V_eu_kmh1,
    "Velocidad [km/h]"
)
plot_panel_with_ref(
    "Comparación con referencia usada",
    "Aceleración — 4 caídas sucesivas",
    T_ref1, A_ref1, T_rk1, A_rk1, T_eu1, A_eu1,
    "Aceleración [m/s^2]"
)
