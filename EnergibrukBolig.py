
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sjekk lastet ned data:  brukes koma eller punktum

"Data for beregningen"
# Dimensjoner rom
R1_L = 9          # Lengde, rom 1 (m)
R1_B = 5.4        # Bredde, rom 1 (m)
R2_L = 6          # Lengde, rom 2 (m)
R2_B = 4.2        # Bredde, rom 2 (m)
R3_L = 3          # Lengde, rom 3 (m)
R3_B = 4.2        # Bredde, rom 3 (m)
RH = 3          # Etasjehøyde (m)

"Dimensjoner på¸rer og vinduer"
dor_B = 1          # Bredde, innerdører/ytterdør (m)
dor_H = 2          # Høyde, innerdører/ytterdør (m)
vindu_B = 0.15       # Bredde, vindu (m)
vindu_H = 0.25       # Høyde, vindu (m)

" Dimensjoner veggkonstruksjonen"
x_gips = 0.0125                   # Tykkelse, gipsplate (m)
x_stn = 0.148                    # Tykkelse, stendere (m)
x_ull = 0.148                    # Tykkelse, mineralull (m)
x_teglstein = 0.1                      # Tykkelse, teglstein (m)
x_cc = 0.6                      # Senteravstand, stendere (m)
x_B_stn = 0.036                    # Bredde, stendere (m)
x_B_ull = x_cc - x_B_stn           # Bredde, isolasjon (m)

"Parametre for varmeegenskaper"
R_i = 0.13                     # Overgangsresistans, innervegg (m2*K/W)
R_u = 0.04                     # Overgangsresistans, yttervegg (m2*K/W)
R_tak = 2                        # Varmeresistans, loft og tak (m2*K/W)
R_gulv = 3                        # Varmeresistans, gulv mot utsiden (m2*K/W)
k_gips = 0.2                      # Varmekonduktivitet, gipsplate (W/m/K)
k_stn = 0.120                    # Varmekonduktivitet, stendere (W/m/K)
k_ull = 0.034                    # Varmekonduktivitet, mineralull (W/m/K)
k_teglstein = 0.8                      # Varmekonduktivitet, teglstein (W/m/K)
U_kjolevegg = 0.15                     # U-verdi, kjøleromsvegg (W/m2/K)
U_kjoledor = 0.3                      # U-verdi, kjøleromsdør (W/m2/K)
U_utedor = 0.2                      # U-verdi, utedør (W/m2/K)
U_innervegg = 1                        # U-verdi, innervegg (W/m2/K)
U_vindu = 0.8                      # U-verdi, vinduer (W/m2/K)

"Parametre for luft"
T_o = 20            # Temperatur, oppholdsrom (C)
T_k = 4             # Temperatur, kjølerom (C)
luft_rho = 1.24          # Tetthet, luft (kg/m3)
luft_c = 1             # Varmekapasitet, luft (kJ/kg/K)


" Parametre for ventilasjon og infiltrasjon "
# Luftskifte, infiltrasjon sone 1 (1/h)
S1_infiltrasjon = 0.4
# Luftskifte, infiltrasjon sone 2 (1/h)
S2_infiltrasjon = 0.2
# Luftskifte, ventilasjon sone 1 (1/h)
S1_ventilasjon = 0.6
# Luftskifte, ventilasjon sone 2 (1/h)
S2_ventilasjon = 0
eta_vv = 0.8                      # Virkningsgrad, varmeveksler

" Parametre for teknologi for oppvarming og nedkøling "
eta_el = 1     # Virkningsgrad, direkte elektrisk oppvarming
COP_hp = 4     # COP-faktor varmepumpe hele året
COP_kj = 3     # COP-faktor kjølemaskin hele året

" BEREGNE AREAL vinduer"
vindu_Arute = vindu_B*vindu_H
vindu_Arute4 = 4*vindu_Arute
vindu_Arute8 = 8*vindu_Arute
S1_A_vindu = 5*vindu_Arute8 + 2*vindu_Arute4
S2_A_vindu = 2*vindu_Arute8 + 0*vindu_Arute4

" BEREGNE AREAL dører"
dor_A = dor_H*dor_B
S1_A_dor = 1*dor_A
S2_A_dor = 0*dor_A
S1_S2_A_dor = 1*dor_A

" BEREGNE AREAL veggflater inkludert dører/vinduer "
S1_A_yttervegg = (R1_B + R1_L + R1_B + R2_B + R2_L)*RH
S2_A_yttervegg = (R3_B + R3_L)*RH
S1_S2_A_vegg = (R3_B + R3_L)*RH

" Grensesjikt mellom temperatursoner ekskludert dÃ¸rer/vinduer "
S1_Atot_yttervegg = S1_A_yttervegg - S1_A_vindu - S1_A_dor
S2_Atot_yttervegg = S2_A_yttervegg - S2_A_vindu - S2_A_dor
S1_S2_Atot_vegg = S1_S2_A_vegg - S1_S2_A_dor

" Gulvareal og takareal "
S1_A_gulv = R1_B*R1_L + R2_B*R2_L
S2_A_gulv = R3_B*R3_L
S1_A_tak = S1_A_gulv
S2_A_tak = S2_A_gulv
S1_A_Br = S1_A_gulv  # Bruksareal
S2_A_Br = S2_A_gulv  # Bruksareal

" Grensesjikt mellom temperatursoner inkludert dører/vinduer "
S1_Atot_ut = S1_A_yttervegg + S1_A_tak + S1_A_gulv
S2_Atot_ut = S2_A_yttervegg + S2_A_tak + S2_A_gulv
S1_S2_Atot_gs = S1_S2_A_vegg

" Beregne volum "
S1_V = S1_A_Br*RH
S2_V = S2_A_Br*RH


#Plotting av CSV-filer

#Tromsø
def f_temp_tromso(filnavn):
    data = pd.read_csv(filnavn, delimiter=';', decimal=',', header=0)
    tvec_tromso = pd.to_datetime(
        data.iloc[0:8760, 2], format='%d.%m.%Y %H:%M').to_numpy()
    Tuvec_tromso = data.iloc[0:8760, 3].to_numpy()
    return tvec_tromso,Tuvec_tromso


tvec_tromso, Tuvec_tromso = f_temp_tromso('/Users/theahollokken/Desktop/fornybar/Tromso.csv')
print('oppgave 1')
print(tvec_tromso)
print(Tuvec_tromso)
print(len(tvec_tromso))
print(len(Tuvec_tromso))

fig, ax = plt.subplots()
ax.plot(tvec_tromso, Tuvec_tromso)
ax.set_title('Lufttemperatur Tromsø')
plt.xlabel('Dato')
plt.ylabel('Temperatur (°C)')
plt.show()

#Kristiansand


def f_temp_kristiansand(filnavn):
    data = pd.read_csv(filnavn, delimiter=';', decimal=',', header=0)
    tvec_kristiansand = pd.to_datetime(
        data.iloc[0:8760, 2], format='%d.%m.%Y %H:%M').to_numpy()
    Tuvec_kristiansand = data.iloc[0:8760, 3].to_numpy()
    return tvec_kristiansand, Tuvec_kristiansand


tvec_kristiansand, Tuvec_kristiansand = f_temp_kristiansand('/Users/theahollokken/Desktop/fornybar/Kristiandsand.csv')
print('oppgave 1')
print(tvec_kristiansand)
print(Tuvec_kristiansand)
print(len(tvec_kristiansand))
print(len(Tuvec_kristiansand))

fig, ax = plt.subplots()
ax.plot(tvec_kristiansand, Tuvec_kristiansand, color='red')
ax.set_title('Lufttemperatur Kristiansand')
plt.xlabel('Dato')
plt.ylabel('Temperatur (°C)')
plt.show()

#Kombinert plot
fig, ax = plt.subplots()
ax.plot(tvec_tromso, Tuvec_tromso, label='Tromsø')
ax.plot(tvec_kristiansand, Tuvec_kristiansand, label='Kristiansand', color='red')

ax.set_title('Lufttemperatur Tromsø og Kristiansand')
plt.xlabel('Dato')
plt.ylabel('Temperatur (°C)')
plt.legend()
plt.show()


"U-verdimetoden"

# Funksjon for å beregne total varmeresistans

" U-verdimetoden"

# 1 Beregne U-verdi for isolasjonsseksjonen
R_ull = R_i + (x_gips/k_gips) + (x_ull/k_ull) + (x_teglstein/k_teglstein) + R_u
U_ull = 1/R_ull



# 2 Beregne U-verdi for stenderseksjonen
R_stn = R_i + (x_gips/k_gips) + (x_stn/k_stn) + (x_teglstein/k_teglstein) + R_u
U_stn = 1/R_stn



# 3 Beregne U-verdi for yttervegg i sone 1 og 2 uten vinduer/dører
n_ull1 = (2*R1_B + R1_L + R2_B + R2_L)/x_cc
n_stn1 = n_ull1 + 1

Uu_s1vegg = U_ull * ((n_ull1 * x_B_ull)/(n_ull1 * x_B_ull + n_stn1 * x_B_stn)) + U_stn * ((n_stn1 * x_B_stn)/(n_ull1 * x_B_ull + n_stn1 * x_B_stn))


n_ull2 = (R2_B + R3_L)/x_cc
n_stn2 = n_ull2 + 1

Uu_s2vegg = U_ull * ((n_ull2 * x_B_ull)/(n_ull2 * x_B_ull + n_stn2 * x_B_stn)) + U_stn * ((n_stn2 * x_B_stn)/(n_ull2 * x_B_ull + n_stn2 * x_B_stn))


" k-verdimetoden"

# 1 Beregne legert k-verdi for det inhomogene sjiktet
k_leg1 = k_ull * ((n_ull1 * x_B_ull)/(n_ull1 * x_B_ull + n_stn1 * x_B_stn)) + k_stn * ((n_stn1 * x_B_stn)/(n_ull1 * x_B_ull + n_stn1 * x_B_stn))
k_leg2 = k_ull * ((n_ull2 * x_B_ull)/(n_ull2 * x_B_ull + n_stn2 * x_B_stn)) + k_stn * ((n_stn2 * x_B_stn)/(n_ull2 * x_B_ull + n_stn2 * x_B_stn))

# 2. Beregne U-verdi for yttervegg i sone 1 og 2 uten vinduer/dører
Rk_1_vegg = R_i + (x_gips/k_gips) + (x_stn/k_leg1) + (x_teglstein/k_teglstein) + R_u
Uk_1_vegg = 1/Rk_1_vegg

Rk_2_vegg = R_i + (x_gips/k_gips) + (x_stn/k_leg2) + (x_teglstein/k_teglstein) + R_u
Uk_2_vegg = 1/Rk_2_vegg

" Gjennomsnitt av U-verdimetode og K-verdimetode"

U1_vegg = (1/2) * (Uu_s1vegg + Uk_1_vegg)
U2_vegg = (1/2) * (Uu_s2vegg + Uk_2_vegg)


" Beregne U-verdi for hele veggflatene"

U_S1_flate = U1_vegg * (S1_Atot_yttervegg/S1_A_yttervegg) + U_vindu * (S1_A_vindu/S1_A_yttervegg) + U_utedor * (S1_A_dor/S1_A_yttervegg)
U_S2_flate = U2_vegg * (S2_Atot_yttervegg/S2_A_yttervegg) + U_vindu * (S2_A_vindu/S2_A_yttervegg) + U_utedor * (S2_A_dor/S2_A_yttervegg)
U_S1_S2_flate = U_kjolevegg * (S1_S2_Atot_vegg /S1_S2_A_vegg) + U_kjoledor * (S1_S2_A_dor/S1_S2_A_vegg)

" Beregne U-verdier for hele grensen mellom ulike soner:"

U_s1 = U_S1_flate * (S1_A_yttervegg/S1_Atot_ut) + (1/R_tak) * (S1_A_tak/S1_Atot_ut) + (1/R_gulv) * (S1_A_gulv/S1_Atot_ut)
U_s2 = U_S2_flate * (S2_A_yttervegg/S2_Atot_ut) + (1/R_tak) * (S2_A_tak/S2_Atot_ut) + (1/R_gulv) * (S2_A_gulv/S2_Atot_ut)
U_s1_s2 = U_S1_S2_flate

"Termisk energibehov:"
# 1 Regn ut termisk energibehov for the forskjellige rommene
# # call data import
# Generere matrix som i boka
U = np.array([U_s1, U_s2, U_s1_s2])
A = np.array([S1_Atot_ut, S2_Atot_ut, S1_S2_A_vegg])
dT_tromso = np.column_stack([(T_o - Tuvec_tromso), (T_k - Tuvec_tromso), np.ones(len(Tuvec_tromso)) * (T_o - T_k)])
dT_kristiansand = np.column_stack([(T_o - Tuvec_kristiansand), (T_k - Tuvec_kristiansand), np.ones(len(Tuvec_kristiansand)) * (T_o - T_k)])
 # Konduksjon
dQ_tromso = U * A * dT_tromso
dQ_kristiansand = U * A * dT_kristiansand
                     


# 2 Ventilasjon uten varmeveksler

T_S1_dQ_ventilasjon_uvv = luft_rho * luft_c * \
    S1_ventilasjon * S1_V * dT_tromso[:,0]/3.600  # watt
T_S2_dQ_ventilasjon_uvv = luft_rho * luft_c * \
    S2_ventilasjon * S2_V * dT_tromso[:,1]/3.600

K_S1_dQ_ventilasjon_uvv = luft_rho * luft_c * \
    S1_ventilasjon * S1_V * dT_kristiansand[:,0]/3.600
K_S2_dQ_ventilasjon_uvv = luft_rho * luft_c * \
    S2_ventilasjon * S2_V * dT_kristiansand[:,1]/3.600
    
    

# 3 Ventilasjon med varmeveksler

T_S1_dQ_ventilasjon_mvv = T_S1_dQ_ventilasjon_uvv * \
    (1 - eta_vv)  # Ventilasjon med varmeveksler, Lokasjon 1
T_S2_dQ_ventilasjon_mvv = T_S2_dQ_ventilasjon_uvv * \
    (1 - eta_vv)  # Ventilasjon med varmeveksler, Lokasjon 2

K_S1_dQ_ventilasjon_mvv = K_S1_dQ_ventilasjon_uvv * \
    (1 - eta_vv)  # Ventilasjon med varmeveksler, Lokasjon 1
K_S2_dQ_ventilasjon_mvv = K_S2_dQ_ventilasjon_uvv * \
    (1 - eta_vv)  # Ventilasjon med varmeveksler, Lokasjon 2
    
    

# 4 Infiltrasjon
dT_tromso_sone1_inf =  luft_rho * luft_c *  S1_infiltrasjon * S1_V * dT_tromso[:,0] / 3.6
dT_tromso_sone2_inf = luft_rho * luft_c *  S2_infiltrasjon * S1_V * dT_tromso[:,1] / 3.6
dT_kristiansand_sone1_inf = luft_rho * luft_c *  S1_infiltrasjon * S1_V * dT_kristiansand[:,0] / 3.6
dT_kristiansand_sone2_inf = luft_rho * luft_c *  S2_infiltrasjon * S1_V * dT_kristiansand[:,1] / 3.6

# 5 Uten varmeveksler

T_S1_dQ_infiltrasjon_uvv = T_S1_dQ_ventilasjon_uvv + dT_tromso_sone1_inf
T_S2_dQ_infiltrasjon_uvv = T_S2_dQ_ventilasjon_uvv + dT_tromso_sone2_inf
K_S1_dQ_infiltrasjon_uvv = K_S1_dQ_ventilasjon_uvv + dT_kristiansand_sone1_inf
K_S2_dQ_infiltrasjon_uvv = K_S2_dQ_ventilasjon_uvv + dT_kristiansand_sone2_inf

# 6 Med varmeveksler
S1_tromso_mVV = T_S1_dQ_ventilasjon_mvv + dT_tromso_sone1_inf
S2_tromso_mVV = T_S2_dQ_ventilasjon_mvv + dT_tromso_sone2_inf
S1_kristiansand_mVV = K_S1_dQ_ventilasjon_mvv + dT_kristiansand_sone1_inf
S2_kristiansand_mVV = K_S2_dQ_ventilasjon_mvv + dT_kristiansand_sone2_inf




"--Totalt oppvarmings- og kjølebehov beregnet som netto oppvarmingsbehov --:"
# Regn ut netto totale oppvarmings- og kjølebehov

# 1 Uten varmeveksler
T_S1_total_oppvarmingsbehov_uvv = T_S1_dQ_infiltrasjon_uvv + dQ_tromso[:,2] + dQ_tromso[:,0]
T_S2_total_oppvarmingsbehov_uvv = T_S2_dQ_infiltrasjon_uvv + dQ_tromso[:,2] + dQ_tromso[:,1]
K_S1_total_oppvarmingsbehov_uvv = K_S1_dQ_infiltrasjon_uvv + dQ_kristiansand[:,2] + dQ_kristiansand[:,0]
K_S2_total_oppvarmingsbehov_uvv = K_S2_dQ_infiltrasjon_uvv + dQ_kristiansand[:,2] + dQ_kristiansand[:,1]

T_total_oppvarmingsbehov_uvv = T_S1_total_oppvarmingsbehov_uvv + \
    T_S2_total_oppvarmingsbehov_uvv
K_total_oppvarmingsbehov_uvv = K_S1_total_oppvarmingsbehov_uvv + \
    K_S2_total_oppvarmingsbehov_uvv


# 2 Med varmeveksler
T_S1_total_oppvarmingsbehov_mvv = S1_tromso_mVV + dQ_tromso[:,2] + dQ_tromso[:,0]
T_S2_total_oppvarmingsbehov_mvv = S2_tromso_mVV + dQ_tromso[:,2] + dQ_tromso[:,1] # HELT LIK BARE MVV VENT
K_S1_total_oppvarmingsbehov_mvv = S1_kristiansand_mVV + dQ_kristiansand[:,2] + dQ_kristiansand[:,0]
K_S2_total_oppvarmingsbehov_mvv = S2_kristiansand_mVV + dQ_kristiansand[:,2] + dQ_kristiansand[:,1]

T_total_oppvarmingsbehov_mvv = T_S1_total_oppvarmingsbehov_mvv + \
    T_S2_total_oppvarmingsbehov_mvv
K_total_oppvarmingsbehov_mvv = K_S1_total_oppvarmingsbehov_mvv + \
    K_S2_total_oppvarmingsbehov_mvv
    
    
    

"--elektrisk energibehov --:"
# Regn ut det elektriske energibehovet
# Kjølebehov

Qs1_KB_tromsoUvv = np.minimum(T_S1_total_oppvarmingsbehov_uvv,0)
Qs2_KB_tromsoUvv = np.minimum(T_S2_total_oppvarmingsbehov_uvv,0)
Qs1_KB_tromsoMvv = np.minimum(T_S1_total_oppvarmingsbehov_mvv,0)
Qs2_KB_tromsoMvv = np.minimum(T_S2_total_oppvarmingsbehov_mvv,0)

Qs1_KB_kristiansandUvv = np.minimum(K_S1_total_oppvarmingsbehov_uvv,0)
Qs2_KB_kristiansandUvv = np.minimum(K_S2_total_oppvarmingsbehov_uvv,0)
Qs1_KB_kristiansandMvv = np.minimum(K_S1_total_oppvarmingsbehov_mvv,0)
Qs2_KB_kristiansandMvv = np.minimum(K_S2_total_oppvarmingsbehov_mvv,0)

Qs1_VB_tromsoUvv = np.maximum(T_S1_total_oppvarmingsbehov_uvv,0)
Qs2_VB_tromsoUvv = np.maximum(T_S2_total_oppvarmingsbehov_uvv,0)
Qs1_VB_tromsoMvv = np.maximum(T_S1_total_oppvarmingsbehov_mvv,0)
Qs2_VB_tromsoMvv = np.maximum(T_S2_total_oppvarmingsbehov_mvv,0)

Qs1_VB_nristiansandUvv = np.maximum(K_S1_total_oppvarmingsbehov_uvv,0)
Qs2_VB_kristiansandUvv = np.maximum(K_S2_total_oppvarmingsbehov_uvv,0)
Qs1_VB_kristiansandMvv = np.maximum(K_S1_total_oppvarmingsbehov_mvv,0)
Qs2_VB_kristiansandMvv = np.maximum(K_S2_total_oppvarmingsbehov_mvv,0)



"--case 1 elektrisk oppvarming i sone 1, ingen varmegjenvinning i ventilasjon  --:"
# Tromso
c1_forbruk_T = (abs(Qs1_KB_tromsoUvv) + Qs1_VB_tromsoUvv) / eta_el + (abs(Qs2_KB_tromsoUvv) + Qs2_VB_tromsoUvv) / COP_kj

# Kristiansand
c1_forbruk_K = (abs(Qs1_KB_kristiansandUvv) + Qs1_VB_nristiansandUvv) / eta_el + (abs(Qs2_KB_tromsoUvv) + Qs2_VB_kristiansandUvv) / COP_kj




"--case 2, varmepumpe i sone 1, ingen varmegjenvinning i ventilasjon--:"
# tromso
c2_forbruk_T = (abs(Qs1_KB_tromsoUvv) + Qs1_VB_tromsoUvv) / COP_hp + (abs(Qs2_KB_tromsoUvv) + Qs2_VB_tromsoUvv) / COP_kj

# Kristiansand
c2_forbruk_K =(abs(Qs1_KB_kristiansandUvv) + Qs1_VB_nristiansandUvv) / COP_hp + (abs(Qs2_KB_tromsoUvv) + Qs2_VB_kristiansandUvv) / COP_kj




"--case 3, varmepumpe i sone 1 +  varmegjenvinning i ventilasjon--:"
# tromso
c3_forbruk_T = (abs(Qs1_KB_tromsoMvv) + Qs1_VB_tromsoMvv) / COP_hp + (abs(Qs2_KB_tromsoMvv) + Qs2_VB_tromsoMvv) / COP_kj

# kristiansand
c3_forbruk_K = (abs(Qs1_KB_kristiansandMvv) + Qs1_VB_kristiansandMvv) / COP_hp + (abs(Qs2_KB_tromsoMvv) + Qs2_VB_kristiansandMvv) / COP_kj



'--------besparelse-------'

# regn ut energibesparelse

#for innstallering av hp, case 1 og 2
#Tromsø
besparelse_case2_T = (c1_forbruk_T - c2_forbruk_T) / 1000

#Kristiandsand
besparelse_case2_K = (c1_forbruk_K- c2_forbruk_K) / 1000


#ved innstallering av hp og vv case 1 til 3
#Tromsø
besparelse_case3_T = (c1_forbruk_T - c3_forbruk_T) / 1000

#Kristiansand
besparelse_case3_K = (c1_forbruk_K - c3_forbruk_K) / 1000


print(f'Besparelse: {besparelse_case2_T}')
print(f'Besparelse: {besparelse_case2_K}')
print(f'Besparelse: {besparelse_case3_T}')
print(f'Besparelse: {besparelse_case3_K}')




"regne ut årlig gjennomsnitt for alle casene slik at jeg kan sammenligne de i rapporten med verdier i tabell "

# Beregn det årlige gjennomsnittet for hvert case ved å ta gjennomsnittet av det totale årlige forbruket
# For Tromsø
gjennomsnittlig_forbruk_c1_T = np.mean(c1_forbruk_T)
gjennomsnittlig_forbruk_c2_T = np.mean(c2_forbruk_T)
gjennomsnittlig_forbruk_c3_T = np.mean(c3_forbruk_T)

# For Kristiansand
gjennomsnittlig_forbruk_c1_K = np.mean(c1_forbruk_K)
gjennomsnittlig_forbruk_c2_K = np.mean(c2_forbruk_K)
gjennomsnittlig_forbruk_c3_K = np.mean(c3_forbruk_K)

# Skriv ut de årlige gjennomsnittene for Tromsø
print(f'Årlig gjennomsnittlig strømforbruk for Tromsø case 1: {gjennomsnittlig_forbruk_c1_T} kW')
print(f'Årlig gjennomsnittlig strømforbruk for Tromsø case 2: {gjennomsnittlig_forbruk_c2_T} kW')
print(f'Årlig gjennomsnittlig strømforbruk for Tromsø case 3: {gjennomsnittlig_forbruk_c3_T} kW')

# Skriv ut de årlige gjennomsnittene for Kristiansand
print(f'Årlig gjennomsnittlig strømforbruk for Kristiansand case 1: {gjennomsnittlig_forbruk_c1_K} kW')
print(f'Årlig gjennomsnittlig strømforbruk for Kristiansand case 2: {gjennomsnittlig_forbruk_c2_K} kW')
print(f'Årlig gjennomsnittlig strømforbruk for Kristiansand case 3: {gjennomsnittlig_forbruk_c3_K} kW')



#Opprett en tidslinje for x-aksedata 
tidslinje = np.arange(len(besparelse_case2_T))

#Opprett en figur og fire subplots
fig, axs = plt.subplots(4, 1, figsize=(12, 12))  # 4 rader, 1 kolonne

# Plot hver besparelse i sitt eget subplot
axs[0].plot(tidslinje, besparelse_case2_T)
axs[0].set_title('Energi Besparelse - Tromsø case 1 - 2')
axs[0].set_xlabel('Timer')
axs[0].set_ylabel('Energi Besparelse (kW)')

axs[1].plot(tidslinje, besparelse_case2_K)
axs[1].set_title('Energi Besparelse - Kristiansand case 1 - 2')
axs[1].set_xlabel('Timer')
axs[1].set_ylabel('Energi Besparelse (kW)')

axs[2].plot(tidslinje, besparelse_case3_T)
axs[2].set_title('Energi Besparelse - Tromsø case 1 - 3')
axs[2].set_xlabel('Timer')
axs[2].set_ylabel('Energi Besparelse (kW)')

axs[3].plot(tidslinje, besparelse_case3_K)
axs[3].set_title('Energi Besparelse - Kristiansand case 1 - 3')
axs[3].set_xlabel('Timer')
axs[3].set_ylabel('Energi Besparelse (kW)')

#Juster layout for å unngå overlap
plt.tight_layout()
plt.show()


#Skriv ut verdiene ved time 400 for å sjekke om alt stemmer
time_400_value_case2_T = besparelse_case2_T[399]
time_400_value_case2_K = besparelse_case2_K[399]
time_400_value_case3_T = besparelse_case3_T[399]
time_400_value_case3_K = besparelse_case3_K[399]


print(f'Verdi ved time 400 for Tromsø case 1 - 2: {time_400_value_case2_T}')
print(f'Verdi ved time 400 for Kristiansand case 1 - 2: {time_400_value_case2_K}')
print(f'Verdi ved time 400 for Tromsø case 1 - 3: {time_400_value_case3_T}')
print(f'Verdi ved time 400 for Kristiansand case 1 - 3: {time_400_value_case3_K}')





'-----daglig og månedsgjennomsnitt------'

data_arrays= [c1_forbruk_T,c1_forbruk_K,c2_forbruk_T,c2_forbruk_K,c3_forbruk_T,c3_forbruk_K]


# Funksjonen som beregner daglige gjennomsnitt for hver av de seks variablene
def calculate_daily_averages(data_arrays):
    # Anta at hver array har lengden 8760 (365 dager * 24 timer)
    E_behov_daglig = np.zeros((365, len(data_arrays)))  # Array med 365 dager for hver av de seks variablene
    
    # Gå gjennom hver dag i året
    for day in range(365):
        # Beregn dagens sum for hver av de seks variablene
        for array_index, data_array in enumerate(data_arrays):
            daily_sum = sum(data_array[day*24:(day+1)*24])
            E_behov_daglig[day, array_index] = daily_sum / 24  # Beregn og lagre dagens gjennomsnitt
    
    return E_behov_daglig



## Beregn daglige gjennomsnitt
daily_averages = calculate_daily_averages(data_arrays)



# Funksjon for å beregne månedlige gjennomsnitt
def beregn_manedlig_gjennomsnitt(data):
    # Opprett en liste for å lagre gjennomsnittsverdiene for hver måned
    manedlig_gjennomsnitt = []
    
    

    # Definer start- og sluttpunkt for hver måned
    manedsomfang = [(0, 744), (744, 1416), (1416, 2160), (2160, 2880), (2880, 3624), (3624, 4344),
                    (4344, 5088), (5088, 5832), (5832, 6552), (6552, 7296), (7296, 8016), (8016, 8760)]


    # Gå gjennom hver måned
    for start, slutt in manedsomfang:
        maned_data = data[start:slutt]
        dager_i_maned = (slutt - start) / 24  # Forutsetter 24 timer per dag
        manedssum = sum(maned_data)
        gjennomsnitt = manedssum / dager_i_maned
        manedlig_gjennomsnitt.append(gjennomsnitt)

    return manedlig_gjennomsnitt

maanedlig_gjennomsnitt_case1_T = beregn_manedlig_gjennomsnitt(c1_forbruk_T)
maanedlig_gjennomsnitt_case2_T = beregn_manedlig_gjennomsnitt(c2_forbruk_T)
maanedlig_gjennomsnitt_case3_T = beregn_manedlig_gjennomsnitt(c3_forbruk_T)

maanedlig_gjennomsnitt_case1_K = beregn_manedlig_gjennomsnitt(c1_forbruk_K)
maanedlig_gjennomsnitt_case2_K = beregn_manedlig_gjennomsnitt(c2_forbruk_K)
maanedlig_gjennomsnitt_case3_K = beregn_manedlig_gjennomsnitt(c3_forbruk_K)

"-----Plotting-----"


c1_forbruk_T_sum= sum(c1_forbruk_T)
print(c1_forbruk_T_sum)



'-----daglig og månedslig gjennomsnitt----'


"daglig gjennomsnitt plotting"

# Plot daglige gjennomsnitt for datasett 1, 3 og 5
plt.figure(figsize=(15, 7))
for i in [0, 2, 4]:  # indeksene for datasett 1, 3 og 5
    plt.plot(daily_averages[:, i], label=f'Datasett {i+1}')

plt.title('Daglige Gjennomsnitt - case 1 til 3 Tromsø')
plt.xlabel('Dag i Året')
plt.ylabel('Gjennomsnittlig Energiforbruk (kW)')
plt.legend()
plt.show()

# Plot daglige gjennomsnitt for datasett 2, 4 og 6
plt.figure(figsize=(15, 7))
for i in [1, 3, 5]:  # indeksene for datasett 2, 4 og 6
    plt.plot(daily_averages[:, i], label=f'Datasett {i+1}')

plt.title('Daglige Gjennomsnitt - case 1 til 3 Kristiansand')
plt.xlabel('Dag i Året')
plt.ylabel('Gjennomsnittlig Energiforbruk (kW)')
plt.legend()
plt.show()



"månedlig gjennomsnitt plotting"


# Månedsnavn for utskriftsformål
maneder = ["Januar", "Februar", "Mars", "April", "Mai", "Juni",
           "Juli", "August", "September", "Oktober", "November", "Desember"]



# Plotting månedlig gjennomsnitt Tromsø
stolpebredde = 0.2
indeks = np.arange(len(maneder))
plt.figure(figsize=(12, 8))

plt.bar(indeks - stolpebredde, maanedlig_gjennomsnitt_case1_T, stolpebredde, label='Case 1 - Tromsø', color='blue')
plt.bar(indeks, maanedlig_gjennomsnitt_case2_T, stolpebredde, label='Case 2 - Tromsø', color='orange')
plt.bar(indeks + stolpebredde, maanedlig_gjennomsnitt_case3_T, stolpebredde, label='Case 3 - Tromsø', color='green')

plt.xlabel('Måned')
plt.ylabel('Månedlig Gjennomsnitt (Wh)')
plt.title('Månedlig Gjennomsnitt for Strømforbruk (Tromsø)')
plt.xticks(indeks, maneder)
plt.legend()
plt.tight_layout()

plt.show()



# Plotting månedlig gjennomsnitt Kristiansand
plt.figure(figsize=(12, 8))

plt.bar(indeks - stolpebredde, maanedlig_gjennomsnitt_case1_K, stolpebredde, label='Case 1 - Kristiansand', color='blue')
plt.bar(indeks, maanedlig_gjennomsnitt_case2_K, stolpebredde, label='Case 2 - Kristiansand', color='orange')
plt.bar(indeks + stolpebredde, maanedlig_gjennomsnitt_case3_K, stolpebredde, label='Case 3 - Kristiansand', color='green')

plt.xlabel('Måned')
plt.ylabel('Månedlig Gjennomsnitt (Wh)')
plt.title('Månedlig Gjennomsnitt for Strømforbruk (Kristiansand)')
plt.xticks(indeks, maneder)
plt.legend()
plt.tight_layout()

plt.show()





' ------Lagre data i .npy file------'
#Last inn dataene fra filene
#Tuvec_tromso = np.load('Sted1_T.npy')
np.save('Sted1_T.npy', Tuvec_tromso)

np.save('Sted1_Besparelse12.npy', besparelse_case2_T)
np.save('Sted1_Besparelse13.npy', besparelse_case3_T)

#c1_forbruk_T  = np.load('Sted1_ForbrukC1.npy')
#c2_forbruk_T = np.load('Sted1_ForbrukC2.npy')
#c3_forbruk_T = np.load('Sted1_ForbrukC3.npy')
np.save('Sted1_ForbrukC1.npy', c1_forbruk_T)
np.save('Sted1_ForbrukC2.npy', c2_forbruk_T)
np.save('Sted1_ForbrukC3.npy', c3_forbruk_T)


#Tuvec_kristiansand= np.load('Sted2_T.npy')
np.save('Sted2_T.npy', Tuvec_kristiansand)

#besparelse_case2_K = np.load('Sted2_Besparelse12.npy')
#besparelse_case3_K = np.load('Sted2_Besparelse13.npy')
np.save('Sted2_Besparelse12.npy', besparelse_case2_K)
np.save('Sted2_Besparelse13.npy', besparelse_case3_K)

#c1_forbruk_K = np.load('Sted2_ForbrukC1.npy')
#c2_forbruk_K = np.load('Sted2_ForbrukC2.npy')
#c3_forbruk_K = np.load('Sted2_ForbrukC3.npy')
np.save('Sted2_ForbrukC1.npy', c1_forbruk_K)
np.save('Sted2_ForbrukC2.npy', c2_forbruk_K)
np.save('Sted2_ForbrukC3.npy', c3_forbruk_K)
