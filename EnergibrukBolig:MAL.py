import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


" Importering av utetemperatur"

# Sjekk lastet ned data:  brukes koma eller punktum   

"Data for beregningen"
# Dimensjoner rom
R1_L  = 9;          # Lengde, rom 1 (m)
R1_B  = 5.4;        # Bredde, rom 1 (m)
R2_L  = 6;          # Lengde, rom 2 (m)
R2_B  = 4.2;        # Bredde, rom 2 (m)
R3_L  = 3;          # Lengde, rom 3 (m)
R3_B  = 4.2;        # Bredde, rom 3 (m)
RH    = 3;          # Etasjehøyde (m)

"Dimensjoner på¸rer og vinduer"
dor_B       = 1;          # Bredde, innerdører/ytterdør (m)
dor_H       = 2;          # Høyde, innerdører/ytterdør (m)
vindu_B     = 0.15;       # Bredde, vindu (m)
vindu_H     = 0.25;       # Høyde, vindu (m)

" Dimensjoner veggkonstruksjonen"
x_gips          = 0.0125;                   # Tykkelse, gipsplate (m)
x_stn           = 0.148;                    # Tykkelse, stendere (m)
x_ull           = 0.148;                    # Tykkelse, mineralull (m)
x_teglstein     = 0.1;                      # Tykkelse, teglstein (m)
x_cc            = 0.6;                      # Senteravstand, stendere (m)
x_B_stn         = 0.036;                    # Bredde, stendere (m)
x_B_ull         = x_cc - x_B_stn;           # Bredde, isolasjon (m)

"Parametre for varmeegenskaper"
R_i             = 0.13;                     # Overgangsresistans, innervegg (m2*K/W)
R_u             = 0.04;                     # Overgangsresistans, yttervegg (m2*K/W)
R_tak           = 2;                        # Varmeresistans, loft og tak (m2*K/W)
R_gulv          = 3;                        # Varmeresistans, gulv mot utsiden (m2*K/W)
k_gips          = 0.2;                      # Varmekonduktivitet, gipsplate (W/m/K)
k_stn           = 0.120;                    # Varmekonduktivitet, stendere (W/m/K)
k_ull           = 0.034;                    # Varmekonduktivitet, mineralull (W/m/K)
k_teglstein 	= 0.8;                      # Varmekonduktivitet, teglstein (W/m/K)
U_kjolevegg     = 0.15;                     # U-verdi, kjøleromsvegg (W/m2/K)
U_kjoledor      = 0.3;                      # U-verdi, kjøleromsdør (W/m2/K)
U_utedor        = 0.2;                      # U-verdi, utedør (W/m2/K)
U_innervegg     = 1;                        # U-verdi, innervegg (W/m2/K)
U_vindu         = 0.8;                      # U-verdi, vinduer (W/m2/K)

"Parametre for luft"
T_o              = 20;            # Temperatur, oppholdsrom (C)
T_k              = 4;             # Temperatur, kjølerom (C)
luft_rho         = 1.24;          # Tetthet, luft (kg/m3)
luft_c           = 1;             # Varmekapasitet, luft (kJ/kg/K)



" Parametre for ventilasjon og infiltrasjon "
S1_infiltrasjon =   0.4;                      # Luftskifte, infiltrasjon sone 1 (1/h)
S2_infiltrasjon =   0.2;                      # Luftskifte, infiltrasjon sone 2 (1/h)
S1_ventilasjon  =   0.6;                      # Luftskifte, ventilasjon sone 1 (1/h)
S2_ventilasjon  =   0;                        # Luftskifte, ventilasjon sone 2 (1/h)
eta_vv          =   0.8;                      # Virkningsgrad, varmeveksler   

" Parametre for teknologi for oppvarming og nedkøling "
eta_el          = 1;     # Virkningsgrad, direkte elektrisk oppvarming
COP_hp          = 4;     # COP-faktor varmepumpe hele året
COP_kj          = 3;     # COP-faktor kjølemaskin hele året

" BEREGNE AREAL vinduer"
vindu_Arute     = vindu_B*vindu_H;
vindu_Arute4    = 4*vindu_Arute;
vindu_Arute8    = 8*vindu_Arute;
S1_A_vindu       = 5*vindu_Arute8 + 2*vindu_Arute4;
S2_A_vindu       = 2*vindu_Arute8 + 0*vindu_Arute4;

" BEREGNE AREAL dører"
dor_A         = dor_H*dor_B;
S1_A_dor      = 1*dor_A;
S2_A_dor      = 0*dor_A;
S1_S2_A_dor   = 1*dor_A;

" BEREGNE AREAL veggflater inkludert dører/vinduer "
S1_A_yttervegg = (R1_B + R1_L + R1_B + R2_B + R2_L)*RH;
S2_A_yttervegg = (R3_B + R3_L)*RH;
S1_S2_A_vegg   = (R3_B + R3_L)*RH;

" Grensesjikt mellom temperatursoner ekskludert dÃ¸rer/vinduer "
S1_Atot_yttervegg  = S1_A_yttervegg - S1_A_vindu - S1_A_dor;
S2_Atot_yttervegg  = S2_A_yttervegg - S2_A_vindu - S2_A_dor;
S1_S2_Atot_vegg	   = S1_S2_A_vegg - S1_S2_A_dor;

" Gulvareal og takareal "
S1_A_gulv       = R1_B*R1_L + R2_B*R2_L;
S2_A_gulv       = R3_B*R3_L;
S1_A_tak        = S1_A_gulv;
S2_A_tak        = S2_A_gulv;
S1_A_Br         = S1_A_gulv; # Bruksareal
S2_A_Br         = S2_A_gulv; # Bruksareal

" Grensesjikt mellom temperatursoner inkludert dører/vinduer "
S1_Atot_ut      = S1_A_yttervegg + S1_A_tak + S1_A_gulv;
S2_Atot_ut      = S2_A_yttervegg + S2_A_tak + S2_A_gulv;
S1_S2_Atot_gs   = S1_S2_A_vegg;

" Beregne volum "
S1_V            = S1_A_Br*RH;
S2_V            = S2_A_Br*RH;



" U-verdimetoden"

# 1 Beregne U-verdi for isolasjonsseksjonen

# 2 Beregne U-verdi for stenderseksjonen

# 3 Beregne U-verdi for yttervegg i sone 1 og 2 uten vinduer/dører


" k-verdimetoden"

# 1 Beregne legert k-verdi for det inhomogene sjiktet

# 2. Beregne U-verdi for yttervegg i sone 1 og 2 uten vinduer/dører



" Gjennomsnitt av U-verdimetode og K-verdimetode"




" Beregne U-verdi for hele veggflatene"


" Beregne U-verdier for hele grensen mellom ulike soner:"

"Termisk energibehov"
# 1 Regn ut termisk energibehov for the forskjellige rommene
 # call data import
# Generere matrix som i boka
 # Konduksjon
                                            
# 2 Ventilasjon uten varmeveksler

# 3 Ventilasjon med varmeveksler

# 4 Infiltrasjon

# 5 Uten varmeveksler

# 6 Med varmeveksler



"Totalt oppvarmings- og kjølebehov beregnet som netto oppvarmingsbehov"
# Regn ut netto totale oppvarmings- og kølebehov
# 1 Uten varmeveksler


# 2 Med varmeveksler


"Elektrisk energibehov"
# Regn ut det elektriske energibehovet
# 1 Kølebehov

# 3. UTEN VARMEVEKSLER (CASE 1 & 2)
# 3.1 El.forbruk for oppvarmings/kjølebehov: el i S1, HP i S2

# 3.2 El.forbruk for oppvarmings/kjølebehov: HP i S1, HP i S2


# 4. MED VARMEVEKSLER (CASE 3)
# 4.1 El.forbruk for oppvarmings/kjølebehov: HP i S1, HP i S2


"Besparelse "
# Regn ut energibesparelser



#print('\n \n')
#print('Besparelse strømbehov:')
#print(f'   Ved installering av HP (Case 1 til 2):          {round(Wbesp12aar)} kWh/år')
#print(f'   Ved installering av HP & VV (Case 1 til 3):     {round(Wbesp13aar)} kWh/år')



"Måndesgjennomsnitt"


# Calculate daily averages

# Calculate monthly averages


# Clear variables (optional in Python, as memory management is handled by Python's garbage collector)

# Display the monthly averages DataFrame


"Plotting"
# Lag grafer for å visualisere resultater

# Plotting daily averages


# Plotting monthly averages


# Lagre data i .npy file
#np.save('Sted1_T.npy', Tuvec)
#np.save('Sted1_Besparelse12.npy', Wbesp12)
#np.save('Sted1_Besparelse13.npy', Wbesp13)
#np.save('Sted1_ForbrukC1.npy', Wvarme_C1)
#np.save('Sted1_ForbrukC2.npy', Wvarme_C2)
#np.save('Sted1_ForbrukC3.npy', Wvarme_C3)

#