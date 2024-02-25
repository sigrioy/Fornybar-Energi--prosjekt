import numpy as np



'----import filer fra energiberegning bolig -----'
Tuvec_tromso = np.load('/Users/theahollokken/Downloads/Sted1_T.npy')
besparelse_case2_T = np.load('/Users/theahollokken/Downloads/Sted1_Besparelse12.npy') 
besparelse_case3_T = np.load('/Users/theahollokken/Downloads/Sted1_Besparelse13.npy')
c1_forbruk_T  = np.load('/Users/theahollokken/Downloads/Sted1_ForbrukC1.npy')/1000  # få verdiene i kw slik at veridene er like som verdiene i sol og vind 
c2_forbruk_T = np.load('/Users/theahollokken/Downloads/Sted1_ForbrukC2.npy')/1000
c3_forbruk_T = np.load('/Users/theahollokken/Downloads/Sted1_ForbrukC3.npy')/1000


Tuvec_kristiansand= np.load('/Users/theahollokken/Downloads/Sted2_T.npy')
besparelse_case2_K = np.load('/Users/theahollokken/Downloads/Sted2_Besparelse12.npy')
besparelse_case3_K = np.load('/Users/theahollokken/Downloads/Sted2_Besparelse13.npy')
c1_forbruk_K = np.load('/Users/theahollokken/Downloads/Sted2_ForbrukC1.npy')/1000
c2_forbruk_K = np.load('/Users/theahollokken/Downloads/Sted2_ForbrukC2.npy')/1000
c3_forbruk_K = np.load('/Users/theahollokken/Downloads/Sted2_ForbrukC3.npy')/1000

'----import filer fra vind  -----'
P_tromso_kWh=(np.load('/Users/theahollokken/Downloads/power_tromso_kwh.npy'))
P_kristiansand_kWh=(np.load('/Users/theahollokken/Downloads/power_kristiansand_kwh.npy'))

'----import filer fra sol  -----'
stromproduksjon_tromso=(np.load('/Users/theahollokken/Downloads/stromproduksjon_tromso.npy'))/1000
stromproduksjon_kristiansand =(np.load('/Users/theahollokken/Downloads/stromproduksjon_kristiansand.npy'))/1000





'-- energiberegninger----'
'---2---'

#Beregning av energibesparelse med varmepumpe for Tromsø
energibesparelse_T = c1_forbruk_T - c2_forbruk_T

#Gjør det samme for Kristiansand
energibesparelse_K = c1_forbruk_K - c2_forbruk_K



'----3----'
#Tromsø
netto_stromproduksjon_T = stromproduksjon_tromso + P_tromso_kWh
netto_stromproduksjon_T_sum= np.sum(netto_stromproduksjon_T)


#Kristiansand
netto_stromproduksjon_K = stromproduksjon_kristiansand + P_kristiansand_kWh
netto_stromproduksjon_K_sum= np.sum(netto_stromproduksjon_K)



'--4----'
#For scenario 1 til 3 i Tromsø
netto_forbruk_T_c1 = (c1_forbruk_T - netto_stromproduksjon_T)
netto_forbruk_T_c1_sum= np.sum(netto_forbruk_T_c1)

netto_forbruk_T_c2 = (c2_forbruk_T - netto_stromproduksjon_T)
netto_forbruk_T_c2_sum= np.sum(netto_forbruk_T_c2)

netto_forbruk_T_c3 = (c3_forbruk_T - netto_stromproduksjon_T)
netto_forbruk_T_c3_sum= np.sum(netto_forbruk_T_c3)

#For scenario 1 til 3 i Krtistiansand 
netto_forbruk_K_c1 = (c1_forbruk_K - netto_stromproduksjon_K)
netto_forbruk_K_c1_sum= np.sum(netto_forbruk_K_c1)

netto_forbruk_K_c2 = (c2_forbruk_K - netto_stromproduksjon_K)
netto_forbruk_K_c2_sum= np.sum(netto_forbruk_K_c2)

netto_forbruk_K_c3 = (c3_forbruk_K - netto_stromproduksjon_K)
netto_forbruk_K_c3_sum= np.sum(netto_forbruk_K_c3)





'----5----'
"bergn hvor mye energi som må kjøpes totalt på ett år, positivt netto energiforbruk"
#For Tromsø
kjøpt_energi_T_c1 = np.maximum(netto_forbruk_T_c1[:],0)
kjøpt_energi_T_c2 = np.maximum(netto_forbruk_T_c2[:],0)
kjøpt_energi_T_c3 = np.maximum(netto_forbruk_T_c3[:],0)

#For Kristiansand 
kjøpt_energi_K_c1 = np.maximum(netto_forbruk_K_c1[:],0)
kjøpt_energi_K_c2 = np.maximum(netto_forbruk_K_c2[:],0)
kjøpt_energi_K_c3 = np.maximum(netto_forbruk_K_c3[:],0)




'-----6----'
"bergn hvor mye energi som kan selges totalt på ett år, positivt netto energiforbruk"
# For Tromsø scenario 1
solgt_energi_T_c1= np.minimum(netto_forbruk_T_c1[:],0)
solgt_energi_T_c2= np.minimum(netto_forbruk_T_c2[:],0)
solgt_energi_T_c3= np.minimum(netto_forbruk_T_c3[:],0)

#For Kristiansand
solgt_energi_K_c1 = np.minimum(netto_forbruk_K_c1[:],0)
solgt_energi_K_c2 = np.minimum(netto_forbruk_K_c2[:],0)
solgt_energi_K_c3 = np.minimum(netto_forbruk_K_c3[:],0)




'-----7 og 8 ----'
"Gitt at 10 m^2 solceller og 1 vindturbin hver produserer de mengedene strøm som ble beregnet i oppgave 2 og 3 "
#Tromsø- summerer opp array i case 1 og 2 (totalt energiforbruk ila 1 år i kw)
c3_forbruk_T_sum= sum(c3_forbruk_T)

#Krisitiansand- summerer opp array i case 1 og 2 (totalt energiforbruk ila 1 år i kw)
c3_forbruk_K_sum= sum(c3_forbruk_K)


#sum av sol og vind hver for seg i løpet av ett år 
#Tromsø
sol_T_sum= sum(stromproduksjon_tromso)
vind_T_sum= sum(P_tromso_kWh)

#Kristiansand
sol_K_sum=sum(stromproduksjon_kristiansand)
vind_K_sum= sum(P_kristiansand_kWh)




"0 vindturbiner, hvor mange m^2 solceller er nødvendig for produserer mer strøm per år enn det totale forbruket i scenario 3?"


per_celle_T= sum(stromproduksjon_tromso/10)
print(c3_forbruk_T_sum)
nodvenig_m2_T =  (c3_forbruk_T_sum/per_celle_T)
print("\n energi per m^2 solcelle tromsø i kwh: ", per_celle_T  )
print("\n nødvendig anntall solceller m^2 tromsø",  nodvenig_m2_T) 

per_celle_K= (sol_K_sum/10 )
nodvenig_m2_K = (c3_forbruk_K_sum/per_celle_K)
print("\n energi per m^2 solcelle kristianasnd i kwh: ", per_celle_K)

print("\n nødvendig anntall solceller m^2  kristiansand :" ,nodvenig_m2_K) 





"0 m^2 solceller, hvor mange vindturbiner nødvendig for produserer mer strøm per år enn det totale forbruket i scenario 3?"

nodvendig_antall_turbin_T= (c3_forbruk_T_sum/vind_T_sum)
print("\n nødvendig anntall turbin tromsø :", nodvendig_antall_turbin_T)  

nodvendig_antall_turbin_K= (c3_forbruk_K_sum/vind_K_sum)
print("\n nødvendig anntall turbin kristiansand :", nodvendig_antall_turbin_K)  





'----kostnader-----------'


# Definerer antall timer i et år
t = np.arange(1, 8761)

# Beregner strømprisen p_strøm for hver time t
p_strøm = 0.68 + 0.15 * np.cos((2 * np.pi * t / 8760) - (np.pi / 8))

# Beregner nettprisen p_nett for hver time t
p_nett = 0.65 - 0.15 * np.cos(2 * np.pi * t / 8760)

# Netto strømpris er summen av strømprisen og nettprisen
p_netto = p_strøm + p_nett

# Beregner gjennomsnittet av strømprisen over året
p_strøm_gjennomsnitt = sum(p_netto)/ 8760

# Beregner salgsprisen p_salg for hver time t
p_salg = np.sqrt(p_strøm / p_strøm_gjennomsnitt)

print("\n netto strømpris i kr/kwh:" , p_salg) #kr/kwh



"årlig kostnad for tre sceanorier i tromsø og kristiansand "


#Tromsø

aarlig_kostnad_T_c1 = sum(kjøpt_energi_T_c1 * p_netto)
aarlig_kostnad_T_c2 = sum(kjøpt_energi_T_c2 * p_netto)
aarlig_kostnad_T_c3 = sum(kjøpt_energi_T_c3 * p_netto)

print("\n årlig kostnad tromsø case 1",  aarlig_kostnad_T_c1)
print("årlig kostnad tromsø case 2",  aarlig_kostnad_T_c2)
print("årlig kostnad tromsø case 3",  aarlig_kostnad_T_c3)


#Kristiansand

aarlig_kostnad_K_c1 = sum(kjøpt_energi_K_c1 * p_netto)
aarlig_kostnad_K_c2 = sum(kjøpt_energi_K_c2 * p_netto)
aarlig_kostnad_K_c3 = sum(kjøpt_energi_K_c3 * p_netto)

print("\n årlig kostnad Kristiansand case 1",  aarlig_kostnad_K_c1)
print("årlig kostnad Kristiansand case 2",  aarlig_kostnad_K_c2)
print("årlig kostnad Kristiansand case 3",  aarlig_kostnad_K_c3)



"årlig inntekt for tre sceanorier i tromsø og kristiansand "

#Tromsø

aarlig_inntekt_T_c1 = sum(solgt_energi_T_c1 * p_salg)
aarlig_inntekt_T_c2 = sum(solgt_energi_T_c2 * p_salg)
aarlig_inntekt_T_c3 = sum(solgt_energi_T_c3 * p_salg)
 
print("\n årlig inntekt Tromsø case 1",  aarlig_inntekt_T_c1)
print("årlig inntekt Tromsø case 2",  aarlig_inntekt_T_c2)
print("årlig inntekt Tromsø case 3",  aarlig_inntekt_T_c3)


#Kristiansand

aarlig_inntekt_K_c1 = sum(solgt_energi_K_c1 * p_salg)
aarlig_inntekt_K_c2 = sum(solgt_energi_K_c2 * p_salg)
aarlig_inntekt_K_c3 = sum(solgt_energi_K_c3 * p_salg)
print("\n årlig inntekt kristiansand case 1",  aarlig_inntekt_K_c1)
print("årlig inntekt Kristiansand case 2",  aarlig_inntekt_K_c2)
print("årlig inntekt Kristiansand case 3",  aarlig_inntekt_K_c3)





'årlig kostnadsbesparlese ved innstallajon av varmepumpe'
besparelse12_T= sum(besparelse_case2_T * p_strøm)
besparelse13_T= sum(besparelse_case3_T * p_strøm)

besparelse12_K= sum(besparelse_case2_K * p_strøm)
besparelse13_K= sum(besparelse_case3_K * p_strøm)



print("\n kostnadsbesparelse case 1 til 2 Trosmø (varmepumpe): " , besparelse12_T)
print("kostnadsbesparelse case 1 til 3 Trosmø (varmepumpe): " , besparelse13_T)
print("kostnadsbesparelse case 1 til 2 Kristiansand (varmepumpe): ", besparelse12_K)
print("kostnadsbesparelse case 1 til 3 Kristiansand (varmepumpe): ", besparelse13_K)
            

            
'med 1 vindturbin'
besparelse1_v_T= sum((P_tromso_kWh)*p_strøm)


besparelse1_v_K= sum((P_kristiansand_kWh)*p_strøm)


'med solcelle'

besparelse1_s_T= sum((stromproduksjon_tromso)*p_strøm)


besparelse1_s_K= sum((stromproduksjon_kristiansand)*p_strøm)


print("\n årlig besparelse case 1 med vind Tromsø: ", besparelse1_v_T)
print("årlig besparelse case 1 med sol Tromsø: ", besparelse1_s_T)
print("årlig besparelse case 1 med vind Kristiansand: ", besparelse1_v_K)
print("årlig besparelse case 1 med sol Kristiansand: ", besparelse1_s_K)
     

print('\n')
      
      
      
'---nåverdiberegninger-------'
#Verdier fra oppgaven
I_sol = 2000  # kr/m^2
I_vind = 50000  # kr/turbin
I_pumpe = 9000  # kr

V_sol = 30  # kr/år/m^2
V_vind = 200  # kr/år/turbin
V_pumpe = 250  # kr/år

N_sol = 30  # år
N_vind = 15  # år
N_varmepumpe = 12  # år

r = 0.07  # 7%



#Nåverdi solcelle Tromsø
NV_sol_C1_T = -I_sol - (V_sol*(1-(1+r)**(-N_sol))/r) + (besparelse1_s_T*((1-(1+r)**(-N_sol)))/r)

print('Tromsø:')
print(f"Nåverdi for solceller scenario 1 : {NV_sol_C1_T} kr")

#Nåverdi solcelle Kristiansand:
NV_sol_C1_K = -I_sol - (V_sol*(1-(1+r)**(-N_sol))/r) + (besparelse1_s_K*((1-(1+r)**(-N_sol)))/r)

print('\n')
print('Kristiansand:')
print(f"Nåverdi for solceller scenario 1 : {NV_sol_C1_K} kr")

print('\n')



#Nåverdi vindturbin

NV_vind_C1_T = -I_vind - (V_vind*(1-(1+r)**(-N_vind))/r) + (besparelse1_v_T*((1-(1+r)**(-N_vind)))/r)


print('Tromsø')
print(f"Nåverdi for vindturbin i scenario C1: {NV_vind_C1_T} kr")


print ('\n')

NV_vind_C1_K = -I_vind - (V_vind*(1-(1+r)**(-N_vind))/r) + (besparelse1_v_K*((1-(1+r)**(-N_vind)))/r)



print('Kristiansand:')
print(f"Nåverdi for vindturbin i scenario C1: {NV_vind_C1_K} kr")

print ('\n')


#Nåverdli varmepumpe

NV_pumpe_T = -I_pumpe - (V_pumpe*(1-(1+r)**(-N_varmepumpe))/r) + (besparelse12_T*((1-(1+r)**(-N_varmepumpe)))/r)


print('Tromsø')
print(f"Nåverdi for varmepumpe: {NV_pumpe_T} kr")


print ('\n')


NV_pumpe_K = -I_pumpe - (V_pumpe*(1-(1+r)**(-N_varmepumpe))/r) + (besparelse12_K*((1-(1+r)**(-N_varmepumpe)))/r)



print('Kristiansand')
print(f"Nåverdi for varmepumpe: {NV_pumpe_K} kr")



print('\n')


'--klimagassberegninger---'
# Definere konstanter basert på oppgaven
ki_vind=0.78 #kw per turbin 
ki_sol = 3 #kw for ti solceller 
ki_varmepumpe= 4 #kw 
k_CO2_NO = (18.9/1000)/1000 # tonn CO2/kWh, emissions from Norwegian electricity production
k_CO2_EU = (300/1000)/1000   # tonn CO2/kWh, emissions from European electricity production
k_solceller = 7000/1000  # tonn CO2/kW, emissions from production/installation of solar cells
k_vindturbin = 700/1000  # tonn CO2/kW, emissions from production/installation of wind turbines
k_varmepumpe = 1800/1000  # tonn CO2/kW, emissions from production/installation of heat pumps
N=10 #antall år 


kjøpt_energi_T_c1_sum = sum(kjøpt_energi_T_c1)
kjøpt_energi_K_c1_sum = sum(kjøpt_energi_K_c1)

kjøpt_energi_T_c3_sum = sum(kjøpt_energi_T_c3)
kjøpt_energi_K_c3_sum = sum(kjøpt_energi_K_c3)



'regne ut totalt klimautslipp  ved kjøp av norskprodusert strøm ila 10 år ved case 1 uten varmepumpe,solcelle og vind'
K_NO_T = N * k_CO2_NO * kjøpt_energi_T_c1_sum #Tromsø
K_NO_K = N * k_CO2_NO * kjøpt_energi_K_c1_sum #Kristiansand  i gram 


'regne ut totalt klimautslipp  ved kjøp av europeisk-produsert strøm ila 10 år ved case 1 uten varmepumpe,solcelle og vind'
K_EU_T = N * k_CO2_EU * kjøpt_energi_T_c1_sum #Tromsø
K_EU_K = N * k_CO2_EU * kjøpt_energi_K_c1_sum  #Kristiansand



print(f"\n Totalt klimagassutslipp over 10 år ved kjøp av norskprodusert strøm i Tromsø: {K_NO_T} tonn CO2")
print(f"Totalt klimagassutslipp over 10 år ved kjøp av norskprodusert strøm i Kristiansand: {K_NO_K} tonn CO2")
print(f"Totalt klimagassutslipp over 10 år ved kjøp av europeiskprodusert strøm i Tromsø: {K_EU_T} tonn CO2")
print(f"Totalt klimagassutslipp over 10 år ved kjøp av europeiskprodusert strøm i Kristiansand: {K_EU_K} tonn CO2")


K_NO_T_c3 = N * k_CO2_NO * kjøpt_energi_T_c3_sum
K_NO_K_c3 = N * k_CO2_NO * kjøpt_energi_K_c3_sum



#Engangsutslipp for installerte solceller og vindturbiner
engangsutslipp_solceller = k_solceller * ki_sol     #Totalt engangsutslipp for solceller i kg
engangsutslipp_varmepumpe= k_varmepumpe *ki_varmepumpe #Totalt engangsutslipp for varmepumpe i kg
engangsutslipp_vindturbin = k_vindturbin * ki_vind  #Totalt engangsutslipp for vindturbin i kg


#Beregning av totalt engangsutslipp for installasjon
totalt_engangsutslipp = engangsutslipp_solceller + engangsutslipp_vindturbin + engangsutslipp_varmepumpe 


#Totalt klimagassutslipp over 10 år for scenario 3 (inkludert engangsutslipp + utslipp fra kjøpt strøm)
K_scenario3_NO_T = K_NO_T_c3 + totalt_engangsutslipp #Tromsø
K_scenario3_NO_K = K_NO_K_c3 + totalt_engangsutslipp #Kristiansand


#Printe ut de beregnede verdiene for scenario 3:
print(f"\n Totalt engangsutslipp fra installasjon av solceller, varmepumpe  og vindturbin: {totalt_engangsutslipp} tonn CO2")
print(f"Totalt klimagassutslipp over 10 år for scenario 3 i Tromsø: {K_scenario3_NO_T} tonn CO2")
print(f"Totalt klimagassutslipp over 10 år for scenario 3 i Kristiansand: {K_scenario3_NO_K} tonn CO2")






