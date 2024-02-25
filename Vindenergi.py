import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import calendar


"---------------------Oppgave 1--------------------"
#Data
vind_h = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
effekt = [0, 0, 0, 20, 50, 90, 165, 270, 360, 410, 540, 720, 760, 775, 780, 780, 780, 780, 780, 780, 780, 780, 780, 780, 780, 780, 0, 0, 0, 0, 0]

maks_effekt = 780  #W
maks_vind_h = 25  #m/s
s_areal = 2.27 #m^2
luft_t = 1.24  #kg/m^3

virkningsgrad_eksempel = 0.35  

#Tilgjengelig vindeffekt
tilgj_vind_effekt = [0.5 * luft_t * s_areal * v**3 * virkningsgrad_eksempel for v in vind_h]

#Beregner virkningsgrad og justerer for maks effekt og maks vindhastighet
virkningsgrad = [min(p / available, 1) * 100 if available > 0 and v <= maks_vind_h else 0 for v, p, available in zip(vind_h, effekt, tilgj_vind_effekt)]

#Plotting
plt.figure(figsize=(10, 6))
plt.plot(vind_h, effekt, label='Effekt Output (W)', marker='o')
plt.plot(vind_h, virkningsgrad, label='Virkningsgrad (%)', marker='o')
plt.xlabel('Vindhastighet (m/s)')
plt.ylabel('Effekt / Virkningsgrad')
plt.legend()
plt.title('Vindturbin effekt og virkningsgrad vs. vindhastighet')
plt.grid(True)
plt.show()



"---------------------Oppgave 2 og 3--------------------"
#Importerer CSV
#Tromsø
def f_vind_tromso(filnavn):
    data = pd.read_csv(filnavn, delimiter=';', decimal=',', header=0)
    tvec1 = pd.to_datetime(data.iloc[0:8760, 2], format='%d.%m.%Y %H:%M').to_numpy()
    Tuvec1 = data.iloc[0:8760, 3].to_numpy()
    return tvec1, Tuvec1

tvec1, Tuvec1 = f_vind_tromso('/Users/theahollokken/Desktop/Fornybar/Tromso_vind.csv')
print('oppgave 1')
print(tvec1)
print(Tuvec1)
print(len(tvec1))
print(len(Tuvec1))

fig, ax = plt.subplots()
ax.plot(tvec1, Tuvec1)
ax.set_title('Vindhastighet Tromsø')
plt.xlabel('Dato')
plt.ylabel('Hastighet m/s')
plt.show()

#Kristiansand
def f_vind_kristiansand(filnavn):
    data = pd.read_csv(filnavn, delimiter=';', decimal=',', header=0)
    tvec2 = pd.to_datetime(data.iloc[0:8760, 2], format='%d.%m.%Y %H:%M').to_numpy()
    Tuvec2 = data.iloc[0:8760, 3].to_numpy()
    return tvec2, Tuvec2

tvec2, Tuvec2 = f_vind_kristiansand('/Users/theahollokken/Desktop/Fornybar/Kristiansand_vind.csv')
print('oppgave 1')
print(tvec2)
print(Tuvec2)
print(len(tvec2))
print(len(Tuvec2))

fig, ax = plt.subplots()
ax.plot(tvec2, Tuvec2, color='red')
ax.set_title('Vindhastighet Kristiansand')
plt.xlabel('Dato')
plt.ylabel('Hastighet m/s')
plt.show()



"---------------------Oppgave 4--------------------"

#Funksjon for å beregne effekt output
def calculate_power(vind_h):
    return 0.5 * luft_t * s_areal * vind_h**3 * virkningsgrad_eksempel

#Vinddata for Tromsø
tvec1, Tuvec1 = f_vind_tromso('/Users/theahollokken/Desktop/Fornybar/Tromso_vind.csv')

#Interpolerer vindturbineffekt for Tromsø
P_tromso = np.interp(Tuvec1, vind_h, [calculate_power(v) for v in vind_h])

#Vinddata for Kristiansand
tvec2, Tuvec2 = f_vind_kristiansand('/Users/theahollokken/Desktop/Fornybar/Kristiansand_vind.csv')

#Interpolerer vindturbineffekt for Kristiansand
P_kristiansand = np.interp(Tuvec2, vind_h, [calculate_power(v) for v in vind_h])

#Plotter interpolert effekt for Tromsø
plt.figure(figsize=(10, 6))
plt.plot(tvec1, P_tromso, label='Tromsø')
plt.xlabel('Dato')
plt.ylabel('Effekt (W)')
plt.legend()
plt.title('Estimert vindturbineffekt - Tromsø')
plt.grid(True)
plt.show()

#Plotter interpolert effekt for Kristiansand
plt.figure(figsize=(10, 6))
plt.plot(tvec2, P_kristiansand, label='Kristiansand', color='green')
plt.xlabel('Dato')
plt.ylabel('Effekt (W)')
plt.legend()
plt.title('Estimert vindturbineffekt - Kristiansand')
plt.grid(True)
plt.show()

#Konverter effekten til kilowatt
P_tromso_kWh = np.array(P_tromso) / 1000
P_kristiansand_kWh = np.array(P_kristiansand) / 1000

#Lagre strømproduksjonen som en vektor
np.save('power_tromso_kWh.npy', P_tromso_kWh)
np.save('power_kristiansand_kWh.npy', P_kristiansand_kWh)





"---------------------Oppgave 5----------------------"

"Månedlig gjennomsnitt for vindhastighet"

#Vinddata for Tromsø and Kristiansand
tvec1, Tuvec1 = f_vind_tromso('/Users/theahollokken/Desktop/Fornybar/Tromso_vind.csv')
tvec2, Tuvec2 = f_vind_kristiansand('/Users/theahollokken/Desktop/Fornybar/Kristiansand_vind.csv')

#DataFrames
df_tromso = pd.DataFrame({'Date': tvec1, 'Wind Speed': Tuvec1})
df_kristiansand = pd.DataFrame({'Date': tvec2, 'Wind Speed': Tuvec2})

#Dato som index
df_tromso.set_index('Date', inplace=True)
df_kristiansand.set_index('Date', inplace=True)

#Data for Januar 2023 til Desember 2023
df_tromso_2023 = df_tromso['2023-01-01':'2023-12-31']
df_kristiansand_2023 = df_kristiansand['2023-01-01':'2023-12-31']

#Resample data til månedlig frekvens og beregn gjennomsnittlig vindhastighet
monthly_mean_tromso = df_tromso_2023.resample('M').mean()
monthly_mean_kristiansand = df_kristiansand_2023.resample('M').mean()

#Print
print("Månedlige gjennomsnitt for vindhastighet (m/s) for Tromsø (januar 2023 - desember 2023):")
print(monthly_mean_tromso['Wind Speed'])
print("\nMånedlige gjennomsnitt for vindhastighet (m/s) for Kristiansand (januar 2023 - desember 2023):")
print(monthly_mean_kristiansand['Wind Speed'])



"Månedlig sum av strømproduksjon"


#DataFrames
df_tromso = pd.DataFrame({'Date': tvec1, 'Power Output': P_tromso})
df_kristiansand = pd.DataFrame({'Date': tvec2, 'Power Output': P_kristiansand})

#Dato som index
df_tromso.set_index('Date', inplace=True)
df_kristiansand.set_index('Date', inplace=True)

#Data for Januar 2023 til Desember 2023
df_tromso_2023 = df_tromso['2023-01-01':'2023-12-31']
df_kristiansand_2023 = df_kristiansand['2023-01-01':'2023-12-31']

#Resample data til månedlig frekvens og beregn sum av strømproduksjon
monthly_sum_tromso = df_tromso_2023.resample('M').sum()
monthly_sum_kristiansand = df_kristiansand_2023.resample('M').sum()

#Print
print("Månedlig sum av strømproduksjon (kWh) for Tromsø (januar 2023 - desember 2023):")
print(monthly_sum_tromso['Power Output'])
print("\nMånedlig sum av strømproduksjon (kWh) for Kristiansand (januar 2023 - desember 2023):")
print(monthly_sum_kristiansand['Power Output'])


"Total strømproduksjon for ett år"

#Beregn total strømproduksjon for ett år
total_power_tromso = df_tromso_2023['Power Output'].sum()
total_power_kristiansand = df_kristiansand_2023['Power Output'].sum()

#Print
print("Total strømproduksjon for ett år for Tromsø (januar 2023 - desember 2023):", total_power_tromso, "kWh")
print("Total strømproduksjon for ett år for Kristiansand (januar 2023 - desember 2023):", total_power_kristiansand, "kWh")






"---------------------Oppgave 6--------------------"

#Beregn kapasitetsfaktor for Tromsø
kapasitetsfaktor_tromso = (monthly_sum_tromso['Power Output'].sum() / (maks_effekt * len(df_tromso_2023))) * 100

#Beregn kapasitetsfaktor for Kristiansand
kapasitetsfaktor_kristiansand = (monthly_sum_kristiansand['Power Output'].sum() / (maks_effekt * len(df_kristiansand_2023))) * 100

# Skriv ut resultatene
print("Kapasitetsfaktor for Tromsø (januar 2023 - desember 2023):", kapasitetsfaktor_tromso, "%")
print("Kapasitetsfaktor for Kristiansand (januar 2023 - desember 2023):", kapasitetsfaktor_kristiansand, "%")




"---------------------Oppgave 7--------------------"


#Funksjon for å konvertere månedsnummer til navn
def month_number_to_name(month_number):
    return calendar.month_abbr[month_number]

#Lag en figur for månedlige gjennomsnitt for vindhastighet
fig, ax_mnd_gjennomsnitt = plt.subplots(figsize=(12, 6))

#Resample vindhastigheter for Tromsø
resampled_tromso = df_tromso_2023.resample('H').mean()
mnd_gjennomsnitt_tromso = resampled_tromso.resample('M').mean()
ax_mnd_gjennomsnitt.plot(mnd_gjennomsnitt_tromso.index, monthly_mean_tromso['Wind Speed'], label='Tromsø', marker='o', color='blue')

#Resample vindhastigheter for Kristiansand
resampled_kristiansand = df_kristiansand_2023.resample('H').mean()
mnd_gjennomsnitt_kristiansand = resampled_kristiansand.resample('M').mean()
ax_mnd_gjennomsnitt.plot(mnd_gjennomsnitt_kristiansand.index, monthly_mean_kristiansand['Wind Speed'], label='Kristiansand', marker='o', color='orange')

ax_mnd_gjennomsnitt.set_xlabel('Dato')
ax_mnd_gjennomsnitt.set_ylabel('Månedlig gjennomsnittlig vindhastighet (m/s)')
ax_mnd_gjennomsnitt.legend()
ax_mnd_gjennomsnitt.set_title('Månedlig gjennomsnittlig vindhastighet - Januar 2023 til Desember 2023')
ax_mnd_gjennomsnitt.grid(True)

#Angi x-aksen til månedsnavn
month_labels = [month_number_to_name(month.month) for month in mnd_gjennomsnitt_tromso.index]
plt.xticks(mnd_gjennomsnitt_tromso.index, month_labels, rotation=45, ha='right')

plt.tight_layout()
plt.show()



"---------------------Oppgave 8--------------------"


#Funksjon for å konvertere månedsnummer til navn
def month_number_to_name(month_number):
    return calendar.month_abbr[month_number]


#Plotting monthly energy production as bar charts

#For Tromsø
plt.figure(figsize=(12, 6))
plt.bar(monthly_sum_tromso.index, monthly_sum_tromso['Power Output'], width=20, color='blue', label='Tromsø')
plt.xlabel('Måned')
plt.ylabel('Strømproduksjon (kWh)')
plt.title('Månedlig strømproduksjon - Tromsø')
plt.xticks(monthly_sum_tromso.index, [month_number_to_name(month.month) for month in monthly_sum_tromso.index], rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

#For Kristiansand
plt.figure(figsize=(12, 6))
plt.bar(monthly_sum_kristiansand.index, monthly_sum_kristiansand['Power Output'], width=20, color='purple', label='Kristiansand')
plt.xlabel('Måned')
plt.ylabel('Strømproduksjon (kWh)')
plt.title('Månedlig strømproduksjon - Kristiansand')
plt.xticks(monthly_sum_kristiansand.index, [month_number_to_name(month.month) for month in monthly_sum_kristiansand.index], rotation=45)
plt.legend()
plt.tight_layout()
plt.show()



