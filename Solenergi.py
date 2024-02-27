
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



" Importing solar radiation data"
# SJEKK SCV FORMAT! Det finnes utfordringen med "date" format

data_Tromso = pd.read_csv('/Users/sigridoyre/Documents/fornybar/prosjekt/solinnstrålingtromsø.csv', skiprows=9)
data_Kristiansand = pd.read_csv('/Users/sigridoyre/Documents/fornybar/prosjekt/solinnstrålingkristiansand.csv')

"-----------------------------Plotter solinnstråling iog temp--------------------------------"

# Plotting av CSV-filer

"Load solinnstråling data"
# Funksjon for å lese solinnstrålingsdata
def f_solinnstraling(filnavn):
    data = pd.read_csv(filnavn, delimiter=',', header=None, names=['time', 'G(i)', 'H_sun', 'T2m', 'WS10m', 'Int.'])
    data['time'] = pd.to_datetime(data['time'], format='%Y%m%d:%H%M', errors='coerce')
    data['G(i)'] = pd.to_numeric(data['G(i)'], errors='coerce')  # Konverterer verdier i 'G(i)' til tall
    return data



# Funksjon for å plotte solinnstrålingsdata
def plot_solinnstraling(data, sted, color):
    fig, ax = plt.subplots()
    ax.plot(data['time'], data['G(i)'], color=color)
    ax.set_title(f'Solinnstråling {sted}')
    plt.xlabel('Dato og tid')
    plt.ylabel('Solinnstråling (W/m^2)')
    plt.show()

# Solinnstrålingsdata for Tromsø
data_tromso = f_solinnstraling('/Users/sigridoyre/Documents/fornybar/prosjekt/solinnstrålingtromsø.csv')
plot_solinnstraling(data_tromso, 'Tromsø', 'yellow')

# Solinnstrålingsdata for Kristiansand 
data_kristiansand = f_solinnstraling('/Users/sigridoyre/Documents/fornybar/prosjekt/solinnstrålingkristiansand.csv')
plot_solinnstraling(data_kristiansand, 'Kristiansand', 'orange')

"Load temperatur data"
# Plotter temp i Tromsø
def f_temp_tromso(filnavn):
    data = pd.read_csv(filnavn, delimiter=';', decimal=',', header=0)
    tvec1 = pd.to_datetime(data.iloc[:, 2], format='%d.%m.%Y %H:%M').to_numpy()
    Tuvec1 = data.iloc[:, 3].to_numpy()  # Endret til 3
    return tvec1, Tuvec1

tvec1, Tuvec1 = f_temp_tromso('Temp.Tromso.csv')
#print('oppgave 1')
#print(tvec1)
#print(Tuvec1)
#print(len(tvec1))
#print(len(Tuvec1))

#fig, ax = plt.subplots()
#ax.plot(tvec1, Tuvec1)
#ax.set_title('Lufttemperatur Tromsø')
#plt.xlabel('Dato')
#plt.ylabel('Temperatur (°C)')
#plt.show()

# Plotter temp i Kristiansand
def f_temp_kristiansand(filnavn):
    data = pd.read_csv(filnavn, delimiter=';', decimal=',', header=0)
    tvec2 = pd.to_datetime(data.iloc[:, 2], format='%d.%m.%Y %H:%M').to_numpy()
    Tuvec2 = data.iloc[:, 3].to_numpy()  # Endret til 3
    return tvec2, Tuvec2

tvec2, Tuvec2 = f_temp_kristiansand('Temp.Kristiansand.csv')
#print('oppgave 1')
#print(tvec2)
#print(Tuvec2)
#print(len(tvec2))
#print(len(Tuvec2))

#fig, ax = plt.subplots()
#ax.plot(tvec2, Tuvec2, color='red')
#ax.set_title('Lufttemperatur Kristiansand')
#plt.xlabel('Dato')
#plt.ylabel('Temperatur (°C)')
#plt.show()

"---------------------------- Effektberegning-------------------------"


"Virkningsgrad"

def beregn_virkningsgrad(utetemperatur):
    base_virkningsgrad = 0.20  # Virkningsgrad ved 25°C
    panelets_temperatur = utetemperatur + 20

    if panelets_temperatur >= 25:  # Beregn virkningsgraden basert på panelets temperatur
        virkningsgrad = base_virkningsgrad + 0.003 * (panelets_temperatur - 25)
    else:
        virkningsgrad = base_virkningsgrad - 0.003 * (25 - panelets_temperatur)

    return max(0, min(1, virkningsgrad))  # Sikrer at virkningsgraden er innenfor [0, 1]

# Legg til en kolonne for virkningsgrad i temperaturdataene for Tromsø
tvec1, Tuvec1 = f_temp_tromso('Temp.Tromso.csv')
data_temp_tromso = pd.DataFrame({'time': tvec1, 'Tuvec1': Tuvec1})
data_temp_tromso['Virkningsgrad'] = data_temp_tromso['Tuvec1'].apply(beregn_virkningsgrad)

# Legg til en kolonne for virkningsgrad i temperaturdataene for Kristiansand
tvec2, Tuvec2 = f_temp_kristiansand('Temp.Kristiansand.csv')
data_temp_kristiansand = pd.DataFrame({'time': tvec2, 'Tuvec2': Tuvec2})
data_temp_kristiansand['Virkningsgrad'] = data_temp_kristiansand['Tuvec2'].apply(beregn_virkningsgrad)

# Plott virkningsgraden over tid for Tromsø
fig, ax = plt.subplots()
ax.plot(data_temp_tromso['time'], data_temp_tromso['Virkningsgrad'], label='Tromsø', color='blue')
ax.set_title('Virkningsgrad over tid (Tromsø)')
plt.xlabel('Dato')
plt.ylabel('Virkningsgrad')
plt.legend()
plt.show()

# Plott virkningsgraden over tid for Kristiansand
fig, ax = plt.subplots()
ax.plot(data_temp_kristiansand['time'], data_temp_kristiansand['Virkningsgrad'], label='Kristiansand', color='green')
ax.set_title('Virkningsgrad over tid (Kristiansand)')
plt.xlabel('Dato')
plt.ylabel('Virkningsgrad')
plt.legend()
plt.show()





"-----------------------------Strømproduksjon fra solcellepanelet---------------------------"

areal_solcelle = 10
# Beregn solinnstråling på solcellepanelet for Tromsø
data_temp_tromso['EffektivSolinnstraling'] = data_temp_tromso['Virkningsgrad'] * data_tromso['G(i)']

# Beregn strømproduksjonen for Tromsø
data_temp_tromso['Stromproduksjon'] = data_temp_tromso['EffektivSolinnstraling'] * areal_solcelle

# Lagre strømproduksjonen for Tromsø i en vektor
stromproduksjon_tromso = data_temp_tromso['Stromproduksjon'].to_numpy()

#Kristiansand
data_temp_kristiansand['EffektivSolinnstraling'] = data_temp_kristiansand['Virkningsgrad'] * data_kristiansand['G(i)']

data_temp_kristiansand['Stromproduksjon'] = data_temp_kristiansand['EffektivSolinnstraling'] * areal_solcelle
stromproduksjon_kristiansand = data_temp_kristiansand['Stromproduksjon'].to_numpy()

#Sjekk og håndter NaN-verdier for Tromsø
stromproduksjon_tromso[np.isnan(stromproduksjon_tromso)] = 0

# Sjekk og håndter NaN-verdier for Kristiansand
stromproduksjon_kristiansand[np.isnan(stromproduksjon_kristiansand)] = 0


'tester lengden------------'

print("Length of tvec1:", len(tvec1))
print("Length of tvec2:", len(tvec2))
print("Length of G(i) for Tromsø:", len(data_tromso['G(i)']))
print("Length of G(i) for Kristiansand:", len(data_kristiansand['G(i)']))
print("Length of stromproduksjon_tromso:", len(stromproduksjon_tromso))
print("Length of stromproduksjon_kristiansand:", len(stromproduksjon_kristiansand))


print("Indeks for data_kristiansand:", data_kristiansand.index)
print("Indeks for stromproduksjon_kristiansand:", pd.RangeIndex(start=0, stop=len(stromproduksjon_kristiansand)))




# Beregn den totale strømproduksjonen for Tromsø
total_stromproduksjon_tromso = round(stromproduksjon_tromso.sum() /1000,2)
print(f'Total strømproduksjon for Tromsø: {total_stromproduksjon_tromso} Kwh')

# Beregn den totale strømproduksjonen for Kristiansand
total_stromproduksjon_kristiansand = round(stromproduksjon_kristiansand.sum() /1000,2)
print(f'Total strømproduksjon for Kristiansand: {total_stromproduksjon_kristiansand} Kwh')

# Plot strømproduksjonen for Tromsø
fig, ax = plt.subplots()
ax.plot(data_temp_tromso['time'], stromproduksjon_tromso, label='Tromsø', color='blue')
ax.set_title('Strømproduksjon over tid (Tromsø)')
plt.xlabel('Dato')
plt.ylabel('Strømproduksjon (Watt)')
plt.legend()
plt.show()

# Plot strømproduksjonen for Kristiansand
fig, ax = plt.subplots()
ax.plot(data_temp_kristiansand['time'], stromproduksjon_kristiansand, label='Kristiansand', color='green')
ax.set_title('Strømproduksjon over tid (Kristiansand)')
plt.xlabel('Dato')
plt.ylabel('Strømproduksjon (Watt)')
plt.legend()
plt.show()




"---------------------månedlig sum for strømproduksjon------------------"

data_temp_tromso['Month'] = data_temp_tromso['time'].dt.month
data_temp_kristiansand['Month'] = data_temp_kristiansand['time'].dt.month

# Månedlig sum for strømproduksjon
monthly_sum_production_tromso = round(data_temp_tromso.groupby('Month')['Stromproduksjon'].sum(),2)
monthly_sum_production_kristiansand = round(data_temp_kristiansand.groupby('Month')['Stromproduksjon'].sum(),2)

E_tromso = monthly_sum_production_tromso.to_numpy()
E_kristiansand = monthly_sum_production_kristiansand.to_numpy()

# Beregn gjennomsnittet pr måned for Tromsø
average_production_tromso = round(E_tromso.mean(),2)

# Beregn gjennomsnittet pr måned for Kristiansand
average_production_kristiansand = round(E_kristiansand.mean(),2)

# Skriv ut resultatene
print(f"Gjennomsnittlig strømproduksjon pr måned for Tromsø: {average_production_tromso} KWh")
print(f"Gjennomsnittlig strømproduksjon pr måned for Kristiansand: {average_production_kristiansand} KWh")


"plott av måndeltlig gjennomsnitt"
# Månedsnavn for x-aksen
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Data for stolpediagrammet
monthly_production_tromso = E_tromso / 1000  # Konverterer fra Wh til KWh
monthly_production_kristiansand = E_kristiansand / 1000  # Konverterer fra Wh til KWh

# Plot stolpediagrammet
fig, ax = plt.subplots()
bar_width = 0.35
bar1 = ax.bar(np.arange(len(months)), monthly_production_tromso, bar_width, label='Tromsø', color='blue')
bar2 = ax.bar(np.arange(len(months)) + bar_width, monthly_production_kristiansand, bar_width, label='Kristiansand', color='green')

# Tilpasse aksene og legende
ax.set_xlabel('Måned')
ax.set_ylabel('Strømproduksjon (KWh)')
ax.set_title('Månedlig strømproduksjon')
ax.set_xticks(np.arange(len(months)) + bar_width / 2)
ax.set_xticklabels(months)
ax.legend()

# Vis stolpediagrammet
plt.show()


'----tabeller------'


data_temp_tromso['Month'] = data_temp_tromso['time'].dt.month
data_temp_kristiansand['Month'] = data_temp_kristiansand['time'].dt.month

# Månedlig sum for strømproduksjon
monthly_sum_production_tromso = round(data_temp_tromso.groupby('Month')['Stromproduksjon'].sum(), 2)
monthly_sum_production_kristiansand = round(data_temp_kristiansand.groupby('Month')['Stromproduksjon'].sum(), 2)

E_tromso = monthly_sum_production_tromso.to_numpy()
E_kristiansand = monthly_sum_production_kristiansand.to_numpy()

# ... (din eksisterende kode)

# Skriv ut resultatene
print(f"Gjennomsnittlig strømproduksjon pr måned for Tromsø: {average_production_tromso} KWh")
print(f"Gjennomsnittlig strømproduksjon pr måned for Kristiansand: {average_production_kristiansand} KWh")

# Opprett en DataFrame for månedlig strømproduksjon
monthly_production_df = pd.DataFrame({
    'Month': months,
    'Tromsø': monthly_production_tromso,
    'Kristiansand': monthly_production_kristiansand
})

# Skriv ut tabellen
print("\nMånedlig strømproduksjon:")
print(monthly_production_df)


# Månedlig sum for solinnstråling
monthly_sum_solar_tromso = round(data_tromso.groupby(data_tromso['time'].dt.month)['G(i)'].sum(), 2)
monthly_sum_solar_kristiansand = round(data_kristiansand.groupby(data_kristiansand['time'].dt.month)['G(i)'].sum(), 2)

# Opprett en DataFrame for månedlig solinnstråling
monthly_solar_df = pd.DataFrame({
    'Month': monthly_sum_solar_tromso.index,
    'Solinnstråling Tromsø': monthly_sum_solar_tromso.values,
    'Solinnstråling Kristiansand': monthly_sum_solar_kristiansand.values
})

# Månedlig sum for strømproduksjon
monthly_sum_production_tromso = round(data_temp_tromso.groupby('Month')['Stromproduksjon'].sum(), 2)
monthly_sum_production_kristiansand = round(data_temp_kristiansand.groupby('Month')['Stromproduksjon'].sum(), 2)

E_tromso = monthly_sum_production_tromso.to_numpy()
E_kristiansand = monthly_sum_production_kristiansand.to_numpy()

# Beregn gjennomsnittet pr måned for Tromsø
average_production_tromso = round(E_tromso.mean(), 2)

# Beregn gjennomsnittet pr måned for Kristiansand
average_production_kristiansand = round(E_kristiansand.mean(), 2)

# Opprett en DataFrame for månedlig strømproduksjon
monthly_production_df = pd.DataFrame({
    'Month': monthly_sum_production_tromso.index,
    'Stromproduksjon Tromsø': monthly_sum_production_tromso.values,
    'Stromproduksjon Kristiansand': monthly_sum_production_kristiansand.values
})

# Skriv ut tabellene
print("\nMånedlig Solinnstråling:")
print(monthly_solar_df)

print("\nMånedlig Strømproduksjon:")
print(monthly_production_df)


'--------lagre data i npy-----'
np.save('stromproduksjon_tromso.npy', stromproduksjon_tromso)
np.save('stromproduksjon_kristiansand.npy', stromproduksjon_kristiansand)


'--------littt ryddigere plott---------'

# Konverter tidspunktet til måneder
data_tromso['Month'] = data_tromso['time'].dt.month
data_kristiansand['Month'] = data_kristiansand['time'].dt.month

# Månedlig gjennomsnittlig stråling
monthly_avg_radiation_tromso = data_tromso.groupby('Month')['G(i)'].mean()
monthly_avg_radiation_kristiansand = data_kristiansand.groupby('Month')['G(i)'].mean()

# Månedsnavn for x-aksen
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Plot stolpediagrammet
fig, ax = plt.subplots()
bar_width = 0.35
bar1 = ax.bar(np.arange(len(months)), monthly_avg_radiation_tromso, bar_width, label='Tromsø', color='blue')
bar2 = ax.bar(np.arange(len(months)) + bar_width, monthly_avg_radiation_kristiansand, bar_width, label='Kristiansand', color='green')

# Tilpasse aksene og legende
ax.set_xlabel('Måned')
ax.set_ylabel('Gjennomsnittlig Solinnstråling (W/m^2)')
ax.set_title('Gjennomsnittlig Solinnstråling per Måned')
ax.set_xticks(np.arange(len(months)) + bar_width / 2)
ax.set_xticklabels(months)
ax.legend()

# Vis stolpediagrammet
plt.show()

import pandas as pd

# Gjennomsnittlig virkningsgrad pr måned for Tromsø
avg_virkningsgrad_tromso = data_temp_tromso.groupby('Month')['Virkningsgrad'].mean()

# Gjennomsnittlig virkningsgrad pr måned for Kristiansand
avg_virkningsgrad_kristiansand = data_temp_kristiansand.groupby('Month')['Virkningsgrad'].mean()

# Opprett en DataFrame for gjennomsnittlig virkningsgrad
avg_virkningsgrad_data = pd.DataFrame({
    'Måned': avg_virkningsgrad_tromso.index,
    'Gjennomsnittlig Virkningsgrad Tromsø': avg_virkningsgrad_tromso.values,
    'Gjennomsnittlig Virkningsgrad Kristiansand': avg_virkningsgrad_kristiansand.values
})

# Skriv ut tabellen
print("Gjennomsnittlig Virkningsgrad pr måned:")
print(avg_virkningsgrad_data)



# Månedlig sum for solinnstråling
monthly_sum_solar_tromso = round(data_tromso.groupby(data_tromso['time'].dt.month)['G(i)'].sum(), 2)
monthly_sum_solar_kristiansand = round(data_kristiansand.groupby(data_kristiansand['time'].dt.month)['G(i)'].sum(), 2)

# Opprett en DataFrame for månedlig solinnstråling
monthly_solar_df = pd.DataFrame({
    'Month': monthly_sum_solar_tromso.index,
    'Solinnstråling Tromsø': monthly_sum_solar_tromso.values,
    'Solinnstråling Kristiansand': monthly_sum_solar_kristiansand.values
})

# Månedlig sum for strømproduksjon
monthly_sum_production_tromso = round(data_temp_tromso.groupby('Month')['Stromproduksjon'].sum(), 2)
monthly_sum_production_kristiansand = round(data_temp_kristiansand.groupby('Month')['Stromproduksjon'].sum(), 2)

E_tromso = monthly_sum_production_tromso.to_numpy()
E_kristiansand = monthly_sum_production_kristiansand.to_numpy()

# Beregn gjennomsnittet pr måned for Tromsø
average_production_tromso = round(E_tromso.mean(), 2)

# Beregn gjennomsnittet pr måned for Kristiansand
average_production_kristiansand = round(E_kristiansand.mean(), 2)

# Opprett en DataFrame for månedlig strømproduksjon
monthly_production_df = pd.DataFrame({
    'Month': monthly_sum_production_tromso.index,
    'Stromproduksjon Tromsø': monthly_sum_production_tromso.values,
    'Stromproduksjon Kristiansand': monthly_sum_production_kristiansand.values
})

# Skriv ut tabellene
print("\nMånedlig Solinnstråling:")
print(monthly_solar_df)

print("\nMånedlig Strømproduksjon:")
print(monthly_production_df)


