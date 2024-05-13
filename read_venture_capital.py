import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import plotting

# import cpi
# cpi.update()

df = pd.read_stata('C:/Users/Jakob/Downloads/venture_data_2005_2015.dta')
df.dropna(subset=['Investment_Date'], inplace=True)
df.Company_Name.nunique()

pd.to_numeric(df['Equity_Amount_Estimated__USD_Mil'], errors='coerce').groupby(df.Company_Name).sum().sort_values(ascending=False)

# path = 'C://Users/Jakob/Downloads/PreqinVentureDeals_1970_July2018/'
# files = os.listdir(path)
#
# chunks = []
# for i, file in tqdm(enumerate(files), total=len(files)):
#     chunk = pd.read_excel(os.path.join(path, file), skiprows=13)
#     chunks.append(chunk)
#
# preqin = pd.concat(chunks)
# preqin.dropna(inplace=True)
#
# preqin.to_csv('C:/Users/Jakob/Downloads/PreqinVentureDeals_1970_July2018.csv', index=False)

## plot deals per year and equity invested per year in the same plot with two y axes
deals_per_year = df['Equity_Amount_Estimated__USD_Mil'].groupby(df['Investment_Date'].dt.year).size()
equity_inv_per_year = pd.to_numeric(df['Equity_Amount_Estimated__USD_Mil'], errors='coerce').groupby(df['Investment_Date'].dt.year).sum()

# adjust for inflation
equity_inv_per_year_infl = equity_inv_per_year.reset_index().apply(lambda x: cpi.inflate(x['Equity_Amount_Estimated__USD_Mil'], int(x['Investment_Date']), to=2015), axis=1)
equity_inv_per_year_infl_billions = equity_inv_per_year_infl / 1000


# plot
fig, ax1 = plt.subplots(figsize=(8, 5))

color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
ax1.set_xlabel('Year')
ax1.set_ylabel('Total number of deals', color=color)
ax1.plot(deals_per_year.index, deals_per_year, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]
ax2.set_ylabel('Total equity invested (Billion 2015 USD)', color=color)  # we already handled the x-label with ax1
ax2.plot(deals_per_year.index, equity_inv_per_year_infl_billions, color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.xaxis.set_major_locator(ticker.MultipleLocator(5))

plt.title('Venture capital deals and equity invested in the US from 2005 to 2015 (Venture Expert Data)')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


# same for preqin data
preqin = pd.read_csv('C:/Users/Jakob/Downloads/PreqinVentureDeals_1970_July2018.csv')

deals_per_year = preqin['Deal Date'].groupby(pd.to_datetime(preqin['Deal Date']).dt.year).size()
equity_inv_per_year = pd.to_numeric(preqin['DealSize (USD mn)'], errors='coerce').groupby(pd.to_datetime(preqin['Deal Date']).dt.year).sum()
equity_inv_per_year_infl = equity_inv_per_year.reset_index().apply(lambda x: cpi.inflate(x['DealSize (USD mn)'], int(x['Deal Date']), to=int(equity_inv_per_year.index.max())), axis=1)
equity_inv_per_year_infl_billions = equity_inv_per_year_infl / 1000

fig, ax1 = plt.subplots(figsize=(8, 5))

color = 'tab:red'
ax1.set_xlabel('Year')
ax1.set_ylabel('Total number of deals', color=color)
ax1.plot(deals_per_year.index, deals_per_year, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Total equity invested (Billion 2018 USD)', color=color)  # we already handled the x-label with ax1
ax2.plot(deals_per_year.index, equity_inv_per_year_infl_billions, color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.xaxis.set_major_locator(ticker.MultipleLocator(5))

plt.title('Venture capital deals and equity invested in the US from 2005 to 2015 (Preqin Data)')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

preqin['Deal Date'] = pd.to_datetime(preqin['Deal Date'], errors='coerce')
preqin.dropna(subset=['Deal Date'], inplace=True)
preqin['Deal Year'] = preqin['Deal Date'].dt.year.apply(int)
preqin['DealSize (2018 USD mn)'] = preqin.apply(lambda x: cpi.inflate(x['DealSize (USD mn)'], x['Deal Year'], to=2018), axis=1)

biggest_vc_recipients = preqin.groupby('Portfolio Company Name')['DealSize (2018 USD mn)'].sum().sort_values().tail(20)/1000

plot_barh(biggest_vc_recipients, '20 biggest venture capital recipients (Preqin Data)', 'Total equity invested (Billion 2018 USD)')

# same for venture expert data
df['Investment_Date'] = pd.to_datetime(df['Investment_Date'], errors='coerce')
df.dropna(subset=['Investment_Date'], inplace=True)
df['Investment_Year'] = df['Investment_Date'].dt.year.apply(int)
df['Equity_Amount_Estimated__USD_Mil'] = pd.to_numeric(df['Equity_Amount_Estimated__USD_Mil'], errors='coerce')
df.dropna(subset=['Equity_Amount_Estimated__USD_Mil'], inplace=True)
df['Equity_Amount_Estimated_2018_USD_Mil'] = df.apply(lambda x: cpi.inflate(x['Equity_Amount_Estimated__USD_Mil'], x['Investment_Year'], to=2018), axis=1)
biggest_vc_recipients = df.groupby('Company_Name')['Equity_Amount_Estimated_2018_USD_Mil'].sum().sort_values().tail(20)/1000
plot_barh(biggest_vc_recipients, '20 biggest venture capital recipients (Venture Expert Data)', 'Total equity invested (Billion 2018 USD)', extend_x_axis=50)


# venture expert
venture_expert = pd.read_excel('C:/Users/Jakob/Downloads/4 VentureExpertData_toFuzzyMatch1970_2016/VentureExpertData.xlsx')
venture_expert.dropna(how='all', inplace=True)

deals_per_year = venture_expert['VXCompanyName'].groupby(venture_expert['VX_DateFirstInvest'].dt.year).size()

# plot only deals per year
deals_per_year.plot(figsize=(8, 5))
plt.title('Venture capital deals in the US from 1970 to 2016 (Venture Expert Data)')
plt.xlabel('Year')
plt.ylabel('Total number of deals')
plt.show()

deals_per_year_venture_expert = venture_expert['VXCompanyName'].groupby(venture_expert['VX_DateFirstInvest'].dt.year).size()
deals_per_year_preqin = preqin['Deal Date'].groupby(preqin['Deal Date'].dt.year).size()

# plot deals per year for both datasets
deals_per_year_venture_expert.plot(figsize=(6, 4), label='Venture Expert')
deals_per_year_preqin.plot(figsize=(6, 4), label='Preqin')
plt.xlabel('Year', )
plt.ylabel('Total number of recorded deals')
plt.legend()
# let x-axis start at 1970
plt.xlim(left=1970)
plt.show()


