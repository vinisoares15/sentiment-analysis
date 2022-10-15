"""

SUMÁRIO:
Valor 1-Comparação de Ativos no gráfico em Volume, Preço de fechamento, etc.
Valor 2-Calculando o beta,CAPM,Sharp
Valor 3-Calculando taxa de retorno logarítimico
Valor 4-Calculando taxa de retorno Simples
Valor 5-Calculando o risco de um ativo, sua correlação e covariância
Valor 5-Calculando o Coeficiente de Fronteira
Valor 6-Previsão do lucro bruto da empresa
Valor 7-Monte Carlo: Previsão dos Preços das Ações
Valor 8-Monte Carlo - Black Scholes Merton
Valor 9-Discretização de Euler

"""


"""
-------------------------------------------------------------------------------
Valor 1 - Comparação de Ativos no gráfico em Volume, Preço de fechamento, etc.
-------------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt

tickers = ['PETR4.SA', 'AZUL4.SA', '^BVSP']
mydata = pd.DataFrame()
for t in tickers:
  mydata[t] = wb.DataReader(
      t, data_source='yahoo', start='2018-1-1')['Volume']

#informações gerais
mydata.info()
mydata.head()
mydata.tail()
mydata.iloc[0]

#Normatização para base 100 e plotar o gráfico.
(mydata / mydata.iloc[0] * 100).plot(figsize = (15,6));
plt.show()

#CALCULATING THE RETURN OF A PORTFOLIO OF SECURITIES
returns = (mydata.shift(1))-1
returns.head()
#esses valores  são os pesos dos ativos em percentual
weights = np.array([0.15,0.25,0.25,0.25])
np.dot(returns,weights)

annual_returns = returns.mean() * 250
annual_returns

np.dot(annual_returns, weights)
#Valor em percentual do retorno anual do portfolio
pfolio_1 = str(round(np.dot(annual_returns,weights), 5) * 100) + '%'
print(pfolio_1)







"""
-------------------------------------------------------------------------------
Valor 2-Calculando o beta,CAPM,Sharp
-------------------------------------------------------------------------------
"""
import numpy as np
import  pandas as pd
from pandas_datareader import data as wb

tickers = ['PETR4.SA', '^BVSP']
data = pd.DataFrame()
for  t in tickers:
  data[t] = wb.DataReader(t, data_source = 'yahoo', start ='2012-1-1',
                          end = '2016-12-31') ['Adj Close']

#Normalmente o beta é medido num intervalo de 5 anos
sec_returns = np.log( data / data.shift(1))

cov = sec_returns.cov() * 250
cov

cov_with_market = cov.iloc[0, 1]
cov_with_market

market_var = sec_returns['^BVSP'].var() * 250
market_var

#Calculando o Beta do ativo

#Como o valor de Beta para a petrobrás é maior que 1, ela se  torna um ativo
#agressivo,ou seja, em tempos  de crise, ela vai ser bastante volátil.
#Um beta  menor do que 1 torna o ativo mais composto em tempos de crise.
PETR4_beta = cov_with_market / market_var
PETR4_beta

#CALCULATE THE EXPECTED RETURN FROM ATIVO (CAPM)

#O 0.0287 é o Yield da empresa em percentual (Procurar valores)
#O 0.05 é o coeficiente de risco subtraido pelo intervalo de confiança. 
#(aqueles 5% do fim da curva tlgd?)
PETR4_er = 0.0287 + PETR4_beta * 0.05
PETR4_er
#Basicamente esse resultado mostra o percentual que um investidor deseja
#de retorno ao investir no ativo.

#CALCULANDO O ÍNDICE DE SHAPE

Sharpe = (PETR4_er - 0.0287) / (sec_returns['PETR4.SA'].std() * 250 ** 0.5)
Sharpe






"""
-------------------------------------------------------------------------------
Valor 3-calculando taxa de retorno logarítimico
-------------------------------------------------------------------------------
"""

#Importando as Bibliotecas necessárias
import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt

#Essas linhas de código abaixo buscam os dados no yahoo finance sobre os
# respectivos ativos

PETR4 = wb.DataReader('PETR4.SA', data_source = 'yahoo', start ='2016-4-14')
IBOV = wb.DataReader('^BVSP', data_source = 'yahoo', start ='2016-4-14')

#Calculando a Taxa Simples de Retorno Logarítmica de cada ativo pelo Adj close
PETR4['Log_return'] = np.log(PETR4['Adj Close'] / PETR4['Adj Close'].shift(1))

print (PETR4['Log_return'])

IBOV['Log_return'] = np.log(IBOV['Adj Close'] / IBOV['Adj Close'].shift(1))

print (IBOV['Log_return'])

#Plotando os gráficos para comparação
IBOV['Log_return'].plot(figsize=(8,5))
plt.show()
PETR4['Log_return'].plot(figsize=(8,5))
plt.show()

#Taxa de retorno anual logarítmica de cada ativo.
Log_return_PETR4 = PETR4['Log_return'].mean() * 250
print( 'A taxa de retorno logarítmica de PETR4 é: ' + 
      str(round(Log_return_PETR4, 5) * 100) + '%')
Log_return_IBOV = IBOV['Log_return'].mean() * 250
print( 'A taxa de retorno logarítmica de IBOV é: ' + 
      str(round(Log_return_PETR4, 5) * 100) + '%')







"""
-------------------------------------------------------------------------------
Valor 4-calculando taxa de retorno Simples
-------------------------------------------------------------------------------
"""
#Importando as bibliotecas necessárias
import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt

#Essas linhas de código abaixo buscam os dados no yahoo finance sobre os
# respectivos ativos

PETR4 = wb.DataReader('PETR4.SA', data_source = 'yahoo', start ='2016-4-14')
IBOV = wb.DataReader('^BVSP', data_source = 'yahoo', start ='2016-4-14')

#Calculando a Taxa Simples de Retorno de cada ativo pelo Adj Close
PETR4['Simple_return'] = (PETR4['Adj Close'] / PETR4['Adj Close'].shift(1)) - 1

print (PETR4['Simple_return'])

IBOV['Simple_return'] = (IBOV['Adj Close'] / IBOV['Adj Close'].shift(1)) - 1

print (IBOV['Simple_return'])

#Plotando os gráficos para comparação
IBOV['Simple_return'].plot(figsize=(8,5))
plt.show()
PETR4['Simple_Return'].plot(figsize=(8,5))
plt.show()





"""
-------------------------------------------------------------------------------
Valor 5-calculando o risco de um ativo, sua correlação e covariância
-------------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt


tickers = ['PETR4.SA', 'AZUL4.SA']
sec_data = pd.DataFrame()
for t in tickers:
  sec_data[t] = wb.DataReader(t, data_source = 'yahoo',
                              start = '2010-1-1')['Adj Close']

sec_data.tail()
# retorno logarítimico
sec_returns = np.log(sec_data /sec_data.shift(1))
sec_returns

# VOLATILIDADE PARA  ATIVO 1
# Média de retorno anual em percentual
sec_returns['PETR4.SA'].mean() * 250
# Desvio padrão indica o percentual de volatilidade da empresa em 1 ano
sec_returns['PETR4.SA'].std() * 250 ** 0.5

# VOLATILIDADE PARA  ATIVO 2
# Média de retorno anual em percentual
sec_returns['AZUL4.SA'].mean() * 250
# Desvio padrão indica o percentual de volatilidade da empresa em 1 ano
sec_returns['AZUL4.SA'].std() * 250 ** 0.5


# COVARIÂNCIA E CORRELAÇÃO ENTRE O RETORNO DOS ATIVOS

# Calcula a variância usando .var()e multiplica por 250 para
# obter a variância anual
PG_var = sec_returns['PETR4.SA'].var() * 250
PG_var
BEI_var = sec_returns['AZUL4.SA'].var() * 250
BEI_var

# Calcula a Covariância usando .cov()
cov_matrix = sec_returns.cov()
cov_matrix

# Multiplicamos por 250 para obter a covariância anual
cov_matrix_a = sec_returns.cov() *250
cov_matrix_a

# Mostra uma matriz de correlção entre os retornos das empresas.
# É diferente da correlação dos preços das ações.
corr_matrix = sec_returns.corr()
corr_matrix

# CALCULANDO O RISCO OU A VOLATILIDADE DO PORTIFÓLIO  
"""
Risco sistemático: Ocorre quando há relações  entre as empresas 
Risco não sistemático: Ocorre por fenômenos externos (Guerras, pandemia, etc)
Criar uma carteira com pelo menos 25 à 30 ações que não estão correlacionadas
faz com que o risco sistemático desapareça.
"""
# Equal weigthing scheme 
# Nessa situação, os pesos dos ativos são iguais (50% e 50%)
weights = np.array([0.5,0.5])
# Variância do Portifólio anual
# Usamos a classe .T para  dizer  que estamos  utilizando a matris transposta
pfolio_var = (np.dot(weights.T, np.dot(sec_returns.cov() * 250, weights))
              ) ** 0.5
pfolio_var

# Volatilidade do portifólio 
pfolio_vol = (np.dot(weights.T, np.dot(sec_returns.cov() * 250, weights))
              ) ** 0.5
pfolio_vol

print(str(round(pfolio_vol, 5)* 100) + '%')

# CALCULATING DIVERSIFIABLE AND NON-DIVERSIFIABLE RISK OF PORTFOLIO

# Para obter o risco diversificável anual, precisamos da variância da carteira.
weights = np.array([0.5, 0.5])
weights[1]
weights[0]
# RISCO DIVERSIFICÁVEL:
# Para obter o risco diversificável anual,
# precisamos da variância da carteira. Em seguida subtrair a variância 
# ponderada anual de cada ação.
PETR4_var_a = sec_returns[['PETR4.SA']].var() * 250
PETR4_var_a
AZUL4_var_a = sec_returns[['AZUL4.SA']].var() * 250
AZUL4_var_a
dr = pfolio_var - (weights[0] ** 2 * PETR4_var_a) - (
    weights[1] **2 * AZUL4_var_a)
dr
float(PETR4_var_a)
PETR4_var_a = sec_returns['PETR4.SA'].var() * 250
PETR4_var_a
AZUL4_var_a = sec_returns['AZUL4.SA'].var() * 250
AZUL4_var_a

dr = pfolio_var - (
    weights[0] ** 2 * PETR4_var_a) - (weights[1] **2 * AZUL4_var_a)

print(str(round(dr*100, 3))) + '%'

# RISCO NÃO-SISTEMÁTICO:

n_dr_1 = pfolio_var - dr
n_dr_1

n_dr_2 = (weights[0] ** 2 * PETR4_var_a) + (weights [1] ** 2 * AZUL4_var_a)
n_dr_2

n_dr_1 == n_dr_2







"""
-------------------------------------------------------------------------------
Valor 5-calculando o Coeficiente de Fronteira
-------------------------------------------------------------------------------
"""

import  numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
# Facilita a plotagem salvando o conteúdo na biblioteca da IDE
# Esse código possuia %matplotlib inline inicializado no jupyter  o colab
# Provavelmente só funcionará no júpter. Logo considere retirar a # da linha
# seguinte:
#%matplotlib inline
# Obtendo os dados
assets = ['PETR4.SA', '^BVSP']
pf_data = pd.DataFrame()

for a in assets:
  pf_data[a] = wb.DataReader(a , data_source='yahoo', 
                             start = '2010-1-1')['Adj Close']

pf_data.tail()
# Mostrando os dados segundo o Adj Close
(pf_data / pf_data.iloc[0] * 100).plot(figsize=(10, 5))

# Mostra o retorno anual de cada ativo
log_returns = np.log(pf_data / pf_data.shift(1))

# Média dos retornos anuais
log_returns.mean() * 250

# Covariância dos  retornos anuais
log_returns.cov() * 250

# Correlação entre os retornos anuais
# Os retornos anuais de Petrobrás e a Ibov estão bem correlacionados.
# 77% de correlação segundo a matriz
log_returns.corr()

num_assets = len(assets)

# Temos dois ativos
num_assets

# Criando pesos randômicos  para os ativos de 0 a 1 (100%)
# Se for colocar pesos reais, desconsidere estas linhas
weights = np.random.random(num_assets)
weights /= np.sum(weights)
weights

weights[0] + weights[1]

# Expected Portifolio Return:
np.sum(weights * log_returns.mean()) * 250
# Expected Portfolio Variance:
np.dot(weights.T, np.dot(log_returns.cov() * 250, weights))
# Expected Portfolio Volatility:
np.sqrt(np.dot(weights.T, np.dot(log_returns.cov() * 250, weights)))

"""
Nessas linhas de código, vamos criar pesos aleatórios de 1 a 1000 para 
combinações diferentes dos dois ativos 
"""

pfolio_returns = []
pfolio_volatilities = []

for x in range (1000):
  weights = np.random.random(num_assets)
  weights /= np.sum(weights)
  pfolio_returns.append(np.sum(weights * log_returns.mean()) * 250)
  pfolio_volatilities.append(np.sqrt(np.dot(weights.T, np.dot(
      log_returns.cov() *250, weights))))

pfolio_returns = np.array(pfolio_returns)
pfolio_volatilities = np.array(pfolio_volatilities)

pfolio_returns, pfolio_volatilities

#Cria um objeto de um dataframe: um para o retorno e outra para volatilidade
portfolios = pd.DataFrame({'Return': pfolio_returns,
                           'Volatility':pfolio_volatilities})

portfolios.head()
portfolios.tail()

portfolios.plot(x='Volatility', y='Return',
                kind = 'scatter', figsize = (10, 6));
plt.xlabel('Expected Volatility')
plt.ylabel('Expecter Return')



"""
-------------------------------------------------------------------------------
Valor 6-Previsão do lucro bruto da empresa
-------------------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt

rev_m = 170         #receita média
rev_stdev = 20      #desvio padrão da receita
iterations = 1000   #número de interações

#Criando valores randômicos sobre a média dos valores acima
rev = np.random.normal(rev_m, rev_stdev, iterations)
rev

#Plotando o gráfico dos valores acima.
plt.figure(figsize=(15, 6))
plt.plot(rev)
plt.show()

#Plotando o lucro bruto (COGS)
COGS = - (rev * np.random.normal(0.6, 0.1))

plt.figure(figsize=(15, 6))
plt.plot(COGS)
plt.show()

COGS.mean()

COGS.std()

#Calculando o CRV da empresa
Gross_Profit = rev + COGS
Gross_Profit

plt.figure(figsize=(15, 6))
plt.plot(Gross_Profit)
plt.show()

max(Gross_Profit)

min(Gross_Profit)

Gross_Profit.mean()

Gross_Profit.std()

#Plotando num histograma especificando as classes
plt.figure(figsize=(10, 6));
plt.hist(Gross_Profit, bins = [40, 50, 60, 70, 80, 90, 100, 110, 120]);
plt.show()

#Plotando num histograma variando o número de classes (melhor visualização)
plt.figure(figsize=(10, 6));
plt.hist(Gross_Profit, bins = 20);
plt.show()



"""
-------------------------------------------------------------------------------
Valor 7-Monte Carlo: Previsão dos Preços das Ações
-------------------------------------------------------------------------------
"""

import  numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
from scipy.stats import norm
%matplotlib inline

#Dados do yahoo
ticker = 'PETR4.SA'
data = pd.DataFrame()
data[ticker] = wb.DataReader(
    ticker, data_source = 'yahoo', start = '2013-1-1')['Adj Close']

#Retorno logarítimico
log_returns = np.log(1+ data.pct_change())

log_returns.tail()

#Retorno ao longo da série temporal
data.plot(figsize=(10, 6));

#plot dos retornos logarítimicos (Não DOS PREÇOS)
log_returns.plot(figsize=(10, 6))

#Calculando o movimento Browniano (Média)
u = log_returns.mean()
u

#Calculando o movimento Browniano (Variância)
var = log_returns.var()
var

#É a melhor aproximação das taxas de retorno futuras de uma ação
drift = u - (0.5 * var)
drift

#Desvio Padrão
stdev = log_returns.std()
stdev

#Define o tipo 
type(drift)

type(stdev)

np.array([drift])

# Ao digitar .values depois de um objeto pandas, faz com que o objeto seja 
# trasnferido para um array numpy
drift.values

stdev.values

#Z Corresponde à distância entre a média e os eventos, expresso pelo número
# de desvios padrão. Essa função à baixo define Z
norm.ppf(0.95) 

x = np.random.rand(10, 2)
x

norm.ppf(x)

Z = norm.ppf(np.random.rand(10, 2))
Z

t_intervals = 1000 #Prever os preços das ações para os próximos 1000 dias
iterations = 10 #10 séries de 1000

#Matrix 10 por 1000 com os retornos diários esperados
daily_returns = np.exp(
    drift.values + stdev.values * norm.ppf(
        np.random.rand(t_intervals, iterations)))
daily_returns

#Preço da Ação hoje. (Na lista)
S0 = data.iloc[-1]
S0

#Cria uma matriz com as mesmas dimensões de outra matriz.
# Nesse caso, criou uma
# matriz com as dimensões  10 por 1000 com todos os valores iguais a 0
price_list = np.zeros_like(daily_returns)

price_list

price_list[0] = S0
price_list

#função para  retorno da ação esperado
for t in range(1, t_intervals):
  price_list[t] = price_list[t - 1] * daily_returns[t]

price_list

#10 possíveis caminhos de  preços esperados pela PETR4.SA
plt.figure(figsize=(10, 6))
plt.plot(price_list);



"""
-------------------------------------------------------------------------------
Valor 8-Monte Carlo - Black Scholes Merton
-------------------------------------------------------------------------------
"""
import numpy as np
import pandas as pd
from pandas_datareader import data as wb
from scipy.stats import norm

#Valores necessários para incluir na função de Black Scholes onde
# S - Preço da Ação  //  K - Preço de  Exercício  //
#  r - Texa Livre de Risco do País(Brasíl)
# stdev - desvio padrão  //  T - intervalo de tempo (anos)
def d1(S, K, r, stdev, T):
    return (np.log(S / K) + (r + stdev ** 2 / 2) * T) / (stdev * np.sqrt(T))

def d2(S, K, r, stdev, T):
    return (np.log(S / K) + (r - stdev ** 2 / 2) * T) / (stdev * np.sqrt(T))

norm.cdf(0)

norm.cdf(0.25)

norm.cdf(0.75)

norm.cdf(9)

#Aplicando Black Scholes (Essa  é a fórmula)
def BSM(S, K, r, stdev, T):
  return( S * norm.cdf(d1(S, K, r, stdev, T))) - (
      K * np.exp(-r * T) * norm.cdf(d2(S, K, r, stdev, T)))

#Dados do yahoo
ticker = 'PETR4.SA'
data = pd.DataFrame()
data[ticker] = wb.DataReader(
    ticker, data_source = 'yahoo', start = '2013-1-1')['Adj Close']

S = data.iloc[-1]
S

log_returns = np.log(1 + data.pct_change())
stdev = log_returns.std() * 250 ** 0.5
stdev

#Dados específicos 
r = 0.0287
K = 110
T = 1

d1(S, K, r, stdev, T)
d2(S, K, r, stdev, T)
BSM(S, K, r, stdev, T)



"""
-------------------------------------------------------------------------------
Valor 9-Discretização de Euler
-------------------------------------------------------------------------------
"""
import numpy as np
import pandas as pd
from pandas_datareader import data as wb
from scipy.stats import norm
import matplotlib.pyplot as plt
%matplotlib inline

#Dados do yahoo
ticker = 'PETR4.SA'
data = pd.DataFrame()
data[ticker] = wb.DataReader(
    ticker, data_source = 'yahoo', start = '2010-1-1',
     end = '2020-1-1')['Adj Close']

log_returns = np.log(1 + data.pct_change())

r = 0.025

stdev = log_returns.std() * 250 ** 0.5
stdev

type(stdev)

stdev= stdev.values
stdev

T = 1.0
t_intervals = 250
delta_t = T / t_intervals
iterations = 10000

Z = np.random.standard_normal((t_intervals + 1, iterations))
S = np.zeros_like(Z)
S0 = data.iloc[-1]
S[0] = S0

for t in range(1, t_intervals + 1):
  S[t] = S[t-1] * np.exp(
      (r - 0.5 * stdev ** 2) * delta_t + stdev * delta_t ** 0.5 * Z[t])

S
S.shape

plt.figure(figsize = (10, 6))
plt.plot(S[:, :10]);

#Calculando o payoff
p = np.maximum(S[-1] - 110, 0)

p

p.shape

C = np.exp(-r * T) * np.sum(p) / interations


















