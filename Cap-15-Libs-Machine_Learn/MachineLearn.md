
# 📚 Bibliotecas para *Machine Learning*  

O **Aprendizado de Máquina** (*Machine Learning*) está cada vez mais popular pelo seu poder e utilidade na sociedade.  
Este guia apresenta os principais conceitos e ferramentas para começar.  

---

## 🌐 Ecossistema de Dados  

![Ecossistema](./IMGs/EcossistemaDados.png)  

Um **ecossistema de dados** é um ambiente colaborativo onde pessoas, ferramentas e informações interagem para gerar valor. Ele é composto por:  

1. **📦 Recursos** → Dados e ferramentas para coletar, armazenar, processar e analisar.  
   *Ex.: banco de dados de clientes, nuvem para armazenamento.*  

2. **⚙️ Funções** → Atividades como produzir (coletar, limpar), consumir (analisar) e intermediar (transformar em produtos/soluções).  
   *Ex.: analista cria relatórios, gestor toma decisões com eles.*  

3. **👥 Atores** → Profissionais como engenheiros de dados, cientistas de dados e especialistas em segurança.  

4. **🔗 Relacionamentos** → Conexões entre pessoas, recursos e ferramentas.  
   *Ex.: banco de dados → API → painel de visualização.*  

---

## 📊 7 Áreas de Atuação para Analistas  

![AtividadeemAnalises](./IMGs/AtividadesEmAnalises.png)  

---

## 🛠️ Ferramentas para *Machine Learning*  

No *Machine Learning*, raramente usamos apenas **uma** biblioteca — o ideal é combinar várias.  

**Exemplo de fluxo de trabalho:**  
1. Definir o objetivo e identificar dados, formatos e fontes.  
2. Tratar e transformar os dados para garantir qualidade.  
3. Realizar análises para responder perguntas de negócio.  
4. 📈 Apresentar resultados com visualizações claras.  

| 🧩 Módulo      | 💡 Descrição                                                       | 🔍 Principais Funções                          |
|---------------|-------------------------------------------------------------------|------------------------------------------------|
| **NumPy**     | Cálculos numéricos eficientes (*Numerical Python*)                | `min`, `max`, `mean`, `median`, `mode`         |
| **Pandas**    | Manipulação de dados em DataFrames                                | `columns`, `rows`, `describe`, `head`          |
| **Sympy**     | Matemática simbólica (limites, derivadas, integrais)              | `limit`, `diff`, `integrate`                   |
| **SciPy**     | Cálculo, probabilidade e testes de hipóteses                      | `binom`, `poisson`, `expon`, `norm`, `t`, `chi2`, `f` |
| **statsmodels** | Modelos estatísticos                                             | `OLS`                                          |
| **scikit-learn** | Análise preditiva e aprendizado supervisionado/não supervisionado | `LinearRegression`, `PCA`, `KMeans`            |

![Ferramentas](./IMGs/Tools.png)  


---
## Exemplo Estatística Descritiva

```Python
df_dados_paises.groupby('pais') \
    .agg(min_idh = pd.NamedAgg('idh', 'min'),
         max_idh = pd.NamedAgg('idh', 'max'),
         media_idh = pd.NamedAgg('idh', 'mean'),
        mediana_idh = pd.NamedAgg('idh', 'median'),
        desvio_padrao_idh = pd.NamedAgg('idh', 'std'))
```

## Probabilidades

```python 
# modulos
from scipy.stats import binom, poisson, norm 
```

- 1️⃣ Distribuição Binomial
Quando usar:
A Distribuição Binomial é usada quando temos um número fixo de tentativas n, cada uma com apenas dois resultados possíveis: sucesso ou fracasso.
Exemplos: número de caras ao lançar uma moeda, número de produtos defeituosos em um lote.

Parâmetros:

n → Número de tentativas.

p → Probabilidade de sucesso em cada tentativa.

```Python 
# Binomial 
n = 5 
p = 0.8
binom.pmf(k=1, n=n, p=p) 
```
- 2️⃣ Distribuição de Poisson
Quando usar:
A Distribuição de Poisson é usada para modelar o número de eventos que ocorrem em um intervalo fixo de tempo ou espaço, quando esses eventos ocorrem de forma independente e com uma taxa média constante.
Exemplos: número de e-mails recebidos por hora, número de acidentes em uma rodovia por mês.

Parâmetro:

mu → Média esperada de eventos no intervalo.

```Python
mu = 2
poisson.pmf(k=1, mu=mu)
```
 
- ️3️⃣ Distribuição Normal
Quando usar:
A Distribuição Normal (ou Gaussiana) é contínua e muito comum em fenômenos naturais e sociais, como altura de pessoas, notas de provas, erros de medição.
Ela é simétrica em torno da média.

Parâmetros:

loc (média) → Valor central da distribuição.

scale (desvio padrão) → Medida de dispersão dos dados.

```python
media = 30 
desvio_padrao = 5
norm.pdf(x=25, loc=media, scale = desvio_padrao)
```

| Distribuição | Tipo     | Uso principal                                         | Função usada |
|--------------|----------|-------------------------------------------------------|--------------|
| Binomial     | Discreta | Contar sucessos em um número fixo de tentativas        | pmf          |
| Poisson      | Discreta | Contar eventos em um intervalo fixo                    | pmf          |
| Normal       | Contínua | Modelar dados que seguem curva simétrica em torno da média | pdf          |


## 📊 Amostragem

A amostragem é o processo de selecionar um subconjunto de dados (amostra) a partir de um conjunto maior (população).
Ela é usada quando não é viável ou necessário analisar todos os dados disponíveis.
O objetivo é que a amostra represente bem a população, permitindo conclusões confiáveis com menos esforço.

Exemplos de uso da amostragem:

Pesquisas de opinião (entrevistar 1.000 pessoas em vez de toda a população).

Testes de qualidade (verificar alguns produtos de um lote, não todos).

Treinamento de modelos de Machine Learning (usar apenas parte dos dados para treino).


```python
# Verificar a forma (linhas, colunas) do DataFrame original
df_dados_paises.shape

# Criar uma amostra de 50 linhas aleatórias do DataFrame
df_amostra = df_dados_paises.sample(n=50)

# Verificar a forma da amostra
df_amostra.shape

```
💡 Dica: Se quiser garantir que a amostra seja sempre a mesma (reprodutível), adicione o parâmetro random_state

```python

df_amostra = df_dados_paises.sample(n=50, random_state=42)

```

## Teste de Hipótese

O teste de hipóteses é um método estatístico usado para verificar se uma suposição sobre um conjunto de dados é provavelmente verdadeira ou falsa com base em evidências matemáticas.

No processo, formulamos duas hipóteses:

- H₀ (hipótese nula) → Afirmação inicial que será testada (ex.: “o IDH do Brasil é igual a 0,5”).

- H₁ (hipótese alternativa) → Contraposição à hipótese nula (ex.: “o IDH do Brasil é diferente de 0,5”).

🔍 Exemplo: Testando se o IDH médio do Brasil é igual a 0,5 usando um teste t de uma amostra (ttest_1samp).

```python
from scipy import stats as st

dados_brasil = df_dados_paises[df_dados_paises['pais'] == 'Brasil']

st.ttest_1samp(a=dados_brasil['idh'], 
               popmean=0.5)
```

## 🔗 Correlação

A correlação mede a intensidade e a direção da relação entre duas variáveis.
Em outras palavras, indica o quanto uma variável varia em relação à outra.

📌 Exemplo: Quanto maior a temperatura, maior a venda de sorvete — isso indica correlação positiva.
Já se o aumento da temperatura reduzir a venda de casacos, temos correlação negativa.

🔍 Exemplo prático: Calculando a correlação entre alguns indicadores.

```python
cols = ['idh', 'corrupcao_indice', 'competitividade_indice', 'globalizacao_indice']
df_dados_paises[cols].corr()
```

📊 Interpretação dos valores:

- +1 → Correlação positiva perfeita.

- 0 → Sem correlação.

- -1 → Correlação negativa perfeita.

## 📈 Regressão
A regressão é uma técnica estatística usada para entender como uma ou mais variáveis independentes (fatores explicativos) influenciam uma variável dependente (resultado que queremos prever ou explicar).

Ela é muito usada para:

- 📊 Fazer previsões.

- 🔍 Identificar relações entre variáveis.

- ⚖️ Quantificar o impacto de cada variável no resultado.

📌 Exemplo: Vamos estimar como corrupcao_indice, competitividade_indice e globalizacao_indice influenciam o idh.

```python
import statsmodels.formula.api as sm

modelo_regressao = sm.ols(formula='idh ~ corrupcao_indice + competitividade_indice + globalizacao_indice',
                         data=df_dados_paises).fit()

modelo_regressao.summary()               

```

## 🟢 KMeans – Análise de Clusters
O KMeans é uma técnica de clusterização usada para agrupar dados em conjuntos (clusters) de acordo com suas características, de forma que dados dentro de um mesmo grupo sejam mais semelhantes entre si do que com dados de outros grupos.

📌 Exemplo: Agrupar países com base em corrupcao_indice, competitividade_indice e globalizacao_indice.

```python
from sklearn.cluster import kmeans

dados_resumo = df_dados_paises.groupby('pais') \
    .agg(media_corrupcao = pd.NamedAgg('corrupcao_indice', 'mean'),
            media_competitividade = pd.NamedAgg('competitividade_indice', 'mean'),
            media_globalizacao = pd.NamedAgg('globalizacao_indice', 'mean')) \
    .reset_index()

kmeans = KMeans(init="random", 
                n_clusters=3, 
                n_init=10, 
                max_iter=300, 
                random_state=42)

kmeans.fit(dados_resumo[['media_corrupcao', 'media_competitividade', 'media_globalizacao']])

dados_resumo['cluster'] = kmeans.labels_

dados_resumo.head()

```

📊 Interpretação:

- Cada país recebe um rótulo de cluster (0, 1 ou 2).

- Países no mesmo cluster possuem perfis semelhantes nos índices analisados.

- Isso ajuda a identificar padrões e segmentar grupos com características próximas.


## 📦 Box-Plot – Detectando Outliers
O Box-Plot é um gráfico que ajuda a identificar valores discrepantes (outliers) em uma variável.

```python
# estatística descritiva
dados_resumo = df_dados_paises.groupby('pais') \
    .agg(media_pib = pd.NamedAgg('pib', 'mean')) \
    .reset_index()

# boxplot 
dados_resumo[['media_pib']].boxplot()

```


📊 Interpretação:

- A linha central representa a mediana.

- O retângulo (box) mostra o intervalo entre o 1º e 3º quartil.

- Os outliers são pontos que ficam fora dos "bigodes" do boxplot, indicando valores muito discrepantes da média.





