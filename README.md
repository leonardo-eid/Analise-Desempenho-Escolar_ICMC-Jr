# 📊 Análise de Desempenho Escolar — Projeto Trainee ICMC Jr

Análise exploratória e modelagem preditiva aplicados a dados de desempenho de 1.000 estudantes do ensino médio. Projeto desenvolvido no **Processo Trainee de Estatística 2026.1 da ICMC Júnior** (USP – São Carlos).

![Python](https://img.shields.io/badge/Python-3.x-blue)
![pandas](https://img.shields.io/badge/pandas-data%20wrangling-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-green)
![Jupyter](https://img.shields.io/badge/Jupyter-notebooks-green)

---

## 📌 Sumário

- [Contexto](#-contexto)
- [Objetivos](#-objetivos)
- [Base de dados](#-base-de-dados)
- [Estrutura do repositório](#-estrutura-do-repositório)
- [Metodologia e principais resultados](#-metodologia-e-principais-resultados)
  - [1. Limpeza e pré-processamento](#1-limpeza-e-pré-processamento)
  - [2. Análise exploratória e descritiva](#2-análise-exploratória-e-descritiva)
  - [3. Teste estatístico](#3-teste-estatístico)
  - [4. Modelagem preditiva](#4-modelagem-preditiva)
- [Conclusões](#-conclusões)
- [Como executar](#-como-executar)
- [Tecnologias](#-tecnologias)

---

## 🎯 Contexto

O projeto parte de um caso fictício proposto pela ICMC Jr: um cliente, pós-doutorando em pedagogia infantil, acompanhou durante anos **1.000 alunos do ensino médio de São Carlos**, medindo fatores como nota final, desempenho por disciplina, horas de sono, tempo de tela e atividade física. Agora ele quer uma análise profunda das relações entre essas variáveis.

O enunciado completo (contexto, descrição das colunas, recomendações de estudo e escopo) está em [`docs/Projeto_Trainee_Estatística.pdf`](docs/Projeto_Trainee_Estatística.pdf).

## 🎯 Objetivos

- Limpar e preparar a base de dados, identificando e tratando inconsistências.
- Explorar graficamente as relações entre hábitos, contexto e desempenho dos alunos.
- Verificar estatisticamente a relação entre **frequência escolar** e **nota final**.
- Investigar se é possível **prever a nota final** de um aluno a partir das demais variáveis.

## 🗂️ Base de dados

Arquivo: [`data/student_performance_interactions.csv`](data/student_performance_interactions.csv) — **1.000 alunos × 18 colunas**.

| Coluna | Descrição |
|---|---|
| `student_id` | Identificador do estudante |
| `final_score` | Pontuação final (0–100) |
| `grade` | Conceito categórico (A é o melhor) |
| `pass_fail` | Aprovação no ano (1 = passou, 0 = reprovou) |
| `previous_score` | Pontuação do ano anterior |
| `math_prev_score` | Pontuação anterior em matemática |
| `science_prev_score` | Pontuação anterior em ciências |
| `language_prev_score` | Pontuação anterior em línguas |
| `daily_study_hours` | Horas de estudo por dia |
| `attendance_percentage` | Percentual de presença nas aulas |
| `homework_completion_rate` | Percentual de tarefas concluídas |
| `sleep_hours` | Horas de sono por dia |
| `screen_time_hours` | Horas de tela por dia |
| `physical_activity_minutes` | Minutos de atividade física por dia |
| `motivation_score` | Nível de motivação escolar |
| `exam_anxiety_score` | Nível de ansiedade em provas |
| `parent_education_level` | Escolaridade dos pais (High School, Bachelor, Master) |
| `study_environment` | Nível de ruído do ambiente de estudo (Noisy, Moderate, Quiet) |

## 📁 Estrutura do repositório

```
Analise-Desempenho-Escolar_ICMC-Jr/
├── data/
│   └── student_performance_interactions.csv   # base de dados original
├── docs/
│   └── Projeto_Trainee_Estatística.pdf        # enunciado e escopo do projeto
├── limpezaDeDados.ipynb                       # 1. limpeza e pré-processamento
├── exploracaoDeDados.ipynb                    # 2. análise exploratória + teste estatístico
├── modelagemPreditiva.ipynb                   # 3. modelagem preditiva
└── README.md
```

Os notebooks foram pensados para serem lidos em ordem e carregam os dados direto do repositório no GitHub, então podem ser executados no Google Colab sem nenhuma configuração.

| Notebook | Conteúdo | Colab |
|---|---|---|
| `limpezaDeDados.ipynb` | Diagnóstico da base, correção de inconsistências, codificação ordinal e padronização | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leonardo-eid/Analise-Desempenho-Escolar_ICMC-Jr/blob/main/limpezaDeDados.ipynb) |
| `exploracaoDeDados.ipynb` | Box plot, heatmap, histograma, gráfico de dispersão e testes de Shapiro-Wilk e Spearman | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leonardo-eid/Analise-Desempenho-Escolar_ICMC-Jr/blob/main/exploracaoDeDados.ipynb) |
| `modelagemPreditiva.ipynb` | Regressão linear múltipla vs. árvore de decisão para prever `final_score` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leonardo-eid/Analise-Desempenho-Escolar_ICMC-Jr/blob/main/modelagemPreditiva.ipynb) |

---

## 🔬 Metodologia e principais resultados

### 1. Limpeza e pré-processamento

📓 `limpezaDeDados.ipynb`

**Diagnóstico da base**
- Sem valores nulos e sem linhas duplicadas; tipos de dados coerentes.
- `describe()` revelou **valores impossíveis acima de 100** em colunas que representam pontuações ou percentuais:

| Coluna | Valores > 100 |
|---|---|
| `math_prev_score` | 6 |
| `science_prev_score` | 11 |
| `language_prev_score` | 8 |
| `homework_completion_rate` | 47 |

**Tratamento aplicado**
- Remoção de `student_id` (sem valor analítico).
- Exclusão das linhas com valores acima de 100, conforme orientação dos coordenadores do projeto → a base passou de **1.000 para 933 alunos**.
- **Codificação ordinal** das variáveis categóricas, preservando a hierarquia natural:
  - `grade`: F < D < C < B < A → 0 a 4
  - `parent_education_level`: High School < Bachelor < Master → 0 a 2
  - `study_environment`: Noisy < Moderate < Quiet → 0 a 2
- **Padronização por Z-score** das variáveis numéricas contínuas, para que escalas diferentes (ex.: nota 0–100 vs. horas de sono 4–9) tenham o mesmo peso:

$$z = \frac{x - \mu}{\sigma}$$

### 2. Análise exploratória e descritiva

📓 `exploracaoDeDados.ipynb`

Todos os gráficos seguem a identidade visual da ICMC Jr, em **azul** (`#0000FF`) e **verde** (`#00FF00`).

#### Box plot — `daily_study_hours` por `pass_fail`
- A mediana de estudo dos reprovados fica em torno de **1,8 h/dia**; a dos aprovados chega a quase **2,9 h/dia** (cerca de 1 hora a mais).
- As caixas se sobrepõem: há alunos que estudaram mais de 4 h e reprovaram, e alunos que estudaram 1 h e passaram. O tempo de estudo aumenta a chance de aprovação, mas **não é o único fator determinante**.

#### Heatmap de correlações
| Relação | Correlação | Leitura |
|---|---|---|
| `exam_anxiety_score` × `motivation_score` | **−0,87** | Mais ansiedade, menos motivação |
| `screen_time_hours` × `sleep_hours` | **−0,63** | Mais tela, menos sono |
| `final_score` × `previous_score` | **0,84** | Melhor preditor isolado da nota final |
| `daily_study_hours` × `homework_completion_rate` | **0,91** | Quem estuda mais entrega mais tarefas |

Também se observa que o desempenho entre disciplinas é fortemente correlacionado (quem vai bem em uma área tende a ir bem nas outras) e que a frequência escolar se correlaciona positivamente com todas as métricas de nota.

#### Distribuição de `final_score`
- Distribuição **bimodal**, com dois picos: um entre **60 e 66 pontos** (desempenho intermediário) e outro entre **78 e 82 pontos** (alto desempenho).
- Uma pequena cauda à esquerda (40–50 pontos) representa a minoria com maiores dificuldades.

#### Gráfico de pontos — `final_score` por `attendance_percentage`
- Nuvem de pontos com inclinação ascendente: **mais frequência, maior nota final**.
- Os pontos coloridos por `pass_fail` mostram uma linha de aprovação próxima de 50 pontos, e alunos de baixa frequência e baixa nota se concentram entre os reprovados.

### 3. Teste estatístico

📓 `exploracaoDeDados.ipynb`

**Hipótese gerada pelo gráfico de dispersão:** maiores taxas de frequência estão associadas a notas finais mais altas.

1. **Shapiro-Wilk** (normalidade): p-valor < 0,05 tanto para `attendance_percentage` quanto para `final_score` → os dados **não** seguem distribuição normal.
2. Como a normalidade é rejeitada, foi usada a **correlação de Spearman**, que não exige esse pressuposto.
3. **Resultado:** correlação positiva forte (ρ ≈ 0,75) com p-valor praticamente zero → a relação não é fruto do acaso e a hipótese é confirmada.

```python
from scipy.stats import shapiro, spearmanr

stat_freq, p_freq = shapiro(df['attendance_percentage'])
stat_nota, p_nota = shapiro(df['final_score'])

rho, p = spearmanr(df['attendance_percentage'], df['final_score'])
```

### 4. Modelagem preditiva

📓 `modelagemPreditiva.ipynb`

**Pergunta:** é possível prever a nota final (`final_score`) de um aluno?

**Pipeline**
1. **Carregamento e limpeza** encapsulados em uma função reutilizável.
2. **Prevenção de data leakage:** `grade` e `pass_fail` foram removidas das variáveis explicativas, pois são consequências diretas da nota final.
3. **Separação treino/teste:** 70% / 30% (`random_state=50`).
4. **Escalonamento com `StandardScaler`**, ajustado (`fit`) apenas nos dados de treino e aplicado (`transform`) nos de teste, também para evitar vazamento de informação.
5. **Treinamento de dois modelos de regressão:**
   - Regressão Linear Múltipla
   - Árvore de Decisão (`DecisionTreeRegressor`, `max_depth=5` para evitar overfitting)
6. **Avaliação** com R² e erro médio absoluto (MAE) no conjunto de teste.

**Resultados**

| Modelo | R² | MAE (pontos) |
|---|---|---|
| **Regressão Linear Múltipla** | **0,8795** | **4,01** |
| Árvore de Decisão (profundidade 5) | 0,7584 | 5,58 |

A regressão linear múltipla foi o melhor modelo: explica cerca de **88%** da variação das notas finais e erra, em média, **4 pontos** em uma escala de 0 a 100.

---

## ✅ Conclusões

- A base tinha **inconsistências pontuais** (valores acima de 100) que foram corrigidas, resultando em 933 observações válidas.
- O desempenho passado (`previous_score`) é o preditor isolado mais forte da nota final, e **frequência** e **horas de estudo** também estão claramente associadas ao sucesso escolar.
- A relação entre frequência escolar e nota final foi confirmada estatisticamente (Spearman ≈ 0,75, p ≈ 0).
- É **possível prever a nota final** com boa precisão usando regressão linear múltipla (R² ≈ 0,88; MAE ≈ 4 pontos), o que indica que as notas não são aleatórias, mas resultado das características e hábitos dos alunos.

> ⚠️ Correlação não implica causalidade: os resultados descrevem associações presentes nos dados, não relações de causa e efeito.

## ▶️ Como executar

### Opção 1 — Google Colab (mais simples)
Clique nos badges **Open In Colab** acima. Os notebooks já baixam o CSV diretamente do GitHub.

### Opção 2 — Localmente

```bash
# 1. Clone o repositório
git clone https://github.com/leonardo-eid/Analise-Desempenho-Escolar_ICMC-Jr.git
cd Analise-Desempenho-Escolar_ICMC-Jr

# 2. (opcional) crie um ambiente virtual
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Instale as dependências
pip install pandas numpy matplotlib seaborn scipy scikit-learn jupyter

# 4. Abra os notebooks
jupyter notebook
```

> Os notebooks leem os dados pela URL do GitHub. Para rodar offline, troque a variável `url` por `'data/student_performance_interactions.csv'`.

Ordem recomendada: `limpezaDeDados.ipynb` → `exploracaoDeDados.ipynb` → `modelagemPreditiva.ipynb`.

## 🛠️ Tecnologias

- **Python 3**
- **pandas** — manipulação e limpeza de dados
- **matplotlib** e **seaborn** — visualizações
- **SciPy** — testes estatísticos (Shapiro-Wilk, Spearman)
- **scikit-learn** — `train_test_split`, `StandardScaler`, `LinearRegression`, `DecisionTreeRegressor`, `r2_score`, `mean_absolute_error`
- **Jupyter / Google Colab**
