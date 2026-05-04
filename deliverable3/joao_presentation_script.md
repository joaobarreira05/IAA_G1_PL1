# Presentation Script - João Barreira (Data Pipeline & EDA)

**Duration:** ~3m20s (1/3 of the 10-minute presentation)

## Slide 1: Title & Introduction
**Visual:** Project Title, Group Members, Course Name.
**Speaker Notes (João):**
"Bom dia/Boa tarde a todos. Nós somos o Grupo 1, compostos por mim (João), pela Margarida e pelo Gonçalo. O nosso projeto da disciplina de Introdução à Aprendizagem Automática abordou a Deteção de Anomalias em Tráfego de Rede. Hoje vamos apresentar a evolução do nosso trabalho, os desafios encontrados ao lidar com um cenário de *big data* e as arquiteturas clássicas e profundas (Deep Learning) que aplicámos."

## Slide 2: Context and Dataset (CIC-IDS2017)
**Visual:** Brief bullet points on the CIC-IDS2017 dataset. Mention the high volume (>1GB) and the class imbalance (80% benign / 20% anomalous).
**Speaker Notes (João):**
"A nossa investigação assentou no dataset CIC-IDS2017. Trata-se de um conjunto de dados muito representativo de tráfego real com ataques modernos. Os nossos dois maiores desafios iniciais foram o enorme volume de dados — mais de 2.8 milhões de registos originais — e o forte desbalanceamento de classes, em que os ataques representam apenas cerca de 20% de todo o tráfego."

## Slide 3: Exploratory Data Analysis (EDA)
**Visual:** Correlation heatmap of top variance features (from `deliverable3/images/feature_correlation.png`) and Class Distribution bar chart.
**Speaker Notes (João):**
"Através da nossa Análise Exploratória, constatámos que certas *features* extraídas não continham qualquer variação, sendo irrelevantes estatisticamente. Observámos também diferenças enormes de magnitude — a duração de um fluxo variou de microsegundos a mais de 100 segundos, o que afeta drasticamente algoritmos baseados em distâncias. No gráfico podemos observar a correlação forte entre variáveis de volume e velocidade."

## Slide 4: Data Processing Pipeline
**Visual:** A flowchart showing: Raw Data -> Clean NaN/Inf -> Drop Zero-Variance -> 20% Stratified Sampling -> Split (70/15/15) -> StandardScaler.
**Speaker Notes (João):**
"Para resolver o problema da dimensão e dos OOM errors (falta de memória) que as máquinas locais sofriam na fase 1, implementámos um pipeline robusto. Limpámos valores infinitos e *NaNs*, e aplicámos um *Stratified Sampling* de 20%. Desta forma, preservámos de forma fidedigna a distribuição de 80/20 num tamanho de amostra perfeitamente treinável. O conjunto foi dividido e os dados foram normalizados apenas com base no conjunto de treino, prevenindo rigorosamente o *data leakage*."

## Slide 5: Evaluation Strategy
**Visual:** Evaluation Metrics (F1-Score, AUC-ROC, Recall). Mention the Cross-Validation approach implemented vs Single-Split justification.
**Speaker Notes (João):**
"Na avaliação, percebemos que analisar a *Accuracy* seria inútil devido ao forte desbalanceamento. A nossa estratégia assentou numa framework multi-perspetiva focada no Recall, F1-Score e AUC-ROC. Dado o grande volume (565 mil instâncias), implementámos a nossa pipeline de tal forma que testámos os modelos finais num subconjunto de Teste absolutamente isolado, garantindo uma validação fidedigna de que os nossos algoritmos não estavam apenas a decorar os dados de treino."

*(Hands over to Margarida for State of the Art and Classical ML models)*
