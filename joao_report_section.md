# 3. Data Processing Methodology

A preparação dos dados é um passo crítico no ciclo de vida de qualquer projeto de Machine Learning, especialmente em cenários de domínios complexos como a deteção de anomalias em tráfego de rede. O *dataset* CIC-IDS2017 apresenta características inerentes a dados do mundo real: elevada volumetria, presença de valores corrompidos ou em falta, bem como assimetria nas classes. 

O pipeline de processamento foi desenhado tendo em conta a otimização de recursos e a fiabilidade estatística, estruturando-se nas seguintes fases:

## 3.1. Data Ingestion and Sampling
O conjunto de dados totaliza mais de 1GB de tráfego, compreendido em múltiplos ficheiros CSV correspondentes a vários dias de captura. Dada a redundância de padrões normais, optou-se por agregar os dados e aplicar uma **amostragem estratificada (Stratified Sampling) de 20%**. Esta técnica garantiu que a amostra gerada mantém fielmente a mesma proporção de instâncias benignas (80.3%) e de anomalias (19.7%) presente no *dataset* original, resultando num volume de dados em torno de 565 mil instâncias, que se adequa favoravelmente aos requisitos de memória dos algoritmos em ambiente *offline* (como o Isolation Forest) durante o rápido ciclo de experimentação.

## 3.2. Data Cleaning
Durante a fase exploratória (Deliverable 1), identificaram-se problemas de formatação e valores inválidos:
1. **Limpeza Estrutural:** Foram removidos espaços residuais nos nomes das colunas, algo crítico para evitar discrepâncias em *calls* intermédias na análise (*e.g.* repor ` Flow Duration` para `Flow Duration`).
2. **Tratamento de Missing e Infinity Values:** Embora residuais (afetando aproximadamente 2800 instâncias, ou seja, <0.1% do dataset), a presença de valores infinitos (devido a divisões por zero durante a extração das features no pcap, ex: `Flow Bytes/s`) ou nulos (`NaN`) impacta diretamente modelos matemáticos sensíveis a domínios numéricos. Optou-se pela exclusão integral (Drop) destas linhas (`dropna`), assumindo que a sua reduzida predominância não traduz perda de contexto comportamental útil. A imputação de dados num caso de "velocidades infinitas" de pacotes não revelaria comportamento orgânico de rede, introduzindo *bias*.

## 3.3. Feature Engineering & Selection 
A adequação das colunas do dataset ao modelo implica a redução de ruído e da dimensionalidade desnecessária. 
- **Verificação de Variância Zero:** Foram retiradas 8 features que se demonstraram ter uma variância de zero ao longo das capturas de rede (*e.g.* `Bwd PSH Flags`, `Bwd URG Flags`, `Fwd Avg Bytes/Bulk`, etc.). Atributos estáticos limitam a utilidade analítica, introduzindo ruído computacional, e não agregam variabilidade necessária a deteções baseadas em distância.
- **Categorização de Labels:** No contexto desta técnica não supervisionada (ainda que avaliada com *labels* em *baseline* binária), a variável categórica do tipo de tráfego foi mapeada num domínio numérico inteiro (`0` para `BENIGN`, `1` para calhar contexto anómalo). Todas as análises a posteriori ignorarão esta feature nas medições não-supervisionadas, atuando apenas como pilar de avaliação.

## 3.4. Feature Scaling and Train/Test Split
A fase final focou-se na preservação preventiva de viés informacional (Data Leakage):
- **Split do Dataset:** Dividiu-se a amostra estratificada com uma proporção de 70% para Treino (Train), 15% para Validação (Validation) e 15% para Teste (Test). Isto garante instâncias virgens o suficiente num ambiente independente de hiper-teste.
- **Normalização (Standarization):** Verificou-se que métricas como a volumetria de pacotes vs duração do fluxo variam em ordens de grandeza distintas. Assim, procedeu-se ao normalização dos atributos com base no pressuposto da sua variação, recorrendo ao `StandardScaler`. Especial atenção foi dada ao *framework* temporal de scaling: **Fit do scaler exclusivamente ao subset de Train**, transformando subsequentemente os conjuntos de validação e teste com a mesma base.

---

# 4. Adjustments made since Deliverable 1
Durante o desenvolvimento para o Deliverable 2 e as iterações exploratórias associadas, implementaram-se ajustes vitais face a algumas descobertas identificadas após o Deliverable 1:

* ***Feature Usefulness***: Constatou-se na fase preliminar que múltiplas features não possuíam correlação ou desvios relevantes, sendo apenas resultantes de fluxos que não completaram ações na métrica estatuída pela CICFlowMeter (a ferramenta da CIC que extraiu os fluxos de rede originais). A inserção do mecanismo dinâmico de exclusão de `Variance == 0` ajudou a compactar o algoritmo dos modelos face ao inicialmente planeado no Deliverable 1 sem recurso a reduções de PCA cegos.
* ***Formatação de Timestamps/Identificadores***: Uma exploração atrativa revelou que o formato nativo não incluiu, numa proporção considerativa, IPs/Ports absolutos da rede ou timestamps sequenciais limpos. Focou-se, por isso, inteiramente nos artefactos estatísticos contínuos.
* ***Distribuição e Volume***: Relativamente ao problema do desequilíbrio e volume maciço, consolidou-se que carregar as 2.8 milhões de instâncias simultaneamente em algoritmos de modelação matricial (*e.g.* Isolation Forest) num cenário comum submetia as máquinas locais a colapsos de OOM (*Out Of Memory*). Assim, reajustou-se as expectativas operacionais e adotou-se o modelo de sampling estratificado com output físico (`X_train.csv`). O desequilíbrio entre classes (embora substancial, 80-20), não sofreu SMOTE deliberado neste *stage*, visto as métricas de Precision e Recall para Isolation Forests exigirem uma distribuição nativa comportamental realista sob o viés original do dataset para funcionar com fiabilidade métrica final.
