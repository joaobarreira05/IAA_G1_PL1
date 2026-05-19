# Group Presentation Script: Anomaly Detection in Network Traffic
**Total Duration:** 10 minutes (approx. 3m20s per member)

---

## 🎙️ PART 1: João Barreira — Data Pipeline & EDA
**Duration:** ~3 minutes 20 seconds

### Slide 1: Title & Introduction
* **Visuals:** Project Title ("Anomaly Detection in Network Traffic"), Group Members (Gonçalo Almeida, Margarida Ribeiro, João Barreira), Course ("Introdução à Aprendizagem Automática"), Date.
* **João's Notes:**
  > "Bom dia/Boa tarde a todos. Nós somos o Grupo 1 e hoje vamos apresentar o nosso projeto final sobre Deteção de Anomalias em Tráfego de Rede. Focámo-nos em comparar algoritmos de machine learning clássicos e não supervisionados com modelos profundos de Deep Learning baseados em Autoencoders, abordando os desafios práticos de lidar com dados complexos e de grande dimensão."

### Slide 2: Context & Dataset (CIC-IDS2017)
* **Visuals:** Overview of the CIC-IDS2017 dataset. Key points: 2.8 million flows, 78 raw statistical features, severe class imbalance (80% Benign / 20% Anomalous).
* **João's Notes:**
  > "Para este estudo utilizámos o dataset CIC-IDS2017, que reflete tráfego real com ataques cibernéticos modernos. Os nossos dois maiores desafios iniciais foram: primeiro, o volume bruto dos dados (mais de 2.8 milhões de registos), o que provocava erros de memória (OOM) nas nossas máquinas locais na Fase 1; e segundo, o forte desbalanceamento de classes, com os ataques a constituir apenas cerca de 20% do tráfego."

### Slide 3: Exploratory Data Analysis (EDA)
* **Visuals:** Side-by-side plots: Class distribution and feature correlation heatmap.
* **João's Notes:**
  > "A análise exploratória mostrou que os atributos tinham magnitudes drasticamente diferentes — por exemplo, a duração dos fluxos variava de microsegundos a mais de 100 segundos, enquanto os tamanhos dos pacotes estavam limitados a 1500 bytes. Identificámos também que 8 features tinham variância zero, não fornecendo qualquer poder discriminativo."

### Slide 4: Data Processing Pipeline
* **Visuals:** Flowchart: Raw CSVs ➔ Data Cleaning (NaN/Inf removal) ➔ Drop Zero-Variance ➔ 20% Stratified Sampling ➔ 70/15/15 Split ➔ StandardScaler (fitted only on Train).
* **João's Notes:**
  > "Para ultrapassar os limites computacionais de forma rigorosa, criámos uma pipeline de processamento automatizada. Eliminámos NaNs e infinitos (menos de 0.1% dos dados), removemos as variáveis estáticas e aplicámos uma amostragem estratificada de 20%. Isto reduziu a amostra para cerca de 565.000 instâncias, preservando a proporção exata de 80/20. Escalámos os dados com o StandardScaler, ajustado exclusivamente na partição de treino para evitar data leakage."

### Slide 5: Evaluation Strategy
* **Visuals:** Core Metrics (F1-Score, Recall, Precision, AUC-ROC). Comparison of Original vs. Balanced splits.
* **João's Notes:**
  > "Neste cenário de desbalanceamento, a Accuracy tradicional seria uma métrica enganadora (um classificador que preveja sempre 'normal' atinge 80% de precisão mas falha em apanhar qualquer ataque). Focámo-nos por isso no Recall e F1-Score, avaliando os modelos em três perspetivas: a distribuição desbalanceada original, um subconjunto equilibrado 50-50 para expor a sensibilidade do modelo, e análise detalhada por classe."

*(Passes slide control/voice to Margarida)*

---

## 🎙️ PART 2: Margarida Ribeiro — State of the Art & Classical Models
**Duration:** ~3 minutes 20 seconds

### Slide 6: State of the Art & Literature Review
* **Visuals:** Table summarizing the 6 reviewed works (Isolation Forest, LOF, One-Class SVM, Autoencoders, Kitsune, DAGMM).
* **Margarida's Notes:**
  > "Obrigada, João. Começámos por rever o estado da arte na literatura. Analisámos três métodos clássicos não supervisionados e três abordagens de redes neuronais. A literatura aponta que métodos baseados em árvores e redes neuronais dominam abordagens baseadas em densidade como o LOF quando a dimensão do dataset ultrapassa as 20 variáveis. Além disso, identificámos que a otimização sistemática de hiperparâmetros é muitas vezes ignorada na literatura de cibersegurança."

### Slide 7: Classical Models Evaluated
* **Visuals:** Operating principles in brief bullets for Isolation Forest, One-Class SVM (kernel space boundary), and LOF (density ratio).
* **Margarida's Notes:**
  > "Focámo-nos em três algoritmos clássicos: o Isolation Forest, que isola anomalias selecionando aleatoriamente features e cortes; o One-Class SVM, que define uma fronteira em torno do tráfego normal em alta dimensão usando kernels; e o Local Outlier Factor, que estima anomalias com base na densidade local dos k-vizinhos mais próximos."

### Slide 8: Hyperparameter Tuning Strategy (Cross-Validation)
* **Visuals:** Flowchart of the 3-Fold Stratified Cross-Validation loop used to optimize parameters on the training set, keeping the test set strictly hidden.
* **Margarida's Notes:**
  > "Para responder a preocupações sobre a robustez estatística de uma única partição, adotámos Cross-Validation Estratificada de 3-folds para a seleção de parâmetros. Cada configuração candidata foi avaliada pela média do F1-score no CV. A avaliação final dos modelos foi depois realizada no conjunto de teste independente que mantivemos sempre isolado."

### Slide 9: Tuning Heatmaps (Isolation Forest & OC-SVM)
* **Visuals:** Heatmaps of CV F1-Score for Isolation Forest (contamination vs. estimators) and One-Class SVM (nu vs. gamma).
* **Margarida's Notes:**
  > "Os resultados mostram que a variável de 'contamination' é a mais crítica para o Isolation Forest. No One-Class SVM, o kernel RBF superou de longe o polinomial. Devido à complexidade computacional quadrática do OC-SVM, a sintonia teve de ser feita num subconjunto estratificado de 5.000 instâncias."

### Slide 10: Baseline vs. Tuned Performance Comparison
* **Visuals:** Bar plot comparing default vs. tuned configurations on the test set. IF: +5.6% F1 (0.289 to 0.345); OC-SVM: +3.0% F1; LOF: negligible change.
* **Margarida's Notes:**
  > "O Isolation Forest foi o que mais beneficiou da otimização (+5.6 pontos percentuais no F1). O LOF quase não melhorou, confirmando que a sua falha é estrutural devido à curse of dimensionality. O espaço de 70 features faz com que todas as distâncias pareçam uniformes, inviabilizando abordagens baseadas em densidade."

*(Passes control to Gonçalo)*

---

## 🎙️ PART 3: Gonçalo Almeida — Deep Learning, Results & Conclusions
**Duration:** ~3 minutes 20 seconds

### Slide 11: Deep Learning Architectures (Autoencoders)
* **Visuals:** Architecture diagram of Primary AE (trained on Normal), Secondary AE (trained on Attack), and the Ensemble AE (logical OR combination).
* **Gonçalo's Notes:**
  > "Obrigada, Margarida. Para contornar as limitações dos algoritmos clássicos em alta dimensão, desenhámos três arquiteturas de redes neuronais em PyTorch. O Autoencoder Primário foi treinado apenas em tráfego benigno para aprender a reconstruir padrões normais. O Autoencoder Secundário foi treinado em tráfego de ataques para aprender assinaturas maliciosas. E o Autoencoder Ensemble combina as duas decisões numa lógica OR: se qualquer uma das redes detetar, o tráfego é classificado como anomalia."

### Slide 12: Training Convergence & Loss
* **Visuals:** Training loss comparison curve for the Primary and Secondary Autoencoders over epochs.
* **Gonçalo's Notes:**
  > "O Autoencoder Primário convergiu suavemente até à época 29 com uma perda MSE final de 0.0615. A rede secundária convergiu mais rapidamente (em apenas 10 épocas) por os ataques terem padrões muito consistentes. Não observámos sobreajuste (overfitting) nos dados de validação, pois a capacidade foi controlada pelo estreitamento do bottleneck de 70 para 8 neurónios na camada central."

### Slide 13: Final Results & Comparison Table
* **Visuals:** The main comparison table showing metrics for all 6 models under Original and Balanced distributions. Highlighting the Ensemble AE and Second AE.
* **Gonçalo's Notes:**
  > "Os resultados finais revelaram uma divisão clara. Os modelos de deep learning superaram amplamente os clássicos. O Autoencoder Ensemble alcançou uma área sob a curva ROC fantástica de 0.9810 e um Recall de 95.30%, apanhando quase todos os ataques no conjunto de teste. O Autoencoder Secundário destacou-se pela precisão, atingindo 77.81% na distribuição realista, o que reduz bastante os falsos alarmes."

### Slide 14: Diagnostic Analysis: Ensemble Autoencoder
* **Visuals:** Confusion matrix and ROC curve for the Ensemble Autoencoder (AUC = 0.981).
* **Gonçalo's Notes:**
  > "Ao analisar os diagnósticos da nossa melhor rede, vemos que o ROC sobe quase verticalmente. A matriz de confusão mostra que em 16.697 ataques de teste reais, apenas 785 passaram despercebidos. A precisão moderada de 58.55% gera cerca de 8% de falsos alarmes no tráfego normal. Em engenharia de segurança, este compromisso é altamente vantajoso: preferimos investigar falsos alarmes do que sofrer uma intrusão catastrófica silenciosa."

### Slide 15: Critical Reflection & AI Tools
* **Visuals:** Key insights: theory vs. practice gap, ethical dimensions (bias, false positives), and responsible AI tools selection rationale.
* **Gonçalo's Notes:**
  > "Este projeto permitiu-nos ligar a teoria da sala de aula aos desafios práticos da engenharia de ML — como gerir OOMs e evitar fuga de informação (data leakage). Adicionalmente, refletimos sobre as considerações éticas: um falso positivo pode suspender utilizadores legítimos de forma injusta. Declaramos também que usámos o Gemini Pro para ajudar a debugar código PyTorch e ferramentas de IA focadas em pesquisa bibliográfica científica para manter a integridade académica."

### Slide 16: Conclusions & Future Work
* **Visuals:** Bullet summary: Deep learning beats classical in 70D; Ensemble AE wins for high security; Second AE wins for automated operation. Future work: attention mechanisms, hybrid models.
* **Gonçalo's Notes:**
  > "Em conclusão, os autoencoders deep learning demonstraram ser a solução ideal para lidar com a alta dimensionalidade. O Ensemble é a nossa recomendação para monitorização ativa de segurança, e o Autoencoder Secundário é o recomendado para sistemas de mitigação automática. Em trabalho futuro, planeamos estudar mecanismos de atenção temporal e modelos híbridos com GMMs. Obrigado pela vossa atenção, estamos agora abertos a perguntas."
