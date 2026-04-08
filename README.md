# IAA_G1_PL1 - Deteção de Anomalias de Tráfego de Rede (CIC-IDS2017)

## Setup e Data Processing (Para a Margarida e para o Gonçalo)

Como os ficheiros originais (`archive/*`) excedem 1.5GB e os ficheiros processados ultrapassam em muito o limite de 100 MB do GitHub, optei por centralizar o **Data Processing** num simples *script* Python. Desta forma mantemos o Repositório leve, mas toda a gente na equipa consegue gerar a partição exata dos dados (`X_train.csv`, `X_test.csv` etc.) nas suas próprias máquinas em menos de um minuto.

Siga os seguintes passos no vosso computador:

### 1. Garantir que têm os Dados Originais na pasta `archive/`
Certifiquem-se de que a pasta `archive/` contém os 8 ficheiros originais descarregados do CIC-IDS2017 (`Monday-WorkingHours.pcap_ISCX.csv`, etc.).

### 2. Criar Virtual Environment e Instalar Dependências
Abram dois terminais na pasta principal do projeto (`IAA_G1_PL1`) e efetuem a instalação das mesmas packages para todos termos o ambiente igual:

No macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

No Windows (Command Prompt):
```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Rodar o Pipeline de Pré-processamento
Para executar a limpeza (remover *infinitos* e *nulos*), eliminação de features com variância=0 (sem ruído para o Isolation Forest), e para criar uma hiper-extracção equitativa de **20%** dos dados originais estratificados por ataques, executem:

```bash
python3 src/data_processing.py
```

Isto vai ler todos os CSVs, escalar (*StandardScaler*, acoplado apenas no treino), converter as labels, e fazer o split Train (70%), Val (15%), Test (15%). Quando o *script* chegar a 100%, verão a mensagem: **Data processing complete.**

### 4. Utilizar os Dados
Pronto! Verificam que foi gerada uma pasta oculta do *git* chamada **`processed_data/`** no projeto de vocês. Contém:
- `X_train.csv` / `y_train.csv`
- `X_val.csv` / `y_val.csv`
- `X_test.csv` / `y_test.csv`

**Margarida/Gonçalo:** Nas vossas experiências e Notebooks relativos às models, só precisam de começar os vossos códigos fazendo `pd.read_csv('processed_data/X_train.csv')`. Todos estaremos a operar exatamente na mesma partição da base de dados e com as mesmas variáveis normalizadas e com colunas idênticas.

Bom trabalho de modelação! 🚀
*(João Barreira)*
