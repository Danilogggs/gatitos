# PjBL2 — avaliação da classificação de raças

**Modelo:** `efficientnet-b0-oxford-epoch-7-colors-v1-length-v1-reviewed-colors-candidate`. **Base:** Oxford-IIIT Pet, split `test`,
subconjunto balanceado de 240 imagens originais (20 por raça).
Cada imagem foi avaliada individualmente. O script de seleção usa SHA-256 do índice
Oxford e exclui seis imagens usadas antes na galeria de demonstração.

## Aderência aos requisitos do PjBL2

1. **Base com pelo menos 3 categorias:** Oxford-IIIT Pet, com 12 raças de gatos utilizadas como classes.
2. **Protocolo experimental:** o `trainval` oficial é dividido de forma estratificada em treino e validação
   com semente 42; o `test` oficial permanece fora do ajuste. Este relatório usa 20 imagens de teste por classe.
3. **Modelo de visão computacional:** EfficientNet-B0 com transferência de aprendizado e uma cabeça linear
   de 12 saídas. A implementação reproduzível está em `backend/app/ml/train_oxford.py` e no notebook Colab.
4. **Avaliação:** matriz de confusão, acurácia, precisão, recall e F1 por classe, além das médias macro
   e ponderada. Os CSVs e as predições individuais acompanham este relatório.

## Resultado principal: 12 raças treinadas

- Acurácia top-1: **219/240 = 91.25%**.
- Precisão macro: **91.80%**; recall macro: **91.25%**; F1 macro: **91.21%**.
- Precisão ponderada: **91.80%**; recall ponderado: **91.25%**; F1 ponderado: **91.21%**.
- A matriz de confusão está em [matriz_confusao_12_racas.csv](matriz_confusao_12_racas.csv):
linhas são raças reais; colunas, raças previstas; diagonal são acertos.

| Raça | Imagens | Precisão | Recall | F1 |
|---|---:|---:|---:|---:|
| abissínio | 20 | 100.0% | 95.0% | 97.4% |
| bengal | 20 | 88.9% | 80.0% | 84.2% |
| sagrado da Birmânia | 20 | 79.2% | 95.0% | 86.4% |
| bombaim | 20 | 100.0% | 100.0% | 100.0% |
| britânico de pelo curto | 20 | 94.4% | 85.0% | 89.5% |
| mau egípcio | 20 | 85.7% | 90.0% | 87.8% |
| Maine Coon | 20 | 90.5% | 95.0% | 92.7% |
| persa | 20 | 90.9% | 100.0% | 95.2% |
| ragdoll | 20 | 94.1% | 80.0% | 86.5% |
| azul russo | 20 | 82.6% | 95.0% | 88.4% |
| siamês | 20 | 100.0% | 80.0% | 88.9% |
| sphynx | 20 | 95.2% | 100.0% | 97.6% |

### Matriz de confusão — 12 raças

Linhas = raça real; colunas = previsão top-1; a diagonal indica acertos.

| Real \ Prev | Aby | Ben | Bir | Bom | BSh | Mau | MCo | Per | Rag | Rus | Sia | Sph |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Aby | 19 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| Ben | 0 | 16 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | 0 | 0 | 0 |
| Bir | 0 | 0 | 19 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| Bom | 0 | 0 | 0 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| BSh | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 0 | 0 | 3 | 0 | 0 |
| Mau | 0 | 2 | 0 | 0 | 0 | 18 | 0 | 0 | 0 | 0 | 0 | 0 |
| MCo | 0 | 0 | 0 | 0 | 0 | 0 | 19 | 1 | 0 | 0 | 0 | 0 |
| Per | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 20 | 0 | 0 | 0 | 0 |
| Rag | 0 | 0 | 2 | 0 | 0 | 0 | 1 | 1 | 16 | 0 | 0 | 0 |
| Rus | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 19 | 0 | 0 |
| Sia | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 16 | 1 |
| Sph | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 20 |

Legenda: Aby = abissínio; Ben = bengal; Bir = sagrado da Birmânia; Bom = bombaim; BSh = britânico de pelo curto; Mau = mau egípcio; MCo = Maine Coon; Per = persa; Rag = ragdoll; Rus = azul russo; Sia = siamês; Sph = sphynx.

Confusões mais frequentes:
- siamês → sagrado da Birmânia: 3 imagem(ns).
- britânico de pelo curto → azul russo: 3 imagem(ns).
- bengal → mau egípcio: 3 imagem(ns).
- ragdoll → sagrado da Birmânia: 2 imagem(ns).
- mau egípcio → bengal: 2 imagem(ns).
- siamês → sphynx: 1 imagem(ns).
- azul russo → britânico de pelo curto: 1 imagem(ns).
- ragdoll → persa: 1 imagem(ns).

## Resposta exibida no aplicativo

- Com limiar de confiança de 40%, **5/240** imagens receberam “SRD”
  como rejeição por baixa confiança. SRD **não é uma classe treinada**.
- Acertos no conjunto todo: **218/240 = 90.83%**.
- Cobertura com raça sugerida: **235/240 = 97.92%**;
  acurácia apenas nas sugestões emitidas: **218/235 = 92.77%**.
- A matriz operacional está em [matriz_confusao_app_com_srd.csv](matriz_confusao_app_com_srd.csv).

## Método e limites

- O treino de raça usa `trainval` Oxford, separado por classe em treino e validação
  com semente 42; o checkpoint é escolhido pela acurácia de validação.
- O split `test` Oxford fica fora do ajuste dos pesos de raça. Seu resultado, porém,
  já foi consultado em experimentos anteriores: este relatório não é uma estimativa
  totalmente inédita de generalização.
- As métricas principais são de raça, **não** de cor ou comprimento do pelo.
- Precisão = VP/(VP+FP); recall = VP/(VP+FN); F1 = 2·precisão·recall/(precisão+recall).
- [predicoes_teste.csv](predicoes_teste.csv) contém os rótulos e resultados por imagem
  para auditar os cálculos, sem redistribuir as fotos.
- Para reproduzir: em `backend/`, execute
  `python -m app.ml.report_breed_metrics --input ../dataset/avaliacao-oxford-240/resultados.csv --output ../reports/pjbl2`.
