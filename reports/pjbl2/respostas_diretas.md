# PjBL2 — respostas diretas

## 1. Qual base de imagens foi utilizada e quais são as categorias?

Foi utilizada a base pública **Oxford-IIIT Pet**. O projeto utiliza as 12 categorias de gatos presentes na base:

1. Abissínio (`abyssinian`)
2. Bengal (`bengal`)
3. Sagrado da Birmânia (`birman`)
4. Bombaim (`bombay`)
5. Britânico de pelo curto (`british_shorthair`)
6. Mau egípcio (`egyptian_mau`)
7. Maine Coon (`maine_coon`)
8. Persa (`persian`)
9. Ragdoll (`ragdoll`)
10. Azul russo (`russian_blue`)
11. Siamês (`siamese`)
12. Sphynx (`sphynx`)

Portanto, o projeto supera o mínimo solicitado de três categorias. A base é carregada automaticamente pelo `torchvision`; as imagens não são redistribuídas no repositório.

## 2. Como as imagens foram separadas em treino, validação e teste?

Foi aplicado o seguinte protocolo experimental:

- **Treino e validação:** partiram do split oficial `trainval` do Oxford-IIIT Pet, que contém 1.188 imagens de gatos das 12 raças utilizadas.
- **Treino:** 950 imagens.
- **Validação:** 238 imagens.
- A divisão entre treino e validação foi **estratificada por raça**, na proporção aproximada de 80%/20%, com semente fixa `42`.
- **Teste:** foi mantido separado o split oficial `test`, com 1.183 imagens de gatos. Ele não foi usado para ajustar os pesos nem para escolher o melhor checkpoint.
- Para o relatório final foi selecionado um subconjunto balanceado de **240 imagens de teste**, sendo 20 imagens de cada raça.
- A seleção das 240 imagens é determinística, usa SHA-256 do índice da imagem e exclui seis imagens utilizadas anteriormente na galeria de demonstração.
- Durante o treino são aplicados redimensionamento, recorte aleatório e espelhamento horizontal. Validação e teste usam somente redimensionamento e normalização, sem aumentos aleatórios.

## 3. Qual modelo de visão computacional foi aplicado?

Foi utilizada uma **EfficientNet-B0 pré-treinada no ImageNet**, por transferência de aprendizado.

- O classificador original foi removido.
- Foi adicionada uma camada linear com **12 saídas**, uma para cada raça.
- Função de perda: `CrossEntropyLoss`.
- Otimizador: `AdamW`.
- Taxa de aprendizado: `1e-4`.
- Configuração padrão: 8 épocas e lotes de 16 imagens.
- O checkpoint escolhido é o que apresenta a maior acurácia no conjunto de validação.
- Na aplicação, o modelo retorna a pontuação das 12 raças. Quando a maior pontuação é de 40% ou menos, a interface mostra “SRD” como rejeição por baixa confiança. **SRD não é uma classe treinada.**

A implementação reproduzível está em `backend/app/ml/train_oxford.py` e no notebook `notebooks/CatCare_AI_treinamento.ipynb`.

## 4. Quais foram as métricas de desempenho?

O modelo foi avaliado nas 240 imagens balanceadas do conjunto de teste.

- **Acurácia top-1:** 219/240 = **91,25%**.
- **Precisão macro:** **91,80%**.
- **Recall macro:** **91,25%**.
- **F1 macro:** **91,21%**.
- **Precisão ponderada:** **91,80%**.
- **Recall ponderado:** **91,25%**.
- **F1 ponderado:** **91,21%**.

### Métricas por categoria

| Categoria | Imagens | Precisão | Recall | F1 |
|---|---:|---:|---:|---:|
| Abissínio | 20 | 100,0% | 95,0% | 97,4% |
| Bengal | 20 | 88,9% | 80,0% | 84,2% |
| Sagrado da Birmânia | 20 | 79,2% | 95,0% | 86,4% |
| Bombaim | 20 | 100,0% | 100,0% | 100,0% |
| Britânico de pelo curto | 20 | 94,4% | 85,0% | 89,5% |
| Mau egípcio | 20 | 85,7% | 90,0% | 87,8% |
| Maine Coon | 20 | 90,5% | 95,0% | 92,7% |
| Persa | 20 | 90,9% | 100,0% | 95,2% |
| Ragdoll | 20 | 94,1% | 80,0% | 86,5% |
| Azul russo | 20 | 82,6% | 95,0% | 88,4% |
| Siamês | 20 | 100,0% | 80,0% | 88,9% |
| Sphynx | 20 | 95,2% | 100,0% | 97,6% |

### Matriz de confusão

As linhas representam as categorias reais e as colunas representam as previsões. A diagonal contém os acertos.

| Real \ Prevista | Aby | Ben | Bir | Bom | BSh | Mau | MCo | Per | Rag | Rus | Sia | Sph |
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

Os resultados completos e auditáveis estão em:

- `reports/pjbl2/matriz_confusao_12_racas.csv`
- `reports/pjbl2/metricas_por_raca.csv`
- `reports/pjbl2/predicoes_teste.csv`
- `reports/pjbl2/relatorio.md`

## Conclusão direta

O projeto atende ao PjBL2 porque utiliza uma base real com 12 categorias, aplica separação reproduzível entre treino, validação e teste, treina uma rede neural de visão computacional e apresenta matriz de confusão e métricas de classificação. No teste balanceado, o modelo alcançou **91,25% de acurácia** e **91,21% de F1 macro**.
