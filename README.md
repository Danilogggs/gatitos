# CatCare AI

Plataforma de cadastro e divulgação de gatos para adoção com revisão humana obrigatória. O cadastro exige nome e ao menos uma foto. Fotos são armazenadas no Supabase Storage; dados ficam no PostgreSQL. O classificador de imagens sugere características visuais e o Gemini pode escrever a descrição. O administrador confere e corrige antes da publicação.

## Estrutura

- `frontend/`: React, TypeScript e Vite.
- `backend/`: FastAPI, integração com Supabase, serviços ML e Gemini.
- `supabase/schema.sql`: tabelas, índices, RLS e bucket.

## Configuração

1. Clone o repositório em qualquer PC com Python, Node.js e Git instalados. O `.env` público, `frontend/.env` e `backend/models/catcare-oxford.pt` já estão incluídos.
2. Crie `.env.local` na raiz com `SUPABASE_SERVICE_ROLE_KEY=...` para usar banco, Storage e cadastros. Opcionalmente adicione `GEMINI_API_KEY=...` para descrições com Gemini. Esse arquivo é ignorado pelo Git e seus valores substituem os do `.env`.
3. No PowerShell, entre em `backend/`, rode `py -m venv .venv`, `./.venv/Scripts/python.exe -m pip install -r requirements.txt` e `./.venv/Scripts/python.exe -m uvicorn app.main:app --reload`.
4. Em outro terminal, entre em `frontend/`, rode `npm ci` e `npm run dev`. Abra `http://localhost:5173`.

O modelo é carregado por caminho relativo ao backend. Para criar outro projeto Supabase, execute `supabase/schema.sql` no SQL Editor, altere as configurações dos dois `.env` e cadastre um administrador em `public.admin_users`.

## ML e LLM

`ML_MODE=mock` deixa explícito na interface que o modelo não está treinado e **não cria palpites falsos**. Para `ML_MODE=real`, o checkpoint deve conter `backbone`, `breed_head`, `breeds` e `version`. As cabeças opcionais `color_head` e `length_head` só produzem previsões quando estiverem treinadas e listadas em `trained_heads`. O checkpoint local `catcare-oxford-colors-length.pt` preserva a cabeça de raça e acrescenta cabeças de cores e comprimento do pelo. A prévia e o novo cadastro usam somente esses três atributos. O dataset não entra no repositório.

Como o Oxford não contém exemplos de SRD, o backend registra a raça com maior pontuação entre as 12 classes quando ela é **maior que** `ML_BREED_MIN_CONFIDENCE` (padrão: `0.40`). Se a maior pontuação for 40% ou menos, usa `srd` como resultado provisório. Com várias fotos, as pontuações são calculadas a partir da média das representações visuais. Aumentar o limite manda mais gatos para SRD e pode rejeitar raças reais; diminuí-lo aceita mais raças e pode classificar um SRD como uma delas. Uma pontuação alta **não prova** que o gato é de raça pura nem garante a identificação de todo SRD. A revisão do administrador é obrigatória. O campo `breed_confidence` fica vazio no fallback SRD porque o modelo não foi treinado para medir confiança nessa classe.

A prévia do cadastro mostra as pontuações das 12 raças treinadas em ordem decrescente. Esses valores são relativos às classes do checkpoint e não representam a probabilidade de pedigree; SRD não recebe pontuação porque não foi treinado como classe.

O Gemini usa `GEMINI_API_KEY` e `GEMINI_MODEL` apenas para redigir a descrição com os dados já identificados. A análise visual e os campos de características e padrão foram retirados do fluxo de novos cadastros. Raça, cores e comprimento vêm do modelo local; códigos internos permanecem estáveis no banco e são apresentados em português nas telas e na descrição sugerida. Se o Gemini não estiver configurado ou falhar, a chamada é repetida brevemente em erros temporários e uma descrição factual local é usada. Comportamento somente entra no texto se uma pessoa o informou. Ao salvar correções administrativas de raça, pelo, cores ou demais dados, a descrição é atualizada se o administrador não tiver editado o texto manualmente nessa operação. Para registros já corrigidos anteriormente, use **Atualizar descrição** no painel. Saúde, localização e observações são acrescentadas localmente: não são enviadas ao Gemini.

## Treinamento no Colab

Abra o [notebook de treinamento com Oxford-IIIT Pet](https://colab.research.google.com/github/Danilogggs/gatitos/blob/main/notebooks/CatCare_AI_treinamento.ipynb), selecione uma GPU e execute as células em ordem. Ele baixa o dataset automaticamente, seleciona as 12 raças de gatos, treina o EfficientNet-B0 e oferece o download de `catcare-oxford.pt`. Você não precisa enviar fotos nem `labels.csv`. O Oxford não inclui uma classe SRD nem rótulos das características de pelagem do aplicativo; o fallback SRD ocorre no backend após o treino, conforme o limiar acima. O checkpoint deste teste já está incluído no repositório; rode o notebook apenas se quiser treinar outro modelo. Não envie `.env` ou chaves ao Colab.

O [notebook de treinamento completo](https://colab.research.google.com/github/Danilogggs/gatitos/blob/main/notebooks/CatCare_AI_treinamento_completo.ipynb) continua disponível para quando houver imagens com `labels.csv` rotulado nas colunas `path,breed,features,coat_pattern,colors,coat_length,split`.

### Cabeça local de cores

O Oxford-IIIT Pet não possui rótulos de cor. O script `backend/app/ml/train_oxford_colors.py` cria rótulos aproximados medindo apenas os pixels do gato indicados pela máscara de segmentação, acrescenta exemplos sintéticos balanceados no conjunto de treino e treina uma cabeça multirrótulo sobre o backbone existente. Validação e teste continuam usando apenas fotos originais. Para reproduzir a partir de `backend/`:

```powershell
./.venv/Scripts/python.exe -m app.ml.train_oxford_colors
```

O resultado é `models/catcare-oxford-colors.pt`. Ative-o em `.env.local` com `ML_MODEL_PATH=models/catcare-oxford-colors.pt` e reinicie o backend. Neste treino, o macro-F1 de teste foi 68,6%; por cor: preto 84,2%, branco 70,6%, cinza 87,1%, laranja 34,9%, creme 57,2% e marrom 77,6%. Como os rótulos são aproximados — sobretudo para laranja — mantenha a correção humana e use os registros de `prediction_feedback` para formar futuramente um dataset rotulado de verdade.

### Cabeça local de comprimento do pelo

O script `backend/app/ml/train_oxford_length.py` usa o checkpoint de cores e treina as classes `hairless`, `short`, `medium` e `long` (sem pelo, curto, médio e longo). O Oxford não fornece comprimento por foto: o script atribui rótulos aproximados a partir da raça. Birman e ragdoll entram como médio; maine coon e persa como longo; sphynx como sem pelo; as demais raças Oxford como curto. Para reproduzir a partir de `backend/`:

```powershell
./.venv/Scripts/python.exe -m app.ml.train_oxford_length
```

O resultado é `models/catcare-oxford-colors-length.pt`. Ative-o em `.env.local` com `ML_MODEL_PATH=models/catcare-oxford-colors-length.pt` e reinicie o backend. A acurácia no teste separado foi 95,4% contra esses rótulos aproximados; recall: sem pelo 97,0%, curto 97,8%, médio 96,0% e longo 85,5%. Isso **não valida** o desempenho em SRD nem em gatos cuja pelagem difere do padrão da raça. A revisão humana continua necessária.

### Fotos reais rotuladas e próximo treino

Na revisão administrativa, confira raça, cores e pelo na foto e marque a confirmação antes de publicar. O backend exige essa confirmação. Fotos de gatos publicados com uma revisão registrada podem ser exportadas do Supabase para um diretório local ignorado pelo Git:

```powershell
cd backend
./.venv/Scripts/python.exe -m app.ml.export_reviewed
```

Isso cria `dataset/reviewed/labels.csv` e as fotos correspondentes. O exportador usa os valores **corrigidos** da tabela `cats`, agrupa fotos do mesmo gato e ignora registros não publicados ou sem rótulos válidos. Fotos exportadas podem conter dados de terceiros: não as adicione ao repositório nem as distribua.

Uma fonte externa com foto, cor e comprimento no mesmo registro é o [PetFinder.my Adoption Prediction](https://www.kaggle.com/competitions/petfinder-adoption-prediction/data). O download exige conta Kaggle e aceite das regras; o projeto não contorna essa etapa. Após obter e extrair `train.csv`, `color_labels.csv` e `train_images/` em `backend/dataset/petfinder-raw/`, importe todas as fotos disponíveis dos perfis de **um gato** cujas cores se mapeiam sem ambiguidade à taxonomia local:

```powershell
./.venv/Scripts/python.exe -m app.ml.import_petfinder --input-dir dataset/petfinder-raw
```

Se você baixou o ZIP da competição, pode importá-lo diretamente (incluindo `train` e `test` da competição; a avaliação local faz outra divisão por gato):

```powershell
./.venv/Scripts/python.exe -m app.ml.import_petfinder_zip --archive "C:/Users/danil/Downloads/petfinder-adoption-prediction.zip"
```

Nesta execução foram importadas 16.913 fotos de 4.350 gatos. As fotos e os checkpoints candidatos ficam fora do Git. Respeite as [regras da competição](https://www.kaggle.com/competitions/petfinder-adoption-prediction/rules); não redistribua as fotos e não presuma autorização para uso comercial.

Rótulos `Golden` e `Yellow` ficam de fora, pois não equivalem com segurança a `orange` ou `cream`; essa fonte, sozinha, não resolve o desempenho de laranja. O importador não treina raça nem a classe `hairless`.

O candidato treinado somente com PetFinder melhorou no teste local dessa fonte, mas regrediu em cores no teste Oxford. O candidato combinado com Oxford aproximou os resultados de cor do modelo ativo (macro-F1 68,4% contra 68,6% em Oxford), porém ainda perdeu em laranja. **Nenhum dos dois substituiu o modelo ativo.** Métricas de cor do Oxford usam rótulos calculados por pixels; as de pelo usam rótulos derivados da raça e não substituem anotações humanas por foto.

Um treino focado em laranja/creme (`python -m app.ml.train_orange_cream`) também não foi ativado: no teste Oxford, creme passou de 57,2% para 59,9%, mas laranja caiu de 34,9% para 26,2%. A fila local `dataset/orange-cream-review/review_queue.csv` foi criada com 184 fotos. Destas, 157 receberam revisão visual conservadora em `visually_reviewed.csv`: 42 laranjas, 30 cremes e 85 nem laranja nem creme; 27 permaneceram incertas. `suggested_color` **não é rótulo confirmado**. O comando para reproduzir a fila é:

```powershell
./.venv/Scripts/python.exe -m app.ml.prepare_color_review --petfinder-archive "C:/Users/danil/Downloads/petfinder-adoption-prediction.zip"
```

Para reproduzir as pranchas, a revisão e o treino candidato: `python -m app.ml.color_review_sheets`, `python -m app.ml.record_color_review`, `python -m app.ml.train_reviewed_colors`. A divisão por foto única reservou 31 imagens revisadas para teste: o F1 de laranja passou de 54,5% para 76,2% e o de creme de 44,4% para 66,7%. É uma amostra pequena; no teste Oxford maior, com rótulos *automáticos e aproximados*, ambos regrediram. Foi preparado um teste visual independente antes da ativação experimental.

Uma segunda fila **independente do treino** foi criada com 160 fotos novas (120 PetFinder de outros PetIDs e 40 Oxford do conjunto de teste). A inspeção visual aceitou 152: 54 laranjas, 27 cremes e 71 outras cores; 8 ficaram incertas. O arquivo é `dataset/orange-cream-validation/visually_reviewed_validation.csv`. Com o modelo anterior versus o candidato revisado, o F1 nesse teste visual foi 36,1% → 61,2% para laranja e 37,4% → 46,8% para creme. Preto, branco, cinza, marrom, raça e pelo têm pesos idênticos nos dois checkpoints. Para repetir a comparação: `python -m app.ml.evaluate_reviewed_colors`. O candidato está ativo **somente no ambiente local**, via `.env.local`, para teste com revisão humana. Para reverter, defina `ML_MODEL_PATH=models/catcare-oxford-colors-length.pt` nesse arquivo e reinicie o backend. Creme ainda tem falsos positivos: a precisão no teste visual foi 36%, então não publique previsões automaticamente.

Com pelo menos 50 gatos distintos rotulados, treine um **candidato** e compare-o à versão ativa em gatos reservados para teste. Por padrão, o treino usa todos os gatos válidos e todas as suas fotos; a separação é por gato, não por foto:

```powershell
./.venv/Scripts/python.exe -m app.ml.train_labeled --manifest dataset/petfinder/labels.csv --manifest dataset/reviewed/labels.csv
```

O resultado é `models/catcare-labeled-candidate.pt`. O script não altera `.env.local` nem ativa o candidato. Analise as métricas por cor e pelo e teste em fotos reais adicionais antes de trocar o checkpoint. O backend mantém o modelo carregado em memória entre requisições, então o reinício é necessário após qualquer troca de checkpoint.

## Segurança e fluxo

As chaves secretas Supabase e Gemini ficam em `.env.local`, fora do Git. Quem obtiver a service role pode acessar o banco e o Storage sem as restrições de RLS. A API pública só consulta `PUBLISHED` e não retorna previsões nem confiança. Rotas administrativas exigem JWT válido do Supabase Auth e registro em `admin_users`. Todo envio, inclusive de administrador, entra em `PENDING_REVIEW`.

Antes de colocar em produção, configure limites de requisição no gateway, proteção contra spam para uploads públicos e monitore uso do Storage. Não há processo real de adoção: o botão é apenas visual. A busca de semelhantes tem estrutura de embeddings, mas só poderá funcionar após treinamento e indexação.

## Verificação manual sugerida

Após configurar as credenciais, confira cadastro com várias fotos, rejeição de arquivo inválido, revisão e edição, publicação, filtros públicos, login administrativo, correção registrada em `prediction_feedback` e ausência de dados internos nas respostas públicas.

### Evidências de avaliação PjBL2

O [relatório de classificação de raças](reports/pjbl2/relatorio.md) apresenta matriz de
confusão, precisão, recall e F1 por raça, médias macro e ponderada e acurácia. A avaliação
usa 240 fotos originais do split `test` Oxford (20 por cada uma das 12 raças). Os CSVs da
matriz, métricas e previsões por imagem estão na mesma pasta, sem redistribuir as fotos.
O cálculo é reproduzível por `python -m app.ml.report_breed_metrics` a partir de `backend/`.
O relatório distingue a classificação top-1 das 12 classes treinadas da rejeição “SRD”
por baixa confiança exibida pelo aplicativo.

### Ranking e revisão de raça (migração de 07/10/2026)

Em um Supabase já existente, **antes de reiniciar o backend atualizado**, execute o conteúdo de
`supabase/migrations/20261007_review_workflow.sql` no SQL Editor do projeto. O arquivo
`supabase/schema.sql` já inclui as mesmas estruturas para um banco novo. Sem essa migração,
as rotas de cadastro, configuração e revisão não funcionarão com a nova versão.

No painel administrativo, o botão “Ativar/Desativar prévia pública” controla toda a prévia
da IA: quando desativada, o usuário envia o cadastro diretamente para revisão, sem ver raça,
cores, comprimento do pelo, descrição ou ranking. A análise ainda é executada no envio e o
administrador continua vendo as pontuações completas da previsão original.
Cadastros anteriores à migração não possuem ranking salvo e mostram “ranking indisponível”.
Revisões de raça são registradas em `cat_review_events`, inclusive mudanças após publicação;
`prediction_feedback` reflete o último rótulo salvo. SRD mantido como SRD não conta como
acerto de raça, pois SRD não é uma classe treinada.

O campo “Raça com procedência suficiente para uso em treino futuro” é independente da raça
exibida no anúncio e exige uma nota de procedência. Nada é retreinado automaticamente. Para
exportar somente gatos publicados e marcados como aptos para treino de raça:

```powershell
cd backend
./.venv/Scripts/python.exe -m app.ml.export_verified_breeds
```

O manifesto local resultante é `backend/dataset/verified-breeds/labels.csv` (ignorado pelo Git).
O exportador não treina nem ativa um modelo; mantenha um conjunto de teste separado por gato.
