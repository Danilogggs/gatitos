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

`ML_MODE=mock` deixa explícito na interface que o modelo não está treinado e **não cria palpites falsos**. Para `ML_MODE=real`, o checkpoint deve conter `backbone`, `breed_head`, `breeds` e `version`. As cabeças opcionais `feature_head`, `pattern_head`, `color_head` e `length_head` só produzem previsões quando estiverem treinadas e listadas em `trained_heads`. O checkpoint gerado com Oxford treina apenas raça; portanto características, padrão, cores e comprimento ficam sem previsão. O checkpoint está em `backend/models/`; o dataset não entra no repositório.

Como o Oxford não contém exemplos de SRD, o backend registra a raça com maior pontuação entre as 12 classes quando ela é **maior que** `ML_BREED_MIN_CONFIDENCE` (padrão: `0.40`). Se a maior pontuação for 40% ou menos, usa `srd` como resultado provisório. Com várias fotos, as pontuações são calculadas a partir da média das representações visuais. Aumentar o limite manda mais gatos para SRD e pode rejeitar raças reais; diminuí-lo aceita mais raças e pode classificar um SRD como uma delas. Uma pontuação alta **não prova** que o gato é de raça pura nem garante a identificação de todo SRD. A revisão do administrador é obrigatória. O campo `breed_confidence` fica vazio no fallback SRD porque o modelo não foi treinado para medir confiança nessa classe.

A prévia do cadastro mostra as pontuações das 12 raças treinadas em ordem decrescente. Esses valores são relativos às classes do checkpoint e não representam a probabilidade de pedigree; SRD não recebe pontuação porque não foi treinado como classe.

O Gemini usa `GEMINI_API_KEY` e `GEMINI_MODEL` do ambiente. Se não estiver configurado ou falhar, uma descrição factual local é usada. Comportamento somente entra no texto se uma pessoa o informou.

## Treinamento no Colab

Abra o [notebook de treinamento com Oxford-IIIT Pet](https://colab.research.google.com/github/Danilogggs/gatitos/blob/main/notebooks/CatCare_AI_treinamento.ipynb), selecione uma GPU e execute as células em ordem. Ele baixa o dataset automaticamente, seleciona as 12 raças de gatos, treina o EfficientNet-B0 e oferece o download de `catcare-oxford.pt`. Você não precisa enviar fotos nem `labels.csv`. O Oxford não inclui uma classe SRD nem rótulos das características de pelagem do aplicativo; o fallback SRD ocorre no backend após o treino, conforme o limiar acima. O checkpoint deste teste já está incluído no repositório; rode o notebook apenas se quiser treinar outro modelo. Não envie `.env` ou chaves ao Colab.

O [notebook de treinamento completo](https://colab.research.google.com/github/Danilogggs/gatitos/blob/main/notebooks/CatCare_AI_treinamento_completo.ipynb) continua disponível para quando houver imagens com `labels.csv` rotulado nas colunas `path,breed,features,coat_pattern,colors,coat_length,split`.

## Segurança e fluxo

As chaves secretas Supabase e Gemini ficam em `.env.local`, fora do Git. Quem obtiver a service role pode acessar o banco e o Storage sem as restrições de RLS. A API pública só consulta `PUBLISHED` e não retorna previsões nem confiança. Rotas administrativas exigem JWT válido do Supabase Auth e registro em `admin_users`. Todo envio, inclusive de administrador, entra em `PENDING_REVIEW`.

Antes de colocar em produção, configure limites de requisição no gateway, proteção contra spam para uploads públicos e monitore uso do Storage. Não há processo real de adoção: o botão é apenas visual. A busca de semelhantes tem estrutura de embeddings, mas só poderá funcionar após treinamento e indexação.

## Verificação manual sugerida

Após configurar as credenciais, confira cadastro com várias fotos, rejeição de arquivo inválido, revisão e edição, publicação, filtros públicos, login administrativo, correção registrada em `prediction_feedback` e ausência de dados internos nas respostas públicas.
