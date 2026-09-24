# CatCare AI

Plataforma de cadastro e divulgação de gatos para adoção com revisão humana obrigatória. O cadastro exige nome e ao menos uma foto. Fotos são armazenadas no Supabase Storage; dados ficam no PostgreSQL. O classificador de imagens sugere características visuais e o Gemini pode escrever a descrição. O administrador confere e corrige antes da publicação.

## Estrutura

- `frontend/`: React, TypeScript e Vite.
- `backend/`: FastAPI, integração com Supabase, serviços ML e Gemini.
- `supabase/schema.sql`: tabelas, índices, RLS e bucket.

## Configuração

1. Crie um projeto Supabase e execute `supabase/schema.sql` no SQL Editor.
2. Crie um usuário em Authentication e adicione seu UUID a `public.admin_users`: `insert into public.admin_users(user_id) values ('UUID_DO_USUARIO');`.
3. Copie `.env.example` para `.env` na raiz, preencha as variáveis do backend e mantenha o arquivo fora do Git.
4. Crie `frontend/.env` com `VITE_API_URL`, `VITE_SUPABASE_URL` e `VITE_SUPABASE_ANON_KEY`. São credenciais públicas do cliente; a service role e a chave Gemini ficam apenas no backend.
5. Para rodar futuramente: instale `backend/requirements.txt` em ambiente Python isolado e rode `uvicorn app.main:app` a partir de `backend/`; em `frontend/`, instale as dependências do `package.json` e execute `npm run dev`.

Nenhuma dependência foi instalada, e o aplicativo, build e testes não foram executados conforme solicitado.

## ML e LLM

`ML_MODE=mock` deixa explícito na interface que o modelo não está treinado e **não cria palpites falsos**. Para `ML_MODE=real`, o checkpoint deve conter `backbone`, `breed_head`, `breeds` e `version`. As cabeças opcionais `feature_head`, `pattern_head`, `color_head` e `length_head` só produzem previsões quando estiverem treinadas e listadas em `trained_heads`. O checkpoint gerado com Oxford treina apenas raça; portanto características, padrão, cores e comprimento ficam sem previsão. O dataset e os pesos não entram no repositório nem no Storage da aplicação.

Como o Oxford não contém exemplos de SRD, o backend usa `srd` como resultado provisório quando a maior probabilidade entre as 12 raças é menor que `ML_BREED_MIN_CONFIDENCE` (padrão: `0.85`). O valor pode ser ajustado no `.env`: aumentá-lo manda mais gatos para SRD e pode rejeitar raças reais; diminuí-lo aceita mais raças e pode classificar um SRD como uma delas. Uma probabilidade alta **não prova** que o gato é de raça pura nem garante a identificação de todo SRD. A revisão do administrador é obrigatória. O campo `breed_confidence` fica vazio no fallback SRD porque o modelo não foi treinado para medir confiança nessa classe.

O Gemini usa `GEMINI_API_KEY` e `GEMINI_MODEL` do ambiente. Se não estiver configurado ou falhar, uma descrição factual local é usada. Comportamento somente entra no texto se uma pessoa o informou.

## Treinamento no Colab

Abra o [notebook de treinamento com Oxford-IIIT Pet](https://colab.research.google.com/github/Danilogggs/gatitos/blob/main/notebooks/CatCare_AI_treinamento.ipynb), selecione uma GPU e execute as células em ordem. Ele baixa o dataset automaticamente, seleciona as 12 raças de gatos, treina o EfficientNet-B0 e oferece o download de `catcare-oxford.pt`. Você não precisa enviar fotos nem `labels.csv`. O Oxford não inclui uma classe SRD nem rótulos das características de pelagem do aplicativo; o fallback SRD ocorre no backend após o treino, conforme o limiar acima. Após o download, coloque o checkpoint no computador do backend e configure `ML_MODEL_PATH` e `ML_MODE=real` no `.env` local. Não envie `.env` ou chaves ao Colab.

O [notebook de treinamento completo](https://colab.research.google.com/github/Danilogggs/gatitos/blob/main/notebooks/CatCare_AI_treinamento_completo.ipynb) continua disponível para quando houver imagens com `labels.csv` rotulado nas colunas `path,breed,features,coat_pattern,colors,coat_length,split`.

## Segurança e fluxo

Apenas a API usa a service role do Supabase. As tabelas têm RLS sem acesso direto pelo cliente. A API pública só consulta `PUBLISHED` e não retorna previsões nem confiança. Rotas administrativas exigem JWT válido do Supabase Auth e registro em `admin_users`. Todo envio, inclusive de administrador, entra em `PENDING_REVIEW`.

Antes de colocar em produção, configure limites de requisição no gateway, proteção contra spam para uploads públicos e monitore uso do Storage. Não há processo real de adoção: o botão é apenas visual. A busca de semelhantes tem estrutura de embeddings, mas só poderá funcionar após treinamento e indexação.

## Verificação manual sugerida

Após configurar as credenciais, confira cadastro com várias fotos, rejeição de arquivo inválido, revisão e edição, publicação, filtros públicos, login administrativo, correção registrada em `prediction_feedback` e ausência de dados internos nas respostas públicas.
