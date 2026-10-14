-- Aplicar uma vez no SQL Editor do projeto Supabase já existente.
-- Preserva cadastros e previsões existentes.
alter table public.cats
  add column if not exists breed_training_verified boolean not null default false,
  add column if not exists breed_verification_note text;

alter table public.ml_predictions
  add column if not exists breed_scores jsonb not null default '{}'::jsonb;

create table if not exists public.cat_review_events (
  id uuid primary key default gen_random_uuid(),
  cat_id uuid not null references public.cats(id) on delete cascade,
  prediction_id uuid references public.ml_predictions(id) on delete set null,
  event_type text not null check (event_type in ('edit','publish','reject')),
  previous_breed text, reviewed_breed text,
  breed_training_verified boolean not null default false,
  breed_verification_note text,
  reviewed_by uuid references auth.users(id) on delete set null,
  created_at timestamptz not null default now()
);

create table if not exists public.app_settings (
  id boolean primary key default true check (id),
  show_public_breed_ranking boolean not null default true
);
insert into public.app_settings (id) values (true) on conflict (id) do nothing;

alter table public.cat_review_events enable row level security;
alter table public.app_settings enable row level security;
