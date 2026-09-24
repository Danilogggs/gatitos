-- Executar no SQL Editor de um projeto Supabase novo.
create extension if not exists pgcrypto;
create extension if not exists vector;

create table if not exists public.admin_users (
  user_id uuid primary key references auth.users(id) on delete cascade,
  created_at timestamptz not null default now()
);

create table if not exists public.cats (
  id uuid primary key default gen_random_uuid(),
  name text not null check (length(trim(name)) between 1 and 100),
  sex text, approximate_age text, size text, breed text, coat_pattern text,
  coat_length text, primary_color text, secondary_color text,
  behavior text, health_information text, location text, additional_notes text,
  description text,
  status text not null default 'DRAFT' check (status in ('DRAFT','PENDING_REVIEW','PUBLISHED','REJECTED','ADOPTED','ARCHIVED')),
  created_by uuid references auth.users(id) on delete set null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  published_at timestamptz, adopted_at timestamptz
);

create table if not exists public.cat_images (
  id uuid primary key default gen_random_uuid(),
  cat_id uuid not null references public.cats(id) on delete cascade,
  storage_path text not null unique, mime_type text not null, file_size integer not null,
  is_primary boolean not null default false, position integer not null default 0,
  uploaded_by uuid references auth.users(id) on delete set null,
  created_at timestamptz not null default now()
);
create unique index if not exists cat_one_primary on public.cat_images(cat_id) where is_primary;

create table if not exists public.cat_features (
  id uuid primary key default gen_random_uuid(),
  cat_id uuid not null references public.cats(id) on delete cascade,
  feature text not null,
  created_at timestamptz not null default now(),
  unique (cat_id, feature)
);

create table if not exists public.ml_predictions (
  id uuid primary key default gen_random_uuid(),
  cat_id uuid not null references public.cats(id) on delete cascade,
  predicted_breed text, breed_confidence real,
  predicted_features text[] not null default '{}',
  predicted_coat_pattern text, coat_confidence real,
  predicted_colors text[] not null default '{}', predicted_coat_length text,
  model_version text not null,
  created_at timestamptz not null default now()
);

create table if not exists public.prediction_feedback (
  id uuid primary key default gen_random_uuid(),
  prediction_id uuid not null unique references public.ml_predictions(id) on delete cascade,
  cat_id uuid not null references public.cats(id) on delete cascade,
  predicted_breed text, corrected_breed text, breed_correct boolean,
  predicted_coat_pattern text, corrected_coat_pattern text, coat_correct boolean,
  corrected_by uuid references auth.users(id) on delete set null,
  created_at timestamptz not null default now()
);

create table if not exists public.embeddings (
  id uuid primary key default gen_random_uuid(),
  cat_id uuid not null references public.cats(id) on delete cascade,
  image_id uuid references public.cat_images(id) on delete cascade,
  embedding vector(1280) not null,
  model_version text not null,
  created_at timestamptz not null default now()
);

create index if not exists cats_status_published on public.cats(status, published_at desc);
create index if not exists cat_features_lookup on public.cat_features(feature, cat_id);
create index if not exists cat_images_position on public.cat_images(cat_id, position);

create or replace function public.set_updated_at() returns trigger language plpgsql as $$
begin new.updated_at = now(); return new; end; $$;
drop trigger if exists cats_updated_at on public.cats;
create trigger cats_updated_at before update on public.cats for each row execute function public.set_updated_at();

alter table public.admin_users enable row level security;
alter table public.cats enable row level security;
alter table public.cat_images enable row level security;
alter table public.cat_features enable row level security;
alter table public.ml_predictions enable row level security;
alter table public.prediction_feedback enable row level security;
alter table public.embeddings enable row level security;
-- Sem policies nas tabelas: acesso aos dados somente pela API com service role.

insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values ('cat-images', 'cat-images', true, 5242880, array['image/webp'])
on conflict (id) do update set public = true, file_size_limit = 5242880,
  allowed_mime_types = array['image/webp'];
-- Bucket público permite mostrar fotos. A API controla gravação e exclusão.

