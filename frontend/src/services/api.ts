import { createClient } from '@supabase/supabase-js'
import type { Cat, Config, Prediction, Status } from '../types'

const base = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || ''
const supabaseAnon = import.meta.env.VITE_SUPABASE_ANON_KEY || ''
export const auth = supabaseUrl && supabaseAnon ? createClient(supabaseUrl, supabaseAnon) : null

async function request<T>(path: string, options: RequestInit = {}, admin = false): Promise<T> {
  const headers = new Headers(options.headers)
  if (admin) {
    const session = (await auth?.auth.getSession())?.data.session
    if (!session) throw new Error('Faça login para continuar.')
    headers.set('Authorization', `Bearer ${session.access_token}`)
  }
  if (options.body && !(options.body instanceof FormData)) headers.set('Content-Type', 'application/json')
  const response = await fetch(`${base}/api${path}`, { ...options, headers })
  if (!response.ok) {
    const data = await response.json().catch(() => ({}))
    throw new Error(typeof data.detail === 'string' ? data.detail : `Erro ${response.status}`)
  }
  return response.json() as Promise<T>
}

export const api = {
  config: () => request<Config>('/config'),
  cats: (query = '') => request<Cat[]>(`/cats${query ? `?${query}` : ''}`),
  cat: (id: string) => request<Cat>(`/cats/${id}`),
  predict: (name: string, behavior: string, files: File[]) => { const form = new FormData(); form.append('name', name); form.append('behavior', behavior); files.forEach(file => form.append('files', file)); return request<Prediction>('/ml/predict', { method: 'POST', body: form }) },
  create: (name: string, files: File[], optional: Record<string,string>) => { const form = new FormData(); form.append('name', name); form.append('optional', JSON.stringify(optional)); files.forEach(file => form.append('files', file)); return request<{id:string;status:Status;ml_mode:string}>('/cats', { method: 'POST', body: form }) },
  adminCats: (status: Status) => request<Cat[]>(`/admin/cats?status=${status}`, {}, true),
  adminCat: (id: string) => request<Cat>(`/admin/cats/${id}`, {}, true),
  edit: (id: string, cat: Record<string,unknown>) => request<Cat>(`/cats/${id}`, { method: 'PUT', body: JSON.stringify(cat) }, true),
  action: (id: string, action: string) => request<Cat>(`/cats/${id}/${action}`, { method: 'POST' }, true),
  addImages: (id: string, files: File[]) => { const form = new FormData(); files.forEach(file => form.append('files', file)); return request<Cat>(`/cats/${id}/images`, { method: 'POST', body: form }, true) },
  deleteImage: (id: string, imageId: string) => request<Cat>(`/cats/${id}/images/${imageId}`, { method: 'DELETE' }, true),
  primaryImage: (id: string, imageId: string) => request<Cat>(`/cats/${id}/images/${imageId}/primary`, { method: 'PUT' }, true),
  reorder: (id: string, imageIds: string[]) => request<Cat>(`/cats/${id}/images/order`, { method: 'PUT', body: JSON.stringify({image_ids: imageIds}) }, true),
  stats: () => request<Record<string,number>>('/admin/ml/stats', {}, true)
}
