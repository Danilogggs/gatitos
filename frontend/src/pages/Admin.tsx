import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { CatCard } from '../components/CatCard'
import { api, auth } from '../services/api'
import type { Cat, Config, Image, Status } from '../types'

const tabs: [Status,string][] = [['PENDING_REVIEW','Aguardando revisão'],['PUBLISHED','Publicados'],['ADOPTED','Adotados'],['ARCHIVED','Arquivados'],['REJECTED','Rejeitados']]

export function AdminLogin() {
  const navigate = useNavigate()
  const [email,setEmail] = useState('')
  const [password,setPassword] = useState('')
  const [error,setError] = useState('')
  const login = async (event: React.FormEvent) => { event.preventDefault(); if (!auth) {setError('Configure as variáveis VITE_SUPABASE no frontend.');return} const {error} = await auth.auth.signInWithPassword({email,password}); if (error) setError('Não foi possível entrar. Confira suas credenciais.'); else navigate('/admin/dashboard') }
  return <div className="container page-pad login-page"><div className="form-shell"><span className="eyebrow">ÁREA ADMINISTRATIVA</span><h1>Bem-vindo de volta</h1><p>Entre para revisar os cadastros e acompanhar os gatinhos.</p><form onSubmit={login}><label className="field">E-mail<input type="email" required value={email} onChange={e => setEmail(e.target.value)}/></label><label className="field">Senha<input type="password" required value={password} onChange={e => setPassword(e.target.value)}/></label>{error && <p className="error">{error}</p>}<button className="button primary" type="submit">Entrar →</button></form></div></div>
}

export function AdminDashboard() {
  const [status,setStatus] = useState<Status>('PENDING_REVIEW')
  const [cats,setCats] = useState<Cat[]>([])
  const [stats,setStats] = useState<Record<string,number> | null>(null)
  const [error,setError] = useState('')
  const navigate = useNavigate()
  useEffect(() => {api.adminCats(status).then(setCats).catch(err => setError(err.message))},[status])
  useEffect(() => {api.stats().then(setStats).catch(() => {})},[])
  const logout = async () => {await auth?.auth.signOut(); navigate('/admin')}
  return <div className="container page-pad admin-page"><div className="admin-heading"><div><span className="eyebrow">PAINEL ADMINISTRATIVO</span><h1>Cuidando de cada história</h1><p>Revise as informações antes que os gatos apareçam ao público.</p></div><div className="button-row"><Link className="button light" to="/cadastrar">+ Cadastrar gato</Link><button className="button ghost" onClick={logout}>Sair</button></div></div>{stats && <div className="stats-grid"><div><strong>{stats.total_predictions}</strong><span>Previsões</span></div><div><strong>{stats.total_reviews}</strong><span>Revisões</span></div><div><strong>{stats.breed_correct}</strong><span>Acertos de raça</span></div><div><strong>{stats.breed_errors}</strong><span>Correções de raça</span></div></div>}<div className="tab-row">{tabs.map(([value,label]) => <button key={value} className={status === value ? 'active' : ''} onClick={() => setStatus(value)}>{label}</button>)}</div>{error && <p className="error">{error}</p>}{cats.length ? <div className="admin-list">{cats.map(cat => <div className="admin-item" key={cat.id}><div className="admin-photo"><CatCard cat={cat}/></div><div><span className="eyebrow">{cat.status.replaceAll('_',' ')}</span><h3>{cat.name}</h3><p>Enviado em {new Date(cat.created_at).toLocaleDateString('pt-BR')} · {cat.created_by || 'Cadastro público'}</p><p>IA: {cat.prediction?.predicted_breed || 'sem previsão'} · {cat.prediction?.predicted_features?.join(', ') || 'sem características'}</p></div><Link className="button primary small" to={`/admin/revisar/${cat.id}`}>Revisar →</Link></div>)}</div> : <div className="empty-state"><span>🐾</span><h3>Nenhum cadastro nesta categoria</h3></div>}</div>
}

const textFields: [keyof Cat,string][] = [['name','Nome'],['breed','Raça'],['coat_pattern','Padrão técnico'],['coat_length','Comprimento do pelo'],['primary_color','Cor principal'],['secondary_color','Cor secundária'],['sex','Sexo'],['approximate_age','Idade aproximada'],['size','Porte'],['behavior','Comportamento informado'],['health_information','Informações de saúde'],['location','Localização'],['additional_notes','Observações'],['description','Descrição']]

export function AdminReview() {
  const {id = ''} = useParams()
  const navigate = useNavigate()
  const [cat,setCat] = useState<Cat | null>(null)
  const [draft,setDraft] = useState<Cat | null>(null)
  const [config,setConfig] = useState<Config | null>(null)
  const [error,setError] = useState('')
  const [message,setMessage] = useState('')
  const [busy,setBusy] = useState(false)
  const [large,setLarge] = useState<string | null>(null)
  useEffect(() => {api.adminCat(id).then(data => {setCat(data);setDraft(data)}).catch(err => setError(err.message));api.config().then(setConfig).catch(() => {})},[id])
  const refresh = (data: Cat) => {setCat(data);setDraft(data)}
  const save = async (): Promise<boolean> => {
    if (!draft) return false
    setBusy(true);setError('');setMessage('')
    try {
      const payload: Record<string,unknown> = {features:draft.features}
      textFields.forEach(([field]) => {payload[field] = draft[field] || null})
      payload.name = draft.name
      refresh(await api.edit(id,payload));setMessage('Alterações salvas.');return true
    } catch (err) {setError((err as Error).message);return false} finally {setBusy(false)}
  }
  const action = async (name: string) => { if (!(await save())) return; setBusy(true); try {refresh(await api.action(id,name));navigate('/admin/dashboard')} catch (err) {setError((err as Error).message)} finally {setBusy(false)} }
  const imageAction = async (call: () => Promise<Cat>) => {setBusy(true);setError('');try {refresh(await call())} catch(err){setError((err as Error).message)} finally{setBusy(false)}}
  const move = (image: Image, direction: number) => {if(!draft)return;const images = [...draft.images].sort((a,b)=>a.position-b.position);const at=images.findIndex(item=>item.id===image.id);const next=at+direction;if(next<0||next>=images.length)return;[images[at],images[next]]=[images[next],images[at]];imageAction(()=>api.reorder(id,images.map(item=>item.id)))}
  if (!cat || !draft) return <div className="container page-pad">{error || 'Carregando cadastro...'}</div>
  return <div className="container page-pad review-page"><Link className="back-link" to="/admin/dashboard">← Voltar ao painel</Link><div className="admin-heading"><div><span className="eyebrow">REVISÃO ADMINISTRATIVA · {cat.status.replaceAll('_',' ')}</span><h1>Revisar {cat.name}</h1><p>Confira as fotos e corrija todas as informações antes de publicar.</p></div></div>{cat.prediction && <div className="prediction-box"><span className="eyebrow">✦ PREVISÃO ORIGINAL · {cat.prediction.model_version}</span><div className="prediction-grid"><div><small>Raça provável</small><strong>{cat.prediction.predicted_breed || 'Sem previsão'}</strong><small>{cat.prediction.breed_confidence != null ? `${Math.round(cat.prediction.breed_confidence*100)}% de confiança` : 'Sem confiança'}</small></div><div><small>Características</small><strong>{cat.prediction.predicted_features?.join(' · ') || 'Sem previsão'}</strong></div><div><small>Padrão técnico</small><strong>{cat.prediction.predicted_coat_pattern || 'Sem previsão'}</strong><small>{cat.prediction.coat_confidence != null ? `${Math.round(cat.prediction.coat_confidence*100)}% de confiança` : ''}</small></div></div></div>}<div className="review-grid"><section className="form-shell"><div className="step-marker"><span>01</span> Fotografias</div><div className="review-photos">{[...cat.images].sort((a,b)=>a.position-b.position).map(image => <div className="review-photo" key={image.id}><button className="photo-open" onClick={()=>setLarge(image.url)} title="Ampliar foto"><img src={image.url} alt={`Foto de ${cat.name}`}/></button><small>{image.is_primary ? '★ Foto principal' : `Foto ${image.position+1}`}</small><div className="mini-actions"><button disabled={busy || image.is_primary} onClick={()=>imageAction(()=>api.primaryImage(id,image.id))}>Principal</button><button disabled={busy} onClick={()=>move(image,-1)}>←</button><button disabled={busy} onClick={()=>move(image,1)}>→</button><button disabled={busy || cat.images.length<=1} onClick={()=>imageAction(()=>api.deleteImage(id,image.id))}>Remover</button></div></div>)}</div><label className="field">Adicionar fotos<input type="file" multiple accept="image/jpeg,image/png,image/webp" disabled={busy} onChange={e=>{const files=Array.from(e.target.files||[]);if(files.length)imageAction(()=>api.addImages(id,files))}}/></label></section><section className="form-shell"><div className="step-marker"><span>02</span> Dados do gato</div><div className="form-grid">{textFields.map(([key,label])=><label className="field" key={key}>{label}{['behavior','health_information','additional_notes','description'].includes(key)?<textarea value={String(draft[key]||'')} onChange={e=>setDraft({...draft,[key]:e.target.value})}/>:<input value={String(draft[key]||'')} onChange={e=>setDraft({...draft,[key]:e.target.value})}/>}</label>)}</div><div className="field">Características<div className="feature-checks">{Object.entries(config?.features||{}).map(([key,description])=><label title={description} key={key}><input type="checkbox" checked={draft.features.includes(key)} onChange={e=>setDraft({...draft,features:e.target.checked?[...draft.features,key]:draft.features.filter(item=>item!==key)})}/>{key.replaceAll('_',' ')}</label>)}</div></div></section></div>{error && <p className="error">{error}</p>}{message && <p className="notice">{message}</p>}<div className="review-actions"><button className="button light" disabled={busy} onClick={save}>Salvar alterações</button>{cat.status==='PENDING_REVIEW'&&<><button className="button danger" disabled={busy} onClick={()=>action('reject')}>Rejeitar</button><button className="button primary" disabled={busy} onClick={()=>action('publish')}>Aprovar e publicar →</button></>}{cat.status==='PUBLISHED'&&<><button className="button light" disabled={busy} onClick={()=>action('archive')}>Arquivar</button><button className="button primary" disabled={busy} onClick={()=>action('adopt')}>Marcar como adotado</button></>}{cat.status==='ADOPTED'&&<button className="button light" disabled={busy} onClick={()=>action('archive')}>Arquivar</button>}</div>{large&&<div className="lightbox" onClick={()=>setLarge(null)} role="presentation"><button aria-label="Fechar foto">×</button><img src={large} alt="Foto ampliada"/></div>}</div>
}

