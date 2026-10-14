import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { CatCard } from '../components/CatCard'
import { BreedScores } from '../components/BreedScores'
import { api, auth } from '../services/api'
import type { Cat, Image, Status } from '../types'
import { breedLabel, breedLabels, colorLabel, colorLabels } from '../utils/catLabels'
import { coatLengthLabel, coatLengthLabels } from '../utils/coatLength'

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
  const [showPublicRanking,setShowPublicRanking] = useState<boolean | null>(null)
  const [savingRanking,setSavingRanking] = useState(false)
  const navigate = useNavigate()
  useEffect(() => {api.adminCats(status).then(setCats).catch(err => setError(err.message))},[status])
  useEffect(() => {api.stats().then(setStats).catch(() => {})},[])
  useEffect(() => {api.config().then(config => setShowPublicRanking(config.show_public_breed_ranking)).catch(err => setError(err.message))},[])
  const toggleRanking = async () => {
    if (showPublicRanking === null) return
    setSavingRanking(true);setError('')
    try { setShowPublicRanking((await api.setRankingVisibility(!showPublicRanking)).show_public_breed_ranking) }
    catch (err) { setError((err as Error).message) }
    finally { setSavingRanking(false) }
  }
  const logout = async () => {await auth?.auth.signOut(); navigate('/admin')}
  const priority = (cat: Cat) => {
    if (cat.prediction?.predicted_breed === 'srd' || cat.prediction?.breed_confidence == null) return 0
    const ranked = Object.values(cat.prediction.breed_scores || {}).sort((a,b) => b-a)
    return ranked.length > 1 && ranked[0] - ranked[1] < 0.15 ? 0 : 1
  }
  const orderedCats = status === 'PENDING_REVIEW' ? [...cats].sort((a,b) => priority(a) - priority(b)) : cats
  return <div className="container page-pad admin-page">
    <div className="admin-heading"><div><span className="eyebrow">PAINEL ADMINISTRATIVO</span><h1>Cuidando de cada história</h1><p>Revise as informações antes que os gatos apareçam ao público.</p></div><div className="button-row"><Link className="button light" to="/cadastrar">+ Cadastrar gato</Link><button className="button ghost" onClick={logout}>Sair</button></div></div>
    <div className="form-shell"><strong>Prévia da IA no cadastro</strong><p className="muted">Quando desativada, o usuário envia o cadastro direto para revisão, sem ver raça, cores, pelo, descrição ou ranking. O administrador continua vendo a análise completa.</p><button className="button light" disabled={savingRanking || showPublicRanking === null} onClick={toggleRanking}>{showPublicRanking === null ? 'Carregando…' : showPublicRanking ? 'Desativar prévia pública' : 'Ativar prévia pública'}</button><p className="muted">Estado atual: {showPublicRanking === null ? 'indisponível' : showPublicRanking ? 'prévia visível' : 'envio direto para revisão'}</p></div>
    {stats && <div className="stats-grid"><div><strong>{stats.total_predictions}</strong><span>Previsões</span></div><div><strong>{stats.total_reviews}</strong><span>Revisões</span></div><div><strong>{stats.breed_correct}</strong><span>Acertos de raça</span></div><div><strong>{stats.breed_errors}</strong><span>Correções de raça</span></div></div>}
    <div className="tab-row">{tabs.map(([value,label]) => <button key={value} className={status === value ? 'active' : ''} onClick={() => setStatus(value)}>{label}</button>)}</div>
    {status === 'PENDING_REVIEW' && <p className="muted">Casos de baixa confiança ou pontuações próximas entre raças aparecem primeiro.</p>}
    {error && <p className="error">{error}</p>}
    {orderedCats.length ? <div className="admin-list">{orderedCats.map(cat => <div className="admin-item" key={cat.id}><div className="admin-photo"><CatCard cat={cat}/></div><div><span className="eyebrow">{cat.status.replaceAll('_',' ')}</span><h3>{cat.name}</h3><p>Enviado em {new Date(cat.created_at).toLocaleDateString('pt-BR')} · {cat.created_by || 'Cadastro público'}</p><p>IA: {breedLabel(cat.prediction?.predicted_breed || cat.breed) || 'sem previsão'}{priority(cat) === 0 ? ' · revisar prioridade' : ''}</p></div><Link className="button primary small" to={`/admin/revisar/${cat.id}`}>Revisar →</Link></div>)}</div> : <div className="empty-state"><span>🐾</span><h3>Nenhum cadastro nesta categoria</h3></div>}
  </div>
}

const textFields: [keyof Cat,string][] = [['name','Nome'],['breed','Raça'],['coat_length','Comprimento do pelo'],['primary_color','Cor principal'],['secondary_color','Cor secundária'],['sex','Sexo'],['approximate_age','Idade aproximada'],['size','Porte'],['behavior','Comportamento informado'],['health_information','Informações de saúde'],['location','Localização'],['additional_notes','Observações'],['description','Descrição']]

export function AdminReview() {
  const {id = ''} = useParams()
  const navigate = useNavigate()
  const [cat,setCat] = useState<Cat | null>(null)
  const [draft,setDraft] = useState<Cat | null>(null)
  const [error,setError] = useState('')
  const [message,setMessage] = useState('')
  const [busy,setBusy] = useState(false)
  const [confirmedVisual,setConfirmedVisual] = useState(false)
  const [large,setLarge] = useState<string | null>(null)
  useEffect(() => {api.adminCat(id).then(data => {setCat(data);setDraft(data);setConfirmedVisual(false)}).catch(err => setError(err.message))},[id])
  const refresh = (data: Cat) => {setCat(previous => ({...data, review_events: data.review_events ?? previous?.review_events}));setDraft(data)}
  const save = async (regenerateDescription = false): Promise<boolean> => {
    if (!draft) return false
    setBusy(true);setError('');setMessage('')
    try {
      const payload: Record<string,unknown> = {features:draft.features,coat_pattern:draft.coat_pattern,regenerate_description:regenerateDescription,
        breed_training_verified: Boolean(draft.breed_training_verified), breed_verification_note: draft.breed_verification_note || null}
      textFields.forEach(([field]) => {payload[field] = draft[field] || null})
      payload.name = draft.name
      await api.edit(id,payload);refresh(await api.adminCat(id));setMessage(regenerateDescription ? 'Descrição atualizada com os dados revisados.' : 'Alterações salvas.');return true
    } catch (err) {setError((err as Error).message);return false} finally {setBusy(false)}
  }
  const action = async (name: string) => { if (name === 'publish' && !confirmedVisual) {setError('Confirme os rótulos visuais antes de publicar.');return} if (!(await save())) return; setBusy(true); try {refresh(await api.action(id,name,confirmedVisual));navigate('/admin/dashboard')} catch (err) {setError((err as Error).message)} finally {setBusy(false)} }
  const imageAction = async (call: () => Promise<Cat>) => {setBusy(true);setError('');setConfirmedVisual(false);try {refresh(await call())} catch(err){setError((err as Error).message)} finally{setBusy(false)}}
  const move = (image: Image, direction: number) => {if(!draft)return;const images = [...draft.images].sort((a,b)=>a.position-b.position);const at=images.findIndex(item=>item.id===image.id);const next=at+direction;if(next<0||next>=images.length)return;[images[at],images[next]]=[images[next],images[at]];imageAction(()=>api.reorder(id,images.map(item=>item.id)))}
  if (!cat || !draft) return <div className="container page-pad">{error || 'Carregando cadastro...'}</div>
  return <div className="container page-pad review-page">
    <Link className="back-link" to="/admin/dashboard">← Voltar ao painel</Link>
    <div className="admin-heading"><div><span className="eyebrow">REVISÃO ADMINISTRATIVA · {cat.status.replaceAll('_',' ')}</span><h1>Revisar {cat.name}</h1><p>Confira as fotos e corrija todas as informações antes de publicar.</p></div></div>
    {cat.prediction && <div className="prediction-box">
      <span className="eyebrow">✦ PREVISÃO ORIGINAL</span>
      <div className="prediction-grid">
        <div><small>Raça sugerida pela IA</small><strong>{!cat.prediction.predicted_breed || cat.prediction.predicted_breed === 'srd' ? 'Raça não confirmada' : breedLabel(cat.prediction.predicted_breed)}</strong><small>{cat.prediction.breed_confidence != null ? `${Math.round(cat.prediction.breed_confidence * 100)}% de confiança na sugestão` : 'Confiança insuficiente para sugerir uma raça'}</small></div>
        <div><small>Cores</small><strong>{cat.prediction.predicted_colors?.map(colorLabel).join(' · ') || 'Sem previsão'}</strong></div>
        <div><small>Comprimento do pelo</small><strong>{coatLengthLabel(cat.prediction.predicted_coat_length) || 'Sem previsão'}</strong></div>
      </div>
      <p className="muted">A previsão não comprova a raça. Confira as fotos e defina a raça no campo abaixo antes de publicar.</p>
      {cat.prediction.breed_scores && Object.keys(cat.prediction.breed_scores).length > 0
        ? <BreedScores scores={cat.prediction.breed_scores}/>
        : <p className="muted">Ranking indisponível para este cadastro antigo: as pontuações não eram salvas.</p>}
    </div>}
    <div className="review-grid">
      <section className="form-shell">
        <div className="step-marker"><span>01</span> Fotografias</div>
        <div className="review-photos">{[...cat.images].sort((a, b) => a.position - b.position).map(image => <div className="review-photo" key={image.id}>
          <button className="photo-open" onClick={() => setLarge(image.url)} title="Ampliar foto"><img src={image.url} alt={`Foto de ${cat.name}`}/></button>
          <small>{image.is_primary ? '★ Foto principal' : `Foto ${image.position + 1}`}</small>
          <div className="mini-actions"><button disabled={busy || image.is_primary} onClick={() => imageAction(() => api.primaryImage(id, image.id))}>Principal</button><button disabled={busy} onClick={() => move(image, -1)}>←</button><button disabled={busy} onClick={() => move(image, 1)}>→</button><button disabled={busy || cat.images.length <= 1} onClick={() => imageAction(() => api.deleteImage(id, image.id))}>Remover</button></div>
        </div>)}</div>
        <label className="field">Adicionar fotos<input type="file" multiple accept="image/jpeg,image/png,image/webp" disabled={busy} onChange={e => { const files = Array.from(e.target.files || []); if (files.length) imageAction(() => api.addImages(id, files)) }}/></label>
      </section>
      <section className="form-shell">
        <div className="step-marker"><span>02</span> Dados do gato</div>
        <div className="form-grid">{textFields.map(([key, label]) => {
          const choices: Record<string, string> | null = key === 'breed' ? breedLabels : key === 'primary_color' || key === 'secondary_color' ? colorLabels : key === 'coat_length' ? coatLengthLabels : null
          const value = String(draft[key] || '')
          const onChange = (next: string) => { setDraft({ ...draft, [key]: next, ...(key === 'breed' ? {breed_training_verified: false} : {}) }); setConfirmedVisual(false) }
          return <label className="field" key={key}>{label}{choices ? <select value={value} onChange={e => onChange(e.target.value)}>
            <option value="">Não informado</option>
            {value && !choices[value] && <option value={value}>{value.replaceAll('_', ' ')}</option>}
            {Object.entries(choices).map(([code, text]) => <option key={code} value={code}>{text}</option>)}
          </select> : ['behavior', 'health_information', 'additional_notes', 'description'].includes(key) ? <textarea value={value} onChange={e => onChange(e.target.value)}/> : <input value={value} onChange={e => onChange(e.target.value)}/>}</label>
        })}</div>
        <div className="divider"/>
        <label className="review-confirmation"><input type="checkbox" checked={Boolean(draft.breed_training_verified)} disabled={!draft.breed || draft.breed === 'srd'} onChange={e => setDraft({...draft, breed_training_verified:e.target.checked})}/> Raça com procedência suficiente para uso em treino futuro</label>
        <label className="field">Procedência do rótulo de raça (obrigatória ao marcar para treino)<textarea maxLength={500} value={draft.breed_verification_note || ''} onChange={e => setDraft({...draft, breed_verification_note:e.target.value})} placeholder="Ex.: documentação de pedigree conferida; identificação do criador..."/></label>
        <p className="muted">A raça exibida no anúncio pode ser corrigida sem marcar este cadastro para treino. Nenhum retreino é automático.</p>
      </section>
    </div>
    {cat.review_events && cat.review_events.length > 0 && <section className="form-shell"><h2>Histórico de revisão</h2><div className="review-history">{cat.review_events.map(event => <p key={event.id}><strong>{new Date(event.created_at).toLocaleString('pt-BR')}</strong> · {event.event_type === 'edit' ? 'Edição' : event.event_type === 'publish' ? 'Publicação' : 'Rejeição'} · {breedLabel(event.previous_breed) || 'não informado'} → {breedLabel(event.reviewed_breed) || 'não informado'} · {event.breed_training_verified ? 'rótulo marcado para treino' : 'não usar para treino'}{event.reviewed_by ? ` · admin ${event.reviewed_by.slice(0, 8)}` : ''}</p>)}</div></section>}
    {error && <p className="error">{error}</p>}{message && <p className="notice">{message}</p>}
    {cat.status === 'PENDING_REVIEW' && <label className="review-confirmation"><input type="checkbox" checked={confirmedVisual} onChange={e => setConfirmedVisual(e.target.checked)}/> Conferi nas fotos a raça, as cores e o comprimento do pelo (inclusive campos marcados como não informados).</label>}
    <div className="review-actions"><button className="button light" disabled={busy} onClick={() => save()}>Salvar alterações</button><button className="button light" disabled={busy} onClick={() => save(true)}>Atualizar descrição</button>{cat.status === 'PENDING_REVIEW' && <><button className="button danger" disabled={busy} onClick={() => action('reject')}>Rejeitar</button><button className="button primary" disabled={busy || !confirmedVisual} onClick={() => action('publish')}>Aprovar e publicar →</button></>}{cat.status === 'PUBLISHED' && <><button className="button light" disabled={busy} onClick={() => action('archive')}>Arquivar</button><button className="button primary" disabled={busy} onClick={() => action('adopt')}>Marcar como adotado</button></>}{cat.status === 'ADOPTED' && <button className="button light" disabled={busy} onClick={() => action('archive')}>Arquivar</button>}</div>
    {large && <div className="lightbox" onClick={() => setLarge(null)} role="presentation"><button aria-label="Fechar foto">×</button><img src={large} alt="Foto ampliada"/></div>}
  </div>
}

