import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api } from '../services/api'
import type { Prediction } from '../types'

const optionalFields = [['sex','Sexo'],['approximate_age','Idade aproximada'],['size','Porte'],['behavior','Comportamento informado'],['health_information','Informações de saúde'],['location','Localização'],['additional_notes','Observações']] as const

export function Register() {
  const navigate = useNavigate()
  const [name, setName] = useState('')
  const [files, setFiles] = useState<File[]>([])
  const [previews, setPreviews] = useState<string[]>([])
  const [optional, setOptional] = useState<Record<string,string>>({})
  const [prediction, setPrediction] = useState<Prediction | null>(null)
  const [mode, setMode] = useState('mock')
  const [working, setWorking] = useState(false)
  const [error, setError] = useState('')
  useEffect(() => { api.config().then(data => setMode(data.ml_mode)).catch(() => {}) }, [])
  useEffect(() => { const urls = files.map(file => URL.createObjectURL(file)); setPreviews(urls); return () => urls.forEach(URL.revokeObjectURL) }, [files])
  const analyze = async () => {
    if (!name.trim() || !files.length) { setError('Informe o nome e adicione pelo menos uma foto.'); return }
    setError(''); setWorking(true)
    try { setPrediction(await api.predict(name.trim(), optional.behavior || '', files)) } catch (err) { setError((err as Error).message) } finally { setWorking(false) }
  }
  const submit = async () => {
    setWorking(true); setError('')
    try { const result = await api.create(name.trim(), files, optional); navigate(`/enviado/${result.id}`) }
    catch (err) { setError((err as Error).message) } finally { setWorking(false) }
  }
  return <div className="container page-pad form-page"><Link className="back-link" to="/">← Voltar ao início</Link><div className="page-title"><span className="eyebrow">AJUDE UM GATINHO</span><h1>Vamos contar a história dele?</h1><p>Nome e pelo menos uma foto bastam para começar. O resto você preenche se souber.</p></div><div className="form-shell"><div className="step-marker"><span>01</span> Sobre o gatinho</div><label className="field">Nome do gato <strong>*</strong><input value={name} maxLength={100} onChange={e => {setName(e.target.value);setPrediction(null)}} placeholder="Como ele se chama?" /></label><label className="field">Fotos <strong>*</strong><input type="file" accept="image/jpeg,image/png,image/webp" multiple onChange={e => {setFiles(Array.from(e.target.files || []));setPrediction(null)}} /><small>De 1 a 8 fotos JPG, PNG ou WebP, até 5 MB cada.</small></label>{files.length > 0 && <div className="upload-preview">{files.map((file,index) => <div key={`${file.name}-${index}`}><img src={previews[index]} alt={`Prévia ${index+1}`}/>{index === 0 && <span>★ Principal</span>}</div>)}</div>}<div className="divider"/><div className="step-marker"><span>02</span> Informações opcionais</div><div className="form-grid">{optionalFields.map(([key,label]) => <label className="field" key={key}>{label}{['behavior','health_information','additional_notes'].includes(key) ? <textarea value={optional[key] || ''} onChange={e => {setOptional({...optional,[key]:e.target.value});setPrediction(null)}} maxLength={key === 'additional_notes' ? 2000 : 1000}/> : <input value={optional[key] || ''} onChange={e => {setOptional({...optional,[key]:e.target.value});setPrediction(null)}}/>}</label>)}</div>{error && <p className="error">{error}</p>}{!prediction ? <button className="button primary" disabled={working} onClick={analyze}>{working ? `Analisando as características de ${name}...` : 'Analisar e continuar →'}</button> : <div className="prediction-box"><span className="eyebrow">✦ PRÉVIA DA ANÁLISE</span>{mode === 'mock' && <p className="notice">Modo demonstração: o modelo ainda não foi treinado. Nenhuma característica foi inferida.</p>}<div className="prediction-grid"><div><small>Raça provável · IA</small><strong>{prediction.breed || 'Ainda não identificada'}</strong></div><div><small>Características · IA</small><strong>{prediction.features?.join(' · ') || 'Ainda não identificadas'}</strong></div><div><small>Padrão · IA</small><strong>{prediction.coat_pattern || 'Ainda não identificado'}</strong></div><div><small>Cores · IA</small><strong>{prediction.colors?.join(' · ') || 'Ainda não identificadas'}</strong></div></div><p className="muted">Descrição sugerida: {prediction.description}</p><p className="muted">As informações serão revisadas por um administrador antes da publicação.</p><div className="button-row"><button className="button light" onClick={() => setPrediction(null)}>Voltar e editar</button><button className="button primary" disabled={working} onClick={submit}>{working ? 'Enviando...' : 'Enviar para revisão →'}</button></div></div>}</div></div>
}

export function Sent() { return <div className="container page-pad success-page"><div className="success-icon">♡</div><span className="eyebrow">CADASTRO RECEBIDO</span><h1>Obrigado por cuidar!</h1><p>O cadastro foi enviado para revisão. Depois de aprovado, o gatinho aparecerá na página pública.</p><Link className="button primary" to="/">Voltar para o início</Link></div> }
