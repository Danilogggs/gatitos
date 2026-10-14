import { useEffect, useState, type DragEvent } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { BreedScores } from '../components/BreedScores'
import { api } from '../services/api'
import type { Prediction } from '../types'
import { coatLengthLabel } from '../utils/coatLength'
import { breedLabel, colorLabel } from '../utils/catLabels'

const optionalFields = [
  ['sex', 'Sexo'],
  ['approximate_age', 'Idade aproximada'],
  ['size', 'Porte'],
  ['behavior', 'Comportamento informado'],
  ['health_information', 'Informações de saúde'],
  ['location', 'Localização'],
  ['additional_notes', 'Observações'],
] as const

export function Register() {
  const navigate = useNavigate()
  const [name, setName] = useState('')
  const [files, setFiles] = useState<File[]>([])
  const [previews, setPreviews] = useState<string[]>([])
  const [optional, setOptional] = useState<Record<string, string>>({})
  const [prediction, setPrediction] = useState<Prediction | null>(null)
  const [mode, setMode] = useState('mock')
  const [showPublicRanking, setShowPublicRanking] = useState<boolean | null>(null)
  const [working, setWorking] = useState(false)
  const [error, setError] = useState('')
  const [uploadError, setUploadError] = useState('')
  const [dragActive, setDragActive] = useState(false)
  const breedRanking = Object.entries(prediction?.breed_scores || {})
    .sort((a, b) => b[1] - a[1]).slice(0, 3)
  const breedUnconfirmed = !prediction?.breed || prediction.breed === 'srd'

  useEffect(() => {
    api.config().then(data => { setMode(data.ml_mode); setShowPublicRanking(data.show_public_breed_ranking ?? false) }).catch(() => setShowPublicRanking(false))
  }, [])

  useEffect(() => {
    const urls = files.map(file => URL.createObjectURL(file))
    setPreviews(urls)
    return () => urls.forEach(URL.revokeObjectURL)
  }, [files])

  const selectFiles = (selected: File[]) => {
    if (!selected.length) return
    if (selected.length > 8) {
      setUploadError('Selecione no máximo 8 fotos.')
      return
    }
    if (selected.some(file => !['image/jpeg', 'image/png', 'image/webp'].includes(file.type))) {
      setUploadError('Use apenas imagens JPG, PNG ou WebP.')
      return
    }
    if (selected.some(file => file.size > 5 * 1024 * 1024)) {
      setUploadError('Cada foto deve ter no máximo 5 MB.')
      return
    }
    setFiles(selected)
    setPrediction(null)
    setUploadError('')
  }

  const dropFiles = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault()
    setDragActive(false)
    selectFiles(Array.from(event.dataTransfer.files))
  }

  const analyze = async () => {
    if (!name.trim() || !files.length) {
      setError('Informe o nome e adicione pelo menos uma foto.')
      return
    }
    setError('')
    setWorking(true)
    try {
      setPrediction(await api.predict(name.trim(), optional.behavior || '', files))
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setWorking(false)
    }
  }

  const submit = async () => {
    if (!name.trim() || !files.length) {
      setError('Informe o nome e adicione pelo menos uma foto.')
      return
    }
    setWorking(true)
    setError('')
    try {
      const result = await api.create(name.trim(), files, optional)
      navigate(`/enviado/${result.id}`)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setWorking(false)
    }
  }

  return (
    <div className="container page-pad form-page">
      <Link className="back-link" to="/">← Voltar ao início</Link>
      <div className="page-title">
        <span className="eyebrow">AJUDE UM GATINHO</span>
        <h1>Vamos contar a história dele?</h1>
        <p>Nome e pelo menos uma foto bastam para começar. O resto você preenche se souber.</p>
      </div>
      <div className="form-shell">
        <div className="step-marker"><span>01</span> Sobre o gatinho</div>
        <label className="field">
          Nome do gato <strong>*</strong>
          <input
            value={name}
            maxLength={100}
            onChange={event => { setName(event.target.value); setPrediction(null) }}
            placeholder="Como ele se chama?"
          />
        </label>
        <div className="field photo-field">
          <span>Fotos <strong>*</strong></span>
          <label
            className={`photo-dropzone${dragActive ? ' dragging' : ''}`}
            onDragEnter={event => { event.preventDefault(); setDragActive(true) }}
            onDragOver={event => {
              event.preventDefault()
              event.dataTransfer.dropEffect = 'copy'
              setDragActive(true)
            }}
            onDragLeave={event => {
              const next = event.relatedTarget as Node | null
              if (!next || !event.currentTarget.contains(next)) setDragActive(false)
            }}
            onDrop={dropFiles}
          >
            <input
              className="visually-hidden"
              type="file"
              accept="image/jpeg,image/png,image/webp"
              multiple
              onChange={event => selectFiles(Array.from(event.target.files || []))}
            />
            <span className="dropzone-icon" aria-hidden="true">⇩</span>
            <strong>{dragActive ? 'Solte as fotos aqui' : 'Arraste e solte as fotos aqui'}</strong>
            <span>ou clique para escolher no computador</span>
            {files.length > 0 && (
              <span className="dropzone-count">
                {files.length} {files.length === 1 ? 'foto selecionada' : 'fotos selecionadas'}
              </span>
            )}
          </label>
          <small>De 1 a 8 fotos JPG, PNG ou WebP, até 5 MB cada.</small>
          {uploadError && <small className="field-error" role="alert">{uploadError}</small>}
        </div>
        {files.length > 0 && (
          <div className="upload-preview">
            {files.map((file, index) => (
              <div key={`${file.name}-${index}`}>
                <img src={previews[index]} alt={`Prévia ${index + 1}`} />
                {index === 0 && <span>★ Principal</span>}
              </div>
            ))}
          </div>
        )}
        <div className="divider" />
        <div className="step-marker"><span>02</span> Informações opcionais</div>
        <div className="form-grid">
          {optionalFields.map(([key, label]) => (
            <label className="field" key={key}>
              {label}
              {['behavior', 'health_information', 'additional_notes'].includes(key) ? (
                <textarea
                  value={optional[key] || ''}
                  onChange={event => { setOptional({ ...optional, [key]: event.target.value }); setPrediction(null) }}
                  maxLength={key === 'additional_notes' ? 2000 : 1000}
                />
              ) : (
                <input
                  value={optional[key] || ''}
                  onChange={event => { setOptional({ ...optional, [key]: event.target.value }); setPrediction(null) }}
                />
              )}
            </label>
          ))}
        </div>
        {error && <p className="error">{error}</p>}
        {!prediction || !showPublicRanking ? (
          <button className="button primary" disabled={working || showPublicRanking === null} onClick={showPublicRanking ? analyze : submit}>
            {showPublicRanking === null ? 'Carregando…' : showPublicRanking
              ? working ? `Analisando as características de ${name}...` : 'Analisar e continuar →'
              : working ? 'Enviando para revisão...' : 'Enviar para revisão →'}
          </button>
        ) : (
          <div className="prediction-box">
            <span className="eyebrow">✦ PRÉVIA DA ANÁLISE</span>
            {mode === 'mock' && <p className="notice">Modo demonstração: o modelo ainda não foi treinado. Nenhum atributo foi identificado.</p>}
            <div className="prediction-grid">
              <div><small>Raça sugerida · IA</small><strong>{breedUnconfirmed ? 'Raça não confirmada' : breedLabel(prediction.breed)}</strong><small>{prediction.breed_confidence != null ? `${Math.round(prediction.breed_confidence * 100)}% de confiança na sugestão` : 'Confiança insuficiente para sugerir uma raça'}</small></div>
              <div><small>Cores · IA</small><strong>{prediction.colors?.map(colorLabel).join(' · ') || 'Ainda não identificadas'}</strong></div>
              <div><small>Comprimento do pelo · IA</small><strong>{coatLengthLabel(prediction.coat_length) || 'Ainda não identificado'}</strong></div>
            </div>
            {breedRanking.length > 0 && <p className="muted">Principais alternativas: {breedRanking.map(([breed, score]) => `${breedLabel(breed)} (${Math.round(score * 100)}%)`).join(' · ')}. São pontuações da IA, não confirmação de raça.</p>}
            <BreedScores scores={prediction.breed_scores} />
            <p className="muted">Descrição sugerida: {prediction.description}</p>
            <p className="muted">As informações serão revisadas por um administrador antes da publicação.</p>
            <div className="button-row">
              <button className="button light" onClick={() => setPrediction(null)}>Voltar e editar</button>
              <button className="button primary" disabled={working} onClick={submit}>{working ? 'Enviando...' : 'Enviar para revisão →'}</button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export function Sent() {
  return (
    <div className="container page-pad success-page">
      <div className="success-icon">♡</div>
      <span className="eyebrow">CADASTRO RECEBIDO</span>
      <h1>Obrigado por cuidar!</h1>
      <p>O cadastro foi enviado para revisão. Depois de aprovado, o gatinho aparecerá na página pública.</p>
      <Link className="button primary" to="/">Voltar para o início</Link>
    </div>
  )
}
