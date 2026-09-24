import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { CatCard } from '../components/CatCard'
import { api } from '../services/api'
import type { Cat, Config } from '../types'

const filters = [
  ['breed','Raça'], ['feature','Característica'], ['color','Cor'], ['sex','Sexo'],
  ['size','Porte'], ['age','Idade aproximada'], ['coat_length','Pelo']
] as const

export function Home() {
  const [cats, setCats] = useState<Cat[]>([])
  const [config, setConfig] = useState<Config | null>(null)
  const [all, setAll] = useState<Cat[]>([])
  const [query, setQuery] = useState<Record<string,string>>({})
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  useEffect(() => { api.config().then(setConfig).catch(() => {}); api.cats().then(setAll).catch(() => {}) }, [])
  useEffect(() => {
    const params = new URLSearchParams(Object.entries(query).filter(([, value]) => value))
    setLoading(true)
    api.cats(params.toString()).then(setCats).catch(err => setError(err.message)).finally(() => setLoading(false))
  }, [query])
  const options = (key: string) => {
    if (key === 'breed') return config?.breeds || []
    if (key === 'feature') return Object.keys(config?.features || {})
    if (key === 'color') return [...new Set(all.flatMap(cat => [cat.primary_color, cat.secondary_color]).filter(Boolean))] as string[]
    const field = key === 'age' ? 'approximate_age' : key
    return [...new Set(all.map(cat => cat[field as keyof Cat]).filter(value => typeof value === 'string' && value))] as string[]
  }
  return <><section className="hero"><div className="container hero-grid"><div className="hero-copy"><span className="eyebrow"><span className="dot"/> ADOÇÃO COM MAIS CUIDADO</span><h1>Seu próximo grande amor pode ter <em>bigodes.</em></h1><p>Conheça gatos incríveis esperando por um lar. Cada história é cadastrada com ajuda da tecnologia e revisada por pessoas de verdade.</p><div className="hero-actions"><a className="button primary" href="#gatinhos">Conhecer os gatinhos <span>↗</span></a><Link className="button light" to="/cadastrar">Cadastrar um gato</Link></div><div className="hero-note"><span>♡</span> Um encontro pode mudar duas vidas.</div></div><div className="hero-art" aria-hidden="true"><div className="art-orbit orbit-one"/><div className="art-orbit orbit-two"/><div className="art-cat">🐈</div><div className="floating-card">✦ &nbsp;Um lar começa aqui</div><div className="art-star">✳</div></div></div></section>
  <section id="gatinhos" className="listing container"><div className="section-heading"><div><span className="eyebrow">NOSSOS AMIGUINHOS</span><h2>Conheça quem está por aqui</h2><p>Use os filtros para encontrar um gatinho que combine com você.</p></div><span className="count-pill">{cats.length} {cats.length === 1 ? 'gatinho' : 'gatinhos'}</span></div>
    <div className="search-panel"><label className="searchbox"><span>⌕</span><input aria-label="Buscar pelo nome" placeholder="Procure pelo nome..." value={query.name || ''} onChange={e => setQuery({...query, name:e.target.value})}/></label><div className="filter-grid">{filters.map(([key,label]) => <label key={key}>{label}<select value={query[key] || ''} onChange={e => setQuery({...query, [key]:e.target.value})}><option value="">Todos</option>{options(key).map(option => <option key={option} value={option}>{option.replaceAll('_',' ')}</option>)}</select></label>)}</div><button className="clear-link" onClick={() => setQuery({})}>Limpar filtros</button></div>
    {error && <p className="error">{error}</p>}{loading ? <p className="empty-state">Buscando gatinhos...</p> : cats.length ? <div className="cat-grid">{cats.map(cat => <CatCard key={cat.id} cat={cat}/>)}</div> : <div className="empty-state"><span>🐾</span><h3>Nenhum gatinho encontrado</h3><p>Tente outros filtros ou volte em breve para conhecer novos amigos.</p></div>}
  </section></>
}

