import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { api } from '../services/api'
import type { Cat } from '../types'

export function Details() {
  const { id = '' } = useParams()
  const [cat, setCat] = useState<Cat | null>(null)
  const [selected, setSelected] = useState(0)
  const [error, setError] = useState('')
  useEffect(() => { api.cat(id).then(setCat).catch(err => setError(err.message)) }, [id])
  if (error) return <div className="container page-pad"><p className="error">{error}</p><Link to="/">Voltar aos gatos</Link></div>
  if (!cat) return <div className="container page-pad">Carregando...</div>
  const images = [...cat.images].sort((a,b) => Number(b.is_primary) - Number(a.is_primary))
  const facts = [['Raça',cat.breed],['Características',cat.features.join(' · ')],['Cores',[cat.primary_color,cat.secondary_color].filter(Boolean).join(' e ')],['Padrão',cat.coat_pattern],['Pelo',cat.coat_length],['Sexo',cat.sex],['Idade aproximada',cat.approximate_age],['Porte',cat.size],['Comportamento informado',cat.behavior],['Saúde',cat.health_information],['Localização',cat.location],['Observações',cat.additional_notes]].filter(([, value]) => value)
  return <div className="container page-pad"><Link className="back-link" to="/">← Voltar para os gatinhos</Link><div className="detail-grid"><div className="detail-gallery">{images[selected] && <img className="main-photo" src={images[selected].url} alt={`${cat.name}, foto ${selected+1}`}/>}<div className="thumb-row">{images.map((image,index) => <button key={image.id} className={selected === index ? 'selected' : ''} onClick={() => setSelected(index)}><img src={image.url} alt={`Ver foto ${index+1}`}/></button>)}</div></div><div className="detail-copy"><span className="eyebrow">UM NOVO AMIGO ESPERA POR VOCÊ</span><h1>{cat.name}</h1><p className="detail-lead">{cat.description || `${cat.name} está procurando um lar cheio de carinho.`}</p><div className="facts">{facts.map(([label,value]) => <div key={label}><span>{label}</span><strong>{value}</strong></div>)}</div><button className="button primary" disabled title="Adoção disponível em uma versão futura">♡ &nbsp; Adotar {cat.name}</button><p className="muted">O processo de adoção estará disponível futuramente.</p></div></div></div>
}

