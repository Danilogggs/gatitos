import { Link } from 'react-router-dom'
import type { Cat } from '../types'
import { breedLabel, colorLabel } from '../utils/catLabels'

export function CatCard({ cat }: {cat: Cat}) {
  const photo = cat.images.find(image => image.is_primary) || cat.images[0]
  return <article className="cat-card">
    {photo ? <img src={photo.url} alt={`Foto de ${cat.name}`} /> : <div className="photo-fallback">🐈</div>}
    <div className="cat-card-body"><span className="eyebrow">À procura de um lar</span><h3>{cat.name}</h3>
      <p>{[breedLabel(cat.breed), [cat.primary_color, cat.secondary_color].filter(Boolean).map(colorLabel).join(' e ')].filter(Boolean).join('  /  ') || 'Um gatinho especial'}</p>
      <div className="card-actions"><Link className="text-link" to={`/gatos/${cat.id}`}>Conhecer {cat.name} <span>↗</span></Link><button className="button ghost small" disabled title="Adoção disponível em uma versão futura">Adotar</button></div>
    </div>
  </article>
}

