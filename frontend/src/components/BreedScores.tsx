type Props = {
  scores?: Record<string, number>
}

const percent = new Intl.NumberFormat('pt-BR', {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
})

export function BreedScores({ scores }: Props) {
  const ranking = Object.entries(scores || {}).sort((a, b) => b[1] - a[1])
  if (!ranking.length) return null

  return <section className="breed-scores" aria-label="Pontuação de todas as raças">
    <h3>Pontuação de todas as raças</h3>
    <p>Comparação entre as raças treinadas. SRD não foi treinado; estas porcentagens não confirmam a raça.</p>
    <div className="breed-score-list">
      {ranking.map(([breed, score]) => <div className="breed-score-row" key={breed}>
        <span className="breed-score-name">{breed.replaceAll('_', ' ')}</span>
        <div className="breed-score-track" aria-hidden="true"><span style={{ width: `${Math.max(score * 100, 0.2)}%` }} /></div>
        <strong>{percent.format(score * 100)}%</strong>
      </div>)}
    </div>
  </section>
}
