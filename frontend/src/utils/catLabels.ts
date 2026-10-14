export const breedLabels: Record<string, string> = {
  srd: 'Sem raça definida (SRD)',
  abyssinian: 'Abissínio',
  bengal: 'Bengal',
  birman: 'Sagrado da Birmânia',
  bombay: 'Bombaim',
  british_shorthair: 'Britânico de Pelo Curto',
  egyptian_mau: 'Mau Egípcio',
  maine_coon: 'Maine Coon',
  persian: 'Persa',
  ragdoll: 'Ragdoll',
  russian_blue: 'Azul Russo',
  siamese: 'Siamês',
  sphynx: 'Sphynx',
}

export const colorLabels: Record<string, string> = {
  black: 'Preto',
  white: 'Branco',
  gray: 'Cinza',
  orange: 'Laranja',
  cream: 'Creme',
  brown: 'Marrom',
}

const fallbackLabel = (value: string) => value.replaceAll('_', ' ')

export const breedLabel = (value: string | null | undefined): string =>
  value ? breedLabels[value] || fallbackLabel(value) : ''

export const colorLabel = (value: string | null | undefined): string =>
  value ? colorLabels[value] || fallbackLabel(value) : ''
