export const coatLengthLabels: Record<string, string> = {
  hairless: 'Sem pelo',
  short: 'Curto',
  medium: 'Médio',
  long: 'Longo',
}

export function coatLengthLabel(value: string | null | undefined): string {
  return value ? coatLengthLabels[value] || value : ''
}
