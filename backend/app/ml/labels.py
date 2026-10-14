"""Rótulos para textos em português; os códigos persistidos permanecem estáveis."""

BREED_LABELS = {
    'srd': 'sem raça definida (SRD)',
    'abyssinian': 'abissínio',
    'bengal': 'bengal',
    'birman': 'sagrado da Birmânia',
    'bombay': 'bombaim',
    'british_shorthair': 'britânico de pelo curto',
    'egyptian_mau': 'mau egípcio',
    'maine_coon': 'Maine Coon',
    'persian': 'persa',
    'ragdoll': 'ragdoll',
    'russian_blue': 'azul russo',
    'siamese': 'siamês',
    'sphynx': 'sphynx',
}

COLOR_LABELS = {
    'black': 'preto',
    'white': 'branco',
    'gray': 'cinza',
    'orange': 'laranja',
    'cream': 'creme',
    'brown': 'marrom',
}

COAT_LENGTH_LABELS = {
    'hairless': 'sem pelo',
    'short': 'pelo curto',
    'medium': 'pelo médio',
    'long': 'pelo longo',
}


def breed_label(value: str | None) -> str:
    return BREED_LABELS.get(value, value.replace('_', ' ') if value else '')


def color_label(value: str) -> str:
    return COLOR_LABELS.get(value, value.replace('_', ' '))


def coat_length_label(value: str | None) -> str:
    return COAT_LENGTH_LABELS.get(value, value.replace('_', ' ') if value else '')
