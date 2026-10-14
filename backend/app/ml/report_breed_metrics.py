"""Gera métricas PjBL2 a partir das previsões Oxford já registradas.

Uso em backend/:
  python -m app.ml.report_breed_metrics \
    --input ../dataset/avaliacao-oxford-240/resultados.csv \
    --output ../reports/pjbl2
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

from app.ml.labels import breed_label
from app.ml.taxonomy import BREEDS


CLASSES = [breed for breed in BREEDS if breed != 'srd']


def confusion_matrix(rows: list[dict], prediction_key: str, columns: list[str]) -> dict[str, Counter]:
    matrix = {breed: Counter() for breed in CLASSES}
    for row in rows:
        predicted = row[prediction_key]
        if predicted not in columns:
            raise ValueError(f'Classe prevista inválida: {predicted}')
        matrix[row['raca_rotulada']][predicted] += 1
    return matrix


def class_metrics(matrix: dict[str, Counter]) -> list[dict]:
    result = []
    for breed in CLASSES:
        tp = matrix[breed][breed]
        support = sum(matrix[breed].values())
        predicted = sum(matrix[other][breed] for other in CLASSES)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result.append({'breed': breed, 'support': support, 'tp': tp,
                       'precision': precision, 'recall': recall, 'f1': f1})
    return result


def write_matrix(path: Path, matrix: dict[str, Counter], columns: list[str]) -> None:
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['raca_real', *columns])
        for breed in CLASSES:
            writer.writerow([breed, *(matrix[breed][predicted] for predicted in columns)])


def run(input_path: Path, output: Path) -> dict[str, float]:
    with input_path.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError('CSV de previsões vazio')
    expected_counts = Counter(row['raca_rotulada'] for row in rows)
    if set(expected_counts) != set(CLASSES):
        raise ValueError('O teste deve conter todas as 12 raças Oxford e nenhuma classe desconhecida')
    if len(set(expected_counts.values())) != 1:
        raise ValueError('Este relatório exige a avaliação balanceada (mesmo número por raça)')
    sample_ids = [row['oxford_test_index'] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError('Índice Oxford duplicado no teste')
    versions = {row['versao_modelo'] for row in rows}
    if len(versions) != 1:
        raise ValueError('Há previsões de versões diferentes do modelo no mesmo teste')

    raw = confusion_matrix(rows, 'top1', CLASSES)
    shown = confusion_matrix(rows, 'raca_exibida_no_app', [*CLASSES, 'srd'])
    metrics = class_metrics(raw)
    total = len(rows)
    raw_hits = sum(raw[breed][breed] for breed in CLASSES)
    shown_hits = sum(shown[breed][breed] for breed in CLASSES)
    abstentions = sum(shown[breed]['srd'] for breed in CLASSES)
    covered = total - abstentions
    macro = {key: sum(item[key] for item in metrics) / len(metrics)
             for key in ('precision', 'recall', 'f1')}
    weighted = {key: sum(item[key] * item['support'] for item in metrics) / total
                for key in ('precision', 'recall', 'f1')}

    output.mkdir(parents=True, exist_ok=True)
    write_matrix(output / 'matriz_confusao_12_racas.csv', raw, CLASSES)
    write_matrix(output / 'matriz_confusao_app_com_srd.csv', shown, [*CLASSES, 'srd'])
    with (output / 'metricas_por_raca.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['raca', 'suporte', 'acertos', 'precisao', 'recall', 'f1'])
        writer.writeheader()
        for item in metrics:
            writer.writerow({'raca': item['breed'], 'suporte': item['support'], 'acertos': item['tp'],
                             'precisao': f"{item['precision']:.4f}", 'recall': f"{item['recall']:.4f}",
                             'f1': f"{item['f1']:.4f}"})
    with (output / 'predicoes_teste.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['oxford_test_index', 'raca_real', 'top1_12_racas',
                                                   'resposta_app', 'modelo'])
        writer.writeheader()
        for row in rows:
            writer.writerow({'oxford_test_index': row['oxford_test_index'],
                             'raca_real': row['raca_rotulada'], 'top1_12_racas': row['top1'],
                             'resposta_app': row['raca_exibida_no_app'], 'modelo': row['versao_modelo']})

    off_diagonal = sorted(((raw[actual][predicted], actual, predicted)
                           for actual in CLASSES for predicted in CLASSES if actual != predicted
                           and raw[actual][predicted]), reverse=True)
    lines = [
        '# PjBL2 — avaliação da classificação de raças', '',
        f'**Modelo:** `{next(iter(versions))}`. **Base:** Oxford-IIIT Pet, split `test`,',
        f'subconjunto balanceado de {total} imagens originais ({expected_counts[CLASSES[0]]} por raça).',
        'Cada imagem foi avaliada individualmente. O script de seleção usa SHA-256 do índice',
        'Oxford e exclui seis imagens usadas antes na galeria de demonstração.', '',
        '## Aderência aos requisitos do PjBL2', '',
        '1. **Base com pelo menos 3 categorias:** Oxford-IIIT Pet, com 12 raças de gatos utilizadas como classes.',
        '2. **Protocolo experimental:** o `trainval` oficial é dividido de forma estratificada em treino e validação',
        '   com semente 42; o `test` oficial permanece fora do ajuste. Este relatório usa 20 imagens de teste por classe.',
        '3. **Modelo de visão computacional:** EfficientNet-B0 com transferência de aprendizado e uma cabeça linear',
        '   de 12 saídas. A implementação reproduzível está em `backend/app/ml/train_oxford.py` e no notebook Colab.',
        '4. **Avaliação:** matriz de confusão, acurácia, precisão, recall e F1 por classe, além das médias macro',
        '   e ponderada. Os CSVs e as predições individuais acompanham este relatório.', '',
        '## Resultado principal: 12 raças treinadas', '',
        f'- Acurácia top-1: **{raw_hits}/{total} = {raw_hits/total:.2%}**.',
        f"- Precisão macro: **{macro['precision']:.2%}**; recall macro: **{macro['recall']:.2%}**; F1 macro: **{macro['f1']:.2%}**.",
        f"- Precisão ponderada: **{weighted['precision']:.2%}**; recall ponderado: **{weighted['recall']:.2%}**; F1 ponderado: **{weighted['f1']:.2%}**.",
        '- A matriz de confusão está em [matriz_confusao_12_racas.csv](matriz_confusao_12_racas.csv):',
        'linhas são raças reais; colunas, raças previstas; diagonal são acertos.', '',
        '| Raça | Imagens | Precisão | Recall | F1 |',
        '|---|---:|---:|---:|---:|',
    ]
    for item in metrics:
        lines.append(f"| {breed_label(item['breed'])} | {item['support']} | "
                     f"{item['precision']:.1%} | {item['recall']:.1%} | {item['f1']:.1%} |")
    abbreviations = {
        'abyssinian': 'Aby', 'bengal': 'Ben', 'birman': 'Bir', 'bombay': 'Bom',
        'british_shorthair': 'BSh', 'egyptian_mau': 'Mau', 'maine_coon': 'MCo',
        'persian': 'Per', 'ragdoll': 'Rag', 'russian_blue': 'Rus',
        'siamese': 'Sia', 'sphynx': 'Sph',
    }
    lines.extend(['', '### Matriz de confusão — 12 raças', '',
                  'Linhas = raça real; colunas = previsão top-1; a diagonal indica acertos.', '',
                  '| Real \\ Prev | ' + ' | '.join(abbreviations[breed] for breed in CLASSES) + ' |',
                  '|---|' + '---:|' * len(CLASSES)])
    for actual in CLASSES:
        lines.append('| ' + abbreviations[actual] + ' | '
                     + ' | '.join(str(raw[actual][predicted]) for predicted in CLASSES) + ' |')
    lines.extend(['', 'Legenda: ' + '; '.join(f'{abbreviations[breed]} = {breed_label(breed)}'
                                           for breed in CLASSES) + '.'])
    lines.extend(['', 'Confusões mais frequentes:'])
    for count, actual, predicted in off_diagonal[:8]:
        lines.append(f'- {breed_label(actual)} → {breed_label(predicted)}: {count} imagem(ns).')
    lines.extend([
        '', '## Resposta exibida no aplicativo', '',
        f'- Com limiar de confiança de 40%, **{abstentions}/{total}** imagens receberam “SRD”',
        '  como rejeição por baixa confiança. SRD **não é uma classe treinada**.',
        f'- Acertos no conjunto todo: **{shown_hits}/{total} = {shown_hits/total:.2%}**.',
        f'- Cobertura com raça sugerida: **{covered}/{total} = {covered/total:.2%}**;',
        f'  acurácia apenas nas sugestões emitidas: **{shown_hits}/{covered} = {shown_hits/covered:.2%}**.' if covered else '  nenhuma sugestão de raça emitida.',
        '- A matriz operacional está em [matriz_confusao_app_com_srd.csv](matriz_confusao_app_com_srd.csv).', '',
        '## Método e limites', '',
        '- O treino de raça usa `trainval` Oxford, separado por classe em treino e validação',
        '  com semente 42; o checkpoint é escolhido pela acurácia de validação.',
        '- O split `test` Oxford fica fora do ajuste dos pesos de raça. Seu resultado, porém,',
        '  já foi consultado em experimentos anteriores: este relatório não é uma estimativa',
        '  totalmente inédita de generalização.',
        '- As métricas principais são de raça, **não** de cor ou comprimento do pelo.',
        '- Precisão = VP/(VP+FP); recall = VP/(VP+FN); F1 = 2·precisão·recall/(precisão+recall).',
        '- [predicoes_teste.csv](predicoes_teste.csv) contém os rótulos e resultados por imagem',
        '  para auditar os cálculos, sem redistribuir as fotos.',
        '- Para reproduzir: em `backend/`, execute',
        '  `python -m app.ml.report_breed_metrics --input ../dataset/avaliacao-oxford-240/resultados.csv --output ../reports/pjbl2`.',
        '',
    ])
    (output / 'relatorio.md').write_text('\n'.join(lines), encoding='utf-8')
    return {'total': total, 'raw_accuracy': raw_hits / total, 'app_accuracy': shown_hits / total,
            'macro_f1': macro['f1'], 'abstentions': abstentions}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path('../dataset/avaliacao-oxford-240/resultados.csv'))
    parser.add_argument('--output', type=Path, default=Path('../reports/pjbl2'))
    args = parser.parse_args()
    result = run(args.input, args.output)
    print(f"Acurácia 12 raças: {result['raw_accuracy']:.2%}; F1 macro: {result['macro_f1']:.2%}; "
          f"acurácia exibida: {result['app_accuracy']:.2%}; rejeições SRD: {result['abstentions']}. ")
    print(f'Relatório: {args.output / "relatorio.md"}')


if __name__ == '__main__':
    main()
