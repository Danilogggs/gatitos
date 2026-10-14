import csv
import tempfile
import unittest
from pathlib import Path

from app.ml.report_breed_metrics import CLASSES, class_metrics, confusion_matrix, run


class BreedMetricsTest(unittest.TestCase):
    def test_confusion_and_per_class_metrics(self):
        rows = [
            {'raca_rotulada': CLASSES[0], 'top1': CLASSES[1]},
            {'raca_rotulada': CLASSES[1], 'top1': CLASSES[1]},
        ]
        matrix = confusion_matrix(rows, 'top1', CLASSES)
        metrics = class_metrics(matrix)
        self.assertEqual(matrix[CLASSES[0]][CLASSES[1]], 1)
        self.assertEqual(metrics[0]['recall'], 0)
        self.assertEqual(metrics[1]['precision'], 0.5)
        self.assertEqual(metrics[1]['recall'], 1)
        self.assertAlmostEqual(metrics[1]['f1'], 2 / 3)

    def test_report_keeps_srd_as_abstention_not_training_class(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'predictions.csv'
            with source.open('w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=['oxford_test_index', 'raca_rotulada',
                    'top1', 'raca_exibida_no_app', 'versao_modelo'])
                writer.writeheader()
                for index, breed in enumerate(CLASSES):
                    writer.writerow({'oxford_test_index': index, 'raca_rotulada': breed,
                        'top1': breed, 'raca_exibida_no_app': 'srd' if index == 0 else breed,
                        'versao_modelo': 'test-model'})
            result = run(source, root / 'report')
            self.assertEqual(result['raw_accuracy'], 1)
            self.assertEqual(result['app_accuracy'], 11 / 12)
            self.assertEqual(result['abstentions'], 1)
            with (root / 'report' / 'matriz_confusao_app_com_srd.csv').open(encoding='utf-8', newline='') as handle:
                matrix = list(csv.DictReader(handle))
            self.assertEqual(matrix[0]['srd'], '1')


if __name__ == '__main__':
    unittest.main()
