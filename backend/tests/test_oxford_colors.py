import unittest

import numpy as np
from PIL import Image

from app.ml.train_oxford_colors import COLOR_PROTOTYPES, infer_color_labels, recolor_foreground
from app.ml.taxonomy import COLORS


class OxfordColorLabelsTest(unittest.TestCase):
    def test_each_prototype_maps_to_its_color(self):
        mask = Image.fromarray(np.ones((32, 32), dtype=np.uint8))
        for color, rgb in zip(COLORS, COLOR_PROTOTYPES.astype(np.uint8)):
            with self.subTest(color=color):
                image = Image.new('RGB', (32, 32), tuple(int(value) for value in rgb))
                self.assertEqual(infer_color_labels(image, mask), [color])

    def test_multiple_coat_colors_are_multilabel(self):
        pixels = np.zeros((40, 40, 3), dtype=np.uint8)
        pixels[:, :20] = COLOR_PROTOTYPES[COLORS.index('black')]
        pixels[:, 20:] = COLOR_PROTOTYPES[COLORS.index('white')]
        image = Image.fromarray(pixels)
        mask = Image.fromarray(np.ones((40, 40), dtype=np.uint8))
        self.assertEqual(set(infer_color_labels(image, mask)), {'black', 'white'})

    def test_background_pixels_are_ignored(self):
        pixels = np.full((40, 40, 3), (0, 255, 0), dtype=np.uint8)
        pixels[10:30, 10:30] = COLOR_PROTOTYPES[COLORS.index('orange')]
        trimap = np.full((40, 40), 2, dtype=np.uint8)
        trimap[10:30, 10:30] = 1
        image = Image.fromarray(pixels)
        mask = Image.fromarray(trimap)
        self.assertEqual(infer_color_labels(image, mask), ['orange'])

    def test_synthetic_variants_change_only_the_foreground(self):
        pixels = np.full((40, 40, 3), (0, 255, 0), dtype=np.uint8)
        pixels[10:30, 10:30] = (128, 128, 128)
        trimap = np.full((40, 40), 2, dtype=np.uint8)
        trimap[10:30, 10:30] = 1
        image = Image.fromarray(pixels)
        mask = Image.fromarray(trimap)
        recolored = recolor_foreground(image, mask, 'orange')
        result = np.asarray(recolored)
        self.assertTrue(np.array_equal(result[0, 0], pixels[0, 0]))
        self.assertFalse(np.array_equal(result[20, 20], pixels[20, 20]))
        self.assertEqual(infer_color_labels(recolored, mask)[0], 'orange')


if __name__ == '__main__':
    unittest.main()
