import unittest

from app.ml.taxonomy import COAT_LENGTHS
from app.ml.train_oxford import OXFORD_BREEDS
from app.ml.train_oxford_length import BREED_TO_LENGTH


class OxfordLengthMappingTest(unittest.TestCase):
    def test_all_oxford_cat_breeds_have_supported_length_labels(self):
        self.assertEqual(set(BREED_TO_LENGTH), set(OXFORD_BREEDS))
        self.assertEqual(set(BREED_TO_LENGTH.values()), set(COAT_LENGTHS))
        self.assertEqual(BREED_TO_LENGTH['sphynx'], 'hairless')


if __name__ == '__main__':
    unittest.main()
