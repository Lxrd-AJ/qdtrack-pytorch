import os
import unittest
import random

from qdtrack.dataset.mot_dataset import MOTDataset

class TestMOTDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_dir = os.path.join(os.getcwd(), "MOT17/train")
        cls.dataset = MOTDataset(cls.data_dir)

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def test_dataset_setup(self):
        self.assertEqual(self.dataset.num_sequences, 7)

        seq_numbers = self.dataset.seq_numbers()
        for x in ["02", "05", "09", "13"]:
            self.assertIn(x, seq_numbers)

    def test_length(self):
        # 5316 is the length of the unique sequences
        # For each sequence e.g 04 , there are 3 generated sequences for it based on the 
        # object detector used
        self.assertEqual(len(self.dataset), 5316) #length of the training dataset

    def test_indexer(self):
        expected_pth = os.path.join(self.data_dir, "MOT17-02-FRCNN", "img1", "000001.jpg")
        self.assertEqual(self.dataset.seq_indexer("02", 0), expected_pth )

        expected_pth = os.path.join(self.data_dir, "MOT17-04-FRCNN", "img1", "001050.jpg")
        self.assertEqual(self.dataset.seq_indexer("04", 1049), expected_pth )

        (seq, image_num) = self.dataset.indexer(5316)
        self.assertEqual(seq, '13')
        self.assertEqual(image_num, 749)

        (seq, image_num) = self.dataset.indexer(4567)
        self.assertEqual(seq, '13')
        self.assertEqual(image_num, 1)
        
        (seq, image_num) = self.dataset.indexer(1)
        self.assertEqual(seq, '02')
        self.assertEqual(image_num, 1)


    def test_getitem(self):
        idx = random.randint(0, len(self.dataset))
        print(f"Index = {idx}")
        (im1, im2) = self.dataset[idx]
        im1.show()
        im2.show()
        

if __name__ == "__main__":
    unittest.main()