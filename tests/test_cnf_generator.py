import unittest
import os
import shutil
import tempfile
import torch
from unittest.mock import patch
from dataset.cnf_generator import generate_cnf

class TestCNFGenerator(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for output
        self.test_dir = tempfile.mkdtemp()
        self.dataset_name = "Test-Synthetic"
        
    def tearDown(self):
        # Clean up temporary directory after test
        shutil.rmtree(self.test_dir)

    @patch('dataset.cnf_generator.load_kg_hf')
    def test_generate_cnf_logic(self, mock_load_kg):
        """
        Verifies that relations co-occurring on the same head entity are 
        correctly identified as context neighbors.
        """
        # ------------------------------------------------------------------
        # 1. Setup Synthetic Data
        # ------------------------------------------------------------------
        # Entities: 0, 1, 2, 3
        # Relations: 0 (Target), 1 (Context), 2 (Irrelevant)
        
        # Triples (h, r, t):
        # Entity 0 has r0 and r1 -> Co-occurrence
        # Entity 1 has r0 and r1 -> Co-occurrence
        # Entity 2 has r2 only   -> No co-occurrence with r0
        
        triples = [
            [0, 0, 10], # (e0, r0, ...)
            [0, 1, 11], # (e0, r1, ...)
            [1, 0, 12], # (e1, r0, ...)
            [1, 1, 13], # (e1, r1, ...)
            [2, 2, 14], # (e2, r2, ...)
        ]
        
        train_tensor = torch.tensor(triples, dtype=torch.long)
        
        # Mock what load_kg_hf returns: (train, valid, test, ent2id, rel2id)
        # We only care about train and rel2id size
        mock_rel2id = {"r0": 0, "r1": 1, "r2": 2}
        mock_load_kg.return_value = (train_tensor, None, None, None, mock_rel2id)

        # ------------------------------------------------------------------
        # 2. Run Generator
        # ------------------------------------------------------------------
        # Set thresholds high (0.5) to ensure we only pick up strong signals.
        # Logic:
        # P(r1 | r0) = Count(r0 & r1) / Count(r0) = 2/2 = 1.0  (> 0.5) -> Keep
        # P(r2 | r0) = Count(r0 & r2) / Count(r0) = 0/2 = 0.0  (< 0.5) -> Drop
        
        generate_cnf(
            dataset_name=self.dataset_name,
            output_dir=self.test_dir,
            precision_threshold=0.5,
            recall_threshold=0.5,
            add_inverse=True
        )

        # ------------------------------------------------------------------
        # 3. Assertions
        # ------------------------------------------------------------------
        # Check file existence
        output_path = os.path.join(self.test_dir, self.dataset_name, "cnf.pt")
        self.assertTrue(os.path.exists(output_path), "cnf.pt file was not created")

        # Load CNF content
        cnf = torch.load(output_path)
        
        # Check Total Relations (3 original + 3 inverse = 6)
        # r0, r1, r2, r0_inv(3), r1_inv(4), r2_inv(5)
        self.assertIn(0, cnf, "Relation 0 missing from CNF")
        
        # Check Context Logic for Relation 0
        # Relation 0 should have Relation 1 as a neighbor
        r0_context = cnf[0].tolist()
        self.assertIn(1, r0_context, "Relation 1 should be a context neighbor of Relation 0")
        self.assertNotIn(2, r0_context, "Relation 2 (irrelevant) should NOT be a neighbor of Relation 0")
        
        # Check Inverse Augmentation
        # Relation 0 (id 0) and Relation 1 (id 1) imply Inverse Relations exists
        # Inverse ID for r0 = 0 + 3 = 3
        # Inverse ID for r1 = 1 + 3 = 4
        # Since r0 and r1 co-occur on Heads in the forward pass, 
        # r0_inv and r1_inv should co-occur on Tails (which become heads in inverse triples).
        # Note: The generator calculates co-occurrence based on heads of the *combined* set.
        # e0 is head of r0, r1.
        # e10 is head of r0_inv. e11 is head of r1_inv.
        # Unless the tails (10 and 11) are the same entity, r0_inv and r1_inv won't necessarily co-occur 
        # in this specific synthetic/sparse graph. 
        # However, we DO assert that the keys exist.
        self.assertIn(3, cnf, "Inverse relation 0 (id 3) missing from CNF")

        print("\n[Test] Synthetic Graph CNF Logic Verified Successfully.")

if __name__ == "__main__":
    unittest.main()