import unittest
import torch
import os
import shutil
import tempfile
from unittest.mock import patch

# Import the model class and the base registry
from models.compgcn_cp import CompGCN_TransE
from models.registry import get_model_class

class TestCompGCNCP(unittest.TestCase):
    def setUp(self):
        # 1. Create a temporary directory to mimic the data folder
        self.test_dir = tempfile.mkdtemp()
        self.dataset_name = "Test-Synthetic"
        self.dataset_path = os.path.join(self.test_dir, self.dataset_name)
        os.makedirs(self.dataset_path, exist_ok=True)
        
        # 2. Create a dummy CNF file
        # Relation 0 has context neighbor Relation 1
        # Relation 1 has context neighbor Relation 0
        cnf_data = {
            0: torch.tensor([1], dtype=torch.long),
            1: torch.tensor([0], dtype=torch.long)
        }
        torch.save(cnf_data, os.path.join(self.dataset_path, "cnf.pt"))

        # 3. Define Model Hyperparameters
        self.nentity = 5
        self.nrelation = 2  # Real relations 0, 1. Inverse will be 2, 3.
        self.hidden_dim = 16
        self.gamma = 12.0
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch('models.compgcn_cp.load_kg_hf')
    def test_initialization_and_forward(self, mock_load_kg):
        """
        Verifies that CompGCN_TransE initializes, builds the graph, 
        and performs a forward pass successfully.
        """
        # ------------------------------------------------------------------
        # 1. Mock Data Loading
        # ------------------------------------------------------------------
        # Synthetic Triples: (h, r, t)
        # 0 -> r0 -> 1
        # 1 -> r1 -> 2
        # 2 -> r0 -> 3
        # 3 -> r1 -> 4
        triples = [
            [0, 0, 1],
            [1, 1, 2],
            [2, 0, 3],
            [3, 1, 4]
        ]
        train_ids = torch.tensor(triples, dtype=torch.long)
        
        # load_kg_hf returns: (train, valid, test, ent2id, rel2id)
        mock_load_kg.return_value = (train_ids, None, None, {}, {})

        # ------------------------------------------------------------------
        # 2. Initialize Model
        # ------------------------------------------------------------------
        # We pass data_path explicitly to point to our temp dir
        model = CompGCN_TransE(
            nentity=self.nentity,
            nrelation=self.nrelation,
            base_dim=self.hidden_dim,
            gamma=self.gamma,
            data_path=self.test_dir,
            dataset=self.dataset_name
        )
        
        # Assert Graph Construction
        # 4 original edges + 4 inverse edges = 8 edges
        self.assertEqual(model.edge_index.shape[1], 8, "Graph should have 8 edges (4 forward + 4 inverse)")
        self.assertTrue(hasattr(model, 'cnf'), "Model should have loaded CNF")
        self.assertIn(0, model.cnf, "CNF should contain relation 0")

        # ------------------------------------------------------------------
        # 3. Test Forward Pass (Single Mode)
        # ------------------------------------------------------------------
        # Sample: (h=0, r=0, t=1)
        sample = torch.tensor([[0, 0, 1]], dtype=torch.long)
        
        # Run Forward
        score = model(sample, mode='single')
        
        # Check Output
        self.assertEqual(score.shape, (1, 1), "Score shape mismatch for single mode")
        
        # ------------------------------------------------------------------
        # 4. Test Forward Pass (Tail Batch)
        # ------------------------------------------------------------------
        # Predicting t given (h, r). Anchor is h.
        # Head Part: (h=0, r=0)
        # Tail Part: (t=1, t=2, t=3) [Candidates]
        
        # In this codebase, the trainer typically passes (pos, neg, ...)
        # KGModel._index_full handles 'tail-batch' where sample = (head_part, tail_part)
        # head_part is (B, 2) or (B, 3) used for context? 
        # Usually head_part stores [h, r, ...].
        
        batch_size = 2
        
        # Batch of 2 queries: (0, 0, ?) and (3, 1, ?)
        head_part = torch.tensor([
            [0, 0, 0], # h=0, r=0
            [3, 1, 0]  # h=3, r=1
        ], dtype=torch.long)
        
        # Candidates (indices)
        tail_part = torch.tensor([
            [1, 2, 3, 4], # Candidates for query 1
            [0, 1, 2, 3]  # Candidates for query 2
        ], dtype=torch.long)
        
        scores = model((head_part, tail_part), mode='tail-batch')
        
        self.assertEqual(scores.shape, (2, 4), "Score shape mismatch for tail-batch")
        
        # ------------------------------------------------------------------
        # 5. Test Forward Pass (Head Batch)
        # ------------------------------------------------------------------
        # Predicting h given (r, t). Anchor is t.
        # Queries: (?, r=0, t=1) and (?, r=1, t=4)
        tail_part = torch.tensor([
            [0, 0, 1], # r=0, t=1
            [0, 1, 4]  # r=1, t=4
        ], dtype=torch.long)
        
        head_part = torch.tensor([
            [0, 2, 3], 
            [1, 2, 3]
        ], dtype=torch.long)
        
        scores = model((tail_part, head_part), mode='head-batch')
        self.assertEqual(scores.shape, (2, 3), "Score shape mismatch for head-batch")

        print("\n[Test] CompGCN_TransE Forward Pass Successful.")

if __name__ == "__main__":
    unittest.main()