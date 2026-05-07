import unittest

from utils.runtime import model_helpers


class ModelHelpersScalarNormalizationTests(unittest.TestCase):
    def test_extract_transformer_config_normalizes_moe_numeric_strings(self) -> None:
        config = {
            "TransformerConfig": {
                "num_layers": "4",
                "hidden_size": "512",
                "ffn_hidden_size": "2048",
                "num_attention_heads": "8",
                "num_query_groups": "4",
                "num_moe_experts": "8",
                "moe_router_topk": "2",
                "n_shared_experts": "1",
                "moe_ffn_hidden_size": "1024",
                "moe_aux_loss_coeff": "0.01",
                "layernorm_epsilon": "1e-6",
            }
        }

        normalized = model_helpers.extract_transformer_config_from_yaml(config)

        self.assertEqual(normalized["num_layers"], 4)
        self.assertEqual(normalized["hidden_size"], 512)
        self.assertEqual(normalized["ffn_hidden_size"], 2048)
        self.assertEqual(normalized["num_attention_heads"], 8)
        self.assertEqual(normalized["num_query_groups"], 4)
        self.assertEqual(normalized["num_moe_experts"], 8)
        self.assertEqual(normalized["moe_router_topk"], 2)
        self.assertEqual(normalized["n_shared_experts"], 1)
        self.assertEqual(normalized["moe_ffn_hidden_size"], 1024)
        self.assertAlmostEqual(normalized["moe_aux_loss_coeff"], 0.01)
        self.assertAlmostEqual(normalized["layernorm_epsilon"], 1e-6)


if __name__ == "__main__":
    unittest.main()
