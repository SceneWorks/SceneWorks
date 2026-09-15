import importlib.util
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).with_name("starvector-terminal-metrics.py")
SPEC = importlib.util.spec_from_file_location("starvector_terminal_metrics", SCRIPT)
METRICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(METRICS)


class ParityOutcomeTests(unittest.TestCase):
    def test_explicit_generation_limit_is_a_typed_rejection(self):
        native = {
            "accepted": False,
            "outcome": "rejected",
            "finishReason": "token_limit",
            "rejectionStage": "generation_limit",
            "rejectionCode": "token_limit",
            "rejectionReason": "native StarVector stopped at the token_limit",
        }
        self.assertEqual(
            METRICS.native_parity_outcome(native),
            ("rejected", "generation_limit", "token_limit", native["rejectionReason"]),
        )

    def test_all_oracle_generation_limits_equal_native_normalized_outcomes(self):
        for code in ["token_limit", "byte_limit", "wall_time_limit"]:
            native = {"accepted": False, "outcome": "rejected", "finishReason": code,
                      "rejectionStage": "generation_limit", "rejectionCode": code,
                      "rejectionReason": f"native StarVector stopped at the {code}"}
            outcome, stage, normalized, _ = METRICS.native_parity_outcome(native)
            self.assertEqual((outcome, stage, normalized),
                             ("rejected", "generation_limit",
                              METRICS.normalized_rejection(code, "generation_limit")))

    def test_generic_or_inconsistent_failures_cannot_pass_as_rejections(self):
        invalid = [
            {"accepted": False, "finishReason": "token_limit"},
            {"accepted": False, "outcome": "rejected", "rejectionStage": "infrastructure"},
            {"accepted": False, "outcome": "rejected", "finishReason": "cancelled",
             "rejectionStage": "generation_limit", "rejectionCode": "cancelled",
             "rejectionReason": "native StarVector stopped at the cancelled"},
            {"accepted": False, "outcome": "rejected", "finishReason": "byte_limit",
             "rejectionStage": "generation_limit", "rejectionCode": "token_limit",
             "rejectionReason": "native StarVector stopped at the token_limit"},
        ]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(SystemExit):
                METRICS.native_parity_outcome(value)


if __name__ == "__main__":
    unittest.main()
