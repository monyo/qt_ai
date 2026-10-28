import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import premarket_main


class SkipQualitativeTests(unittest.TestCase):
    def test_skip_does_not_run_generator_or_create_qualitative_file(self):
        with tempfile.TemporaryDirectory(prefix="hermes-verify-data-") as temp_dir:
            output_path = Path(temp_dir) / "qualitative.json"
            with patch.object(premarket_main, "generate_qualitative_data") as generate:
                result = premarket_main._ensure_qualitative_data(
                    {"date": "2026-10-06"}, output_path, skip_qualitative=True,
                )

            self.assertTrue(result)
            self.assertFalse(output_path.exists())
            generate.assert_not_called()

    def test_cli_skip_flag_disables_email_and_passes_skip_mode(self):
        with patch.object(premarket_main, "run_premarket") as run, patch.object(
            sys, "argv", ["premarket_main.py", "--skip-qualitative"],
        ):
            premarket_main.main()

        run.assert_called_once_with(
            scan_tw=False, send_email=False, skip_qualitative=True,
        )


if __name__ == "__main__":
    unittest.main()
