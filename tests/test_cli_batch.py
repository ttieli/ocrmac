"""Tests for batch/parallel CLI functionality."""

import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner

from ocrmac.cli import main
from ocrmac.formatter import OCRResult

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
TEST_IMAGE = os.path.join(THIS_FOLDER, "test.png")


def make_mock_result(source="test.png", text="Hello World"):
    """Create a mock OCRResult."""
    result = OCRResult(source=source)
    result.add_page(1, text, [(text, 0.99, [0.0, 0.0, 1.0, 1.0])])
    return result


class TestNoInput:
    """Test that no-input behavior is preserved."""

    def test_no_args_shows_guide(self):
        runner = CliRunner()
        result = runner.invoke(main, [])
        assert result.exit_code == 0
        assert "ocrmac" in result.output
        assert "基本用法" in result.output


class TestSingleInputBackwardCompat:
    """Test that single-input behavior is exactly the same as before."""

    @patch("ocrmac.cli.process_input")
    def test_single_file_stdout(self, mock_process):
        mock_process.return_value = [make_mock_result(TEST_IMAGE, "OCR Text")]
        runner = CliRunner()
        result = runner.invoke(main, [TEST_IMAGE, "--stdout"])
        assert result.exit_code == 0
        assert "OCR Text" in result.output
        # No batch separator for single input (markdown may contain --- as hr)
        assert "--- [1/" not in result.output
        mock_process.assert_called_once()

    @patch("ocrmac.cli.process_input")
    def test_single_file_output_file(self, mock_process):
        mock_process.return_value = [make_mock_result(TEST_IMAGE, "OCR Text")]
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "result.md")
            result = runner.invoke(main, [TEST_IMAGE, "-o", out_file])
            assert result.exit_code == 0
            assert os.path.exists(out_file)
            with open(out_file) as f:
                content = f.read()
            assert "OCR Text" in content

    @patch("ocrmac.cli.process_input")
    def test_single_file_auto_save(self, mock_process):
        mock_process.return_value = [make_mock_result(TEST_IMAGE, "OCR Text")]
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use a file in tmpdir so auto-save goes there
            tmp_img = os.path.join(tmpdir, "sample.png")
            # Create a dummy file so source_path.exists() is True
            with open(tmp_img, "w") as f:
                f.write("dummy")
            mock_process.return_value = [make_mock_result(tmp_img, "OCR Text")]
            result = runner.invoke(main, [tmp_img])
            assert result.exit_code == 0
            assert "Saved:" in result.output


class TestMultiInput:
    """Test multi-input parallel processing."""

    @patch("ocrmac.cli.process_input")
    def test_multi_file_stdout_with_separators(self, mock_process):
        """Multi-file stdout should have separators."""
        def side_effect(path, *args, **kwargs):
            return [make_mock_result(path, f"Text from {os.path.basename(path)}")]
        mock_process.side_effect = side_effect

        runner = CliRunner()
        result = runner.invoke(main, [TEST_IMAGE, TEST_IMAGE, "--stdout"])
        assert result.exit_code == 0
        assert "--- [1/2]" in result.output
        assert "--- [2/2]" in result.output
        assert "Text from test.png" in result.output

    @patch("ocrmac.cli.process_input")
    def test_multi_file_output_dir(self, mock_process):
        """Multi-file with -o should create files in directory."""
        def side_effect(path, *args, **kwargs):
            return [make_mock_result(path, f"Text from {path}")]
        mock_process.side_effect = side_effect

        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = os.path.join(tmpdir, "results")
            result = runner.invoke(main, [TEST_IMAGE, TEST_IMAGE, "-o", out_dir])
            assert result.exit_code == 0
            assert os.path.isdir(out_dir)
            # Should have saved files
            assert "Saved:" in result.output

    @patch("ocrmac.cli.process_input")
    def test_multi_file_preserves_order(self, mock_process):
        """Results should be in input order, not completion order."""
        call_count = [0]

        def side_effect(path, *args, **kwargs):
            call_count[0] += 1
            idx = call_count[0]
            return [make_mock_result(path, f"Result_{idx}")]
        mock_process.side_effect = side_effect

        runner = CliRunner()
        result = runner.invoke(main, [TEST_IMAGE, TEST_IMAGE, TEST_IMAGE, "--stdout"])
        assert result.exit_code == 0
        # Check that [1/3], [2/3], [3/3] appear in order
        lines = result.output
        pos1 = lines.find("[1/3]")
        pos2 = lines.find("[2/3]")
        pos3 = lines.find("[3/3]")
        assert pos1 < pos2 < pos3


class TestMultiInputErrorTolerance:
    """Test that one failure doesn't break others."""

    @patch("ocrmac.cli.process_input")
    def test_error_in_one_input_others_succeed(self, mock_process):
        """If one input fails, others should still produce output."""
        # Use a bad path that will always fail, alongside good mocked paths
        bad_path = "/nonexistent/bad_file.png"

        def side_effect(path, *args, **kwargs):
            if path == bad_path:
                raise ValueError("Simulated failure")
            return [make_mock_result(path, "OK")]
        mock_process.side_effect = side_effect

        runner = CliRunner()
        result = runner.invoke(main, [TEST_IMAGE, bad_path, TEST_IMAGE, "--stdout"])
        assert result.exit_code == 0  # Click doesn't use return value as exit code
        assert "[ERROR]" in result.output
        assert "Simulated failure" in result.output
        # Other results should still be present (3 inputs total)
        assert "[1/3]" in result.output
        assert "[3/3]" in result.output
        # Successful results should have OCR content
        assert "OK" in result.output


class TestBatchStdin:
    """Test --batch stdin mode."""

    @patch("ocrmac.cli.process_input")
    def test_batch_stdin(self, mock_process):
        mock_process.return_value = [make_mock_result(TEST_IMAGE, "Stdin OCR")]
        runner = CliRunner()
        result = runner.invoke(main, ["--batch", "--stdout"], input=f"{TEST_IMAGE}\n")
        assert result.exit_code == 0
        # Single input from stdin → no separator
        assert "Stdin OCR" in result.output

    @patch("ocrmac.cli.process_input")
    def test_batch_stdin_multi(self, mock_process):
        def side_effect(path, *args, **kwargs):
            return [make_mock_result(path, f"Text from {os.path.basename(path)}")]
        mock_process.side_effect = side_effect

        runner = CliRunner()
        result = runner.invoke(main, ["--batch", "--stdout"],
                               input=f"{TEST_IMAGE}\n{TEST_IMAGE}\n")
        assert result.exit_code == 0
        assert "--- [1/2]" in result.output
        assert "--- [2/2]" in result.output

    def test_batch_with_positional_args_errors(self):
        runner = CliRunner()
        result = runner.invoke(main, [TEST_IMAGE, "--batch"])
        assert result.exit_code != 0
        assert "cannot be used with positional arguments" in result.output

    @patch("ocrmac.cli.process_input")
    def test_batch_stdin_empty(self, mock_process):
        runner = CliRunner()
        result = runner.invoke(main, ["--batch"], input="")
        assert result.exit_code == 0
        # Should show usage guide
        assert "ocrmac" in result.output


class TestWorkersOption:
    """Test --workers option."""

    @patch("ocrmac.cli.process_input")
    @patch("ocrmac.cli.concurrent.futures.ThreadPoolExecutor")
    def test_workers_option_passed(self, mock_executor_class, mock_process):
        mock_process.return_value = [make_mock_result(TEST_IMAGE, "OK")]

        # Mock the executor
        mock_executor = MagicMock()
        mock_executor_class.return_value.__enter__ = MagicMock(return_value=mock_executor)
        mock_executor_class.return_value.__exit__ = MagicMock(return_value=False)

        # Make submit return futures
        mock_future = MagicMock()
        mock_future.result.return_value = (0, TEST_IMAGE, [make_mock_result(TEST_IMAGE)], None)
        mock_executor.submit.return_value = mock_future

        # Mock as_completed
        with patch("ocrmac.cli.concurrent.futures.as_completed", return_value=[mock_future, mock_future]):
            runner = CliRunner()
            result = runner.invoke(main, [TEST_IMAGE, TEST_IMAGE, "--stdout", "-w", "4"])

        # Verify ThreadPoolExecutor was called with max_workers=4
        mock_executor_class.assert_called_once_with(max_workers=4)


class TestHelpOption:
    """Test --help still works."""

    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "INPUT_PATH" in result.output
        assert "--batch" in result.output
        assert "--workers" in result.output
