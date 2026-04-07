"""Tests for src/utils/config.py — written BEFORE implementation (TDD RED phase)."""

import pytest

# ---------------------------------------------------------------------------
# NOTE: These imports will FAIL until src/utils/config.py is implemented.
# That failure is intentional: it confirms we are in the RED state.
# ---------------------------------------------------------------------------
from src.utils.config import load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_CONFIG_PATH = "configs/data_config.yaml"


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestLoadValidConfig:
    """load_config returns a dict when given a well-formed YAML file."""

    def test_returns_dict(self):
        result = load_config(VALID_CONFIG_PATH)
        assert isinstance(result, dict), "load_config must return a dict"

    def test_contains_catalog_url(self):
        result = load_config(VALID_CONFIG_PATH)
        assert "catalog_url" in result, "Returned dict must contain 'catalog_url'"

    def test_catalog_url_is_string(self):
        result = load_config(VALID_CONFIG_PATH)
        assert isinstance(result["catalog_url"], str)
        assert result["catalog_url"].startswith("http")


class TestConfigValueTypes:
    """Parsed YAML values have the correct Python types."""

    def test_image_size_is_int(self):
        result = load_config(VALID_CONFIG_PATH)
        assert isinstance(result["image_size"], int), (
            f"image_size should be int, got {type(result['image_size'])}"
        )

    def test_sigma_clip_threshold_is_float(self):
        result = load_config(VALID_CONFIG_PATH)
        assert isinstance(result["sigma_clip_threshold"], float), (
            f"sigma_clip_threshold should be float, got {type(result['sigma_clip_threshold'])}"
        )

    def test_phase_bins_is_int(self):
        result = load_config(VALID_CONFIG_PATH)
        assert isinstance(result["phase_bins"], int)


class TestMissingRequiredKey:
    """load_config raises when a required key is absent from the config dict."""

    def test_raises_on_missing_catalog_url(self, tmp_path):
        """A YAML file without catalog_url must raise KeyError or ValueError."""
        config_file = tmp_path / "bad_config.yaml"
        config_file.write_text("image_size: 64\nsigma_clip_threshold: 5.0\n")

        with pytest.raises((KeyError, ValueError)):
            load_config(str(config_file), required_keys=["catalog_url"])

    def test_raises_on_multiple_missing_keys(self, tmp_path):
        """All missing required keys must trigger an error."""
        config_file = tmp_path / "empty_config.yaml"
        config_file.write_text("placeholder: true\n")

        with pytest.raises((KeyError, ValueError)):
            load_config(
                str(config_file),
                required_keys=["catalog_url", "image_size", "sigma_clip_threshold"],
            )

    def test_passes_when_all_required_keys_present(self, tmp_path):
        """No error when every required key exists."""
        config_file = tmp_path / "ok_config.yaml"
        config_file.write_text("catalog_url: https://example.com\nimage_size: 64\n")

        result = load_config(
            str(config_file),
            required_keys=["catalog_url", "image_size"],
        )
        assert result["catalog_url"] == "https://example.com"

    def test_no_required_keys_by_default(self, tmp_path):
        """Calling load_config without required_keys never raises for content reasons."""
        config_file = tmp_path / "minimal.yaml"
        config_file.write_text("foo: bar\n")

        result = load_config(str(config_file))
        assert result == {"foo": "bar"}


class TestEdgeCases:
    """Edge cases: nonexistent files, empty files, non-YAML content."""

    def test_missing_file_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_config("configs/does_not_exist.yaml")

    def test_empty_yaml_returns_empty_dict_or_raises(self, tmp_path):
        """An empty YAML file should either return {} or raise — not silently return None."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")

        # Acceptable outcomes: empty dict, or a ValueError explaining the file is empty.
        try:
            result = load_config(str(config_file))
            assert result == {} or isinstance(result, dict)
        except ValueError:
            pass  # also acceptable

    def test_invalid_yaml_raises_value_error(self, tmp_path):
        """Malformed YAML must raise ValueError (not crash with an internal yaml exception)."""
        config_file = tmp_path / "broken.yaml"
        config_file.write_text("key: [unclosed bracket\n")

        with pytest.raises((ValueError, Exception)):
            load_config(str(config_file))
