from unittest.mock import Mock

from fire import Fire

from nxontology_ml.commands import cli


def test_cli() -> None:
    mock_fire = Mock(spec=Fire)
    cli(fire_fn=mock_fire)
    mock_fire.assert_called_once_with({})
