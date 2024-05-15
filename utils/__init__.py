"""
This module is for misc stuff
"""

from pathlib import Path


def get_project_root() -> Path:
    """
    get root directory of project
    :return: the root directory
    """
    return Path(__file__).parent.parent.resolve()
