"""Toy auth: the X-User header names a user in data/users.yaml. Anything else is refused (fail closed)."""

from pathlib import Path

import yaml

ADMIN_GROUP = "admin"


class Users:
    def __init__(self, groups: dict[str, list[str]]):
        self._groups = groups

    @classmethod
    def load(cls, path: Path) -> "Users":
        return cls(yaml.safe_load(path.read_text()))

    def groups(self, user: str | None) -> list[str] | None:
        """The user's permission groups, or None if the user is missing or unknown."""
        if not user:
            return None
        return self._groups.get(user)
