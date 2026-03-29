from __future__ import annotations


class SharedContextStore:
    def __init__(self):
        self._store: dict[str, object] = {}

    def set(self, key: str, value: object) -> None:
        self._store[key] = value

    def get(self, key: str, default: object | None = None) -> object | None:
        return self._store.get(key, default)

    def snapshot(self) -> dict[str, object]:
        return dict(self._store)
