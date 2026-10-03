"""Layer rules from docs/architecture.md, enforced on every CI run.

Each package may import only the packages listed for it. Third-party imports are
checked too, so a service can't quietly reach for boto3 or FastAPI."""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "poshak"

# internal packages each layer may depend on (itself always allowed)
ALLOWED_INTERNAL = {
    "domain": set(),
    "engine": {"domain"},
    "core": set(),
    "services": {"domain", "engine", "core"},
    "infrastructure": {"domain", "engine", "core"},
    "api": {"services", "domain", "core", "container"},
}
# third-party packages a layer must never import
_MODELS = {"torch", "transformers", "rembg"}
FORBIDDEN_EXTERNAL = {
    "domain": {"fastapi", "pydantic", "cv2", "boto3", "httpx", "sqlite3", *_MODELS},
    "engine": {"fastapi", "pydantic", "boto3", "httpx", "sqlite3", *_MODELS},
    "core": {"fastapi", "cv2", "boto3", "sqlite3", *_MODELS},
    "services": {"fastapi", "boto3", "httpx", "sqlite3", *_MODELS},
    "infrastructure": {"fastapi"},
    "api": {"cv2", "boto3", "sqlite3", *_MODELS},
}


def _imports(path: Path) -> set[str]:
    """Absolute dotted names imported by a module, with relative imports resolved."""
    package = path.relative_to(SRC.parent).with_suffix("").parts[:-1]
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = list(package[: len(package) - node.level + 1]) if node.level else []
            module = ".".join([*base, node.module] if node.module else base)
            names.add(module)
            if node.level and not node.module:  # `from .. import x` -> x may be a module
                names.update(f"{module}.{a.name}" for a in node.names)
    return names


def _cases():
    for layer in ALLOWED_INTERNAL:
        for path in sorted((SRC / layer).rglob("*.py")):
            yield pytest.param(layer, path, id=str(path.relative_to(SRC)))


@pytest.mark.parametrize(("layer", "path"), list(_cases()))
def test_layer_dependencies(layer, path):
    allowed = ALLOWED_INTERNAL[layer] | {layer}
    for name in _imports(path):
        parts = name.split(".")
        if parts[0] == "poshak" and len(parts) > 1:
            assert parts[1] in allowed, f"{layer} must not import {name}"
        else:
            assert parts[0] not in FORBIDDEN_EXTERNAL[layer], f"{layer} must not import {name}"
