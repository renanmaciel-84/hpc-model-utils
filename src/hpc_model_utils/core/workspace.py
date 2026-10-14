"""ADR-009/R23/R41/R123: the Workspace value and the .hpcmu layout."""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass
from pathlib import Path

from hpc_model_utils.core.errors import UsageError

_JOB_ID_PATTERN = re.compile(r"^[0-9]+$")
_INVALID_PATH_PATTERN = re.compile(r"[\s\x00-\x1f\x7f]")


class Phase(enum.StrEnum):
    MODEL = "model"
    FINALIZE = "finalize"


@dataclass(frozen=True, slots=True)
class Workspace:
    root: Path

    def __post_init__(self) -> None:
        if not self.root.is_absolute():
            raise ValueError(f"root must be absolute: {self.root}")

    @classmethod
    def at(cls, path: Path | str) -> Workspace:
        try:
            resolved = Path(path).resolve(strict=True)
        except OSError as exc:
            raise UsageError(f"workspace root does not exist: {path}") from exc
        if not resolved.is_dir():
            raise UsageError(f"workspace root is not a directory: {resolved}")
        if _INVALID_PATH_PATTERN.search(str(resolved)):
            raise UsageError(
                "workspace root contains whitespace or control "
                f"characters: {resolved}"
            )
        return cls(resolved)

    @property
    def assets(self) -> Path:
        return self.root / "assets"

    @property
    def eco_deck_path(self) -> Path:
        return self.root / "eco_deck.zip"

    @property
    def hpcmu_dir(self) -> Path:
        return self.root / ".hpcmu"

    @property
    def parent_dir(self) -> Path:
        return self.hpcmu_dir / "parent"

    @property
    def state_path(self) -> Path:
        return self.hpcmu_dir / "state.json"

    @property
    def finalize_path(self) -> Path:
        return self.hpcmu_dir / "finalize.json"

    @property
    def jobs_dir(self) -> Path:
        return self.hpcmu_dir / "jobs"

    @property
    def logs_dir(self) -> Path:
        return self.hpcmu_dir / "logs"

    @property
    def synthesis_log_path(self) -> Path:
        return self.logs_dir / "synthesis.out"

    @property
    def outputs_dir(self) -> Path:
        return self.hpcmu_dir / "outputs"

    @property
    def telemetry_dir(self) -> Path:
        return self.hpcmu_dir / "telemetry"

    @property
    def telemetry_outputs_dir(self) -> Path:
        return self.outputs_dir / "telemetry"

    @property
    def model_exit_path(self) -> Path:
        return self.jobs_dir / "model.exit"

    @property
    def legacy_status_path(self) -> Path:
        return self.root / "status.modelops"

    @property
    def legacy_metadata_path(self) -> Path:
        return self.root / "metadata.modelops"

    @property
    def has_state(self) -> bool:
        return self.state_path.is_file()

    def job_script(self, phase: Phase) -> Path:
        return self.jobs_dir / f"{phase}.sbatch"

    def log_path(self, phase: Phase, job_id: str) -> Path:
        if not _JOB_ID_PATTERN.fullmatch(job_id):
            raise ValueError(
                f"job_id must match {_JOB_ID_PATTERN.pattern}: {job_id!r}"
            )
        return self.logs_dir / f"{phase}-{job_id}.out"

    def log_pattern(self, phase: Phase) -> Path:
        return self.logs_dir / f"{phase}-%j.out"

    def ensure_layout(self) -> None:
        for path in (
            self.hpcmu_dir,
            self.jobs_dir,
            self.logs_dir,
            self.outputs_dir,
            self.telemetry_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def relative(self, path: Path) -> str:
        rel = path.relative_to(self.root)
        if ".." in rel.parts:
            raise ValueError(f"path escapes root {self.root}: {path}")
        return rel.as_posix()
