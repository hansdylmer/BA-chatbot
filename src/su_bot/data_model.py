from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Iterable, Iterator
from uuid import uuid5, NAMESPACE_URL

from pydantic import BaseModel, Field, HttpUrl, field_validator, computed_field, RootModel


def _collapse_lines(text: str) -> str:
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


class Section(BaseModel):
    heading: str = Field(default="(ingen overskrift)", description="Sektionens overskrift")
    content: str = Field(..., min_length=1, description="Sektionens fulde brødtekst")

    @field_validator("heading", mode="before")
    @classmethod
    def normalize_heading(cls, v: str | None) -> str:
        v = (v or "").strip()
        if not v:
            return "(ingen overskrift)"
        return _collapse_lines(v)


class Document(BaseModel):
    link: HttpUrl = Field(..., description="Kildens URL")
    title: str = Field(..., min_length=1)
    sections: List[Section] = Field(..., min_items=1)

    @field_validator("sections")
    @classmethod
    def ensure_nonempty_sections(cls, sections: List[Section]) -> List[Section]:
        if not sections:
            raise ValueError("Document must contain at least én sektion.")
        return sections

    @computed_field  # type: ignore[prop-decorator]
    @property
    def doc_id(self) -> str:
        return str(uuid5(NAMESPACE_URL, f"{self.link}::{self.title}"))


class Corpus(RootModel[List[Document]]):
    """Simple wrapper to preserve typing and helpers."""

    def __iter__(self) -> Iterator[Document]:
        return iter(self.root)

    def __len__(self) -> int:
        return len(self.root)

    def documents(self) -> List[Document]:
        return self.root


class Lang(str, Enum):
    da = "da"
    en = "en"


class HQERecord(BaseModel):
    doc_id: str
    section_index: int
    language: Lang
    gen_model: str
    prompt_version: str
    generated_at: str
    questions: List[str]
    meta: Dict[str, Any]


@dataclass(slots=True)
class Budget:
    small: int
    medium: int
    large: int

    def bounds(self) -> Iterable[int]:
        return (self.small, self.medium, self.large)

