"""
What the entry form shows, taken from the annotation registry.

The form is not written out widget by widget. `annotation_keys` already says
what may be recorded about a session or an experiment, with a label, a type, a
unit and whether it is required -- so the form is built from it. Registering a
new key with `Database.add_annotation_key` makes it appear, and retiring one
makes it disappear, without touching the app.

Fields come out in the order the registry holds them, which is the order
`create.sql` seeds them in, so that reads as the order of the form.

**USAGE**
```python
for field in form_fields(db, "session"):
    print(field.label, field.required)

field.parse("25.1")     # 25.1, as the registry's type
field.parse("heavy")    # raises ValueError naming the field
```

This module has no Qt in it: what the form contains and how a typed value is
read are decided here, and only the drawing is left to the widgets.
"""

from __future__ import annotations

import json

from dataclasses import dataclass
from datetime import datetime

# Types the registry can declare, and how each reads a typed-in string.
# `long_text` is stored like `text`; it asks for a box rather than a line.
TEXT, LONG_TEXT, INTEGER, REAL, BOOLEAN, DATE, ENUM = (
    "text",
    "long_text",
    "integer",
    "real",
    "boolean",
    "date",
    "enum",
)

TRUE_WORDS = ("y", "yes", "true", "1")
FALSE_WORDS = ("n", "no", "false", "0")


@dataclass(frozen=True)
class Field:
    """One thing the form asks for, and how to read what was typed."""

    key: str
    label: str
    value_type: str
    description: str
    unit: None | str = None
    required: bool = False
    options: tuple[str, ...] = ()

    @property
    def prompt(self) -> str:
        """The label as the form shows it, with the unit if there is one."""
        return f"{self.label} ({self.unit})" if self.unit else self.label

    def parse(self, written: str):
        """
        Read what someone typed, as the type the registry declared.

        Returns `None` for a blank box, which means "not recorded" rather than
        an empty value. Raises `ValueError` naming the field when the text does
        not read as its type, so the message can go straight on screen.
        """
        text = "" if written is None else str(written).strip()

        if not text:
            return None

        if self.value_type in (TEXT, LONG_TEXT):
            return text

        if self.value_type == INTEGER:
            return _whole_number(text, self)

        if self.value_type == REAL:
            return _number(text, self)

        if self.value_type == BOOLEAN:
            return _yes_or_no(text, self)

        if self.value_type == DATE:
            return _date(text, self)

        if self.value_type == ENUM:
            return _one_of(text, self)

        raise ValueError(
            f"{self.label} is registered as '{self.value_type}', which the form "
            f"does not know how to read."
        )


def form_fields(db, applies_to: str) -> list[Field]:
    """
    Every field the form should show for a session, experiment, program, and so on.

    Retired keys are left out: they are still readable in the database, but
    nothing new should be written under them.
    """
    registry = db.annotation_keys

    if applies_to not in registry.index.get_level_values("applies_to"):
        return []

    fields = []

    for key, row in registry.loc[applies_to].iterrows():
        if bool(row["retired"]):
            continue

        fields.append(
            Field(
                key=str(key),
                label=str(row["label"]),
                value_type=str(row["value_type"]),
                description=str(row["description"]),
                unit=None if row["unit"] is None else str(row["unit"]),
                required=bool(row["required"]),
                options=_options(row["allowed_values"]),
            )
        )

    return fields


def missing_required(fields: list[Field], values: dict) -> list[Field]:
    """
    The required fields nothing has been entered for.

    Shown as a nudge rather than enforced: a session can be submitted without
    them, and `Database.missing_annotations` will keep asking afterwards.
    """
    return [
        field
        for field in fields
        if field.required and values.get(field.key) in (None, "", [])
    ]


def _options(allowed) -> tuple[str, ...]:
    """The choices of an enum key, which the registry stores as JSON."""
    if allowed is None or allowed == "":
        return ()

    if isinstance(allowed, (list, tuple)):
        return tuple(str(option) for option in allowed)

    return tuple(str(option) for option in json.loads(allowed))


def _whole_number(text: str, field: Field) -> int:
    try:
        return int(text)
    except ValueError:
        raise ValueError(f"{field.label} is a whole number, not {text!r}.") from None


def _number(text: str, field: Field) -> float:
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"{field.label} is a number, not {text!r}.") from None


def _yes_or_no(text: str, field: Field) -> bool:
    said = text.lower()

    if said in TRUE_WORDS:
        return True

    if said in FALSE_WORDS:
        return False

    raise ValueError(f"{field.label} is yes or no, not {text!r}.")


def _date(text: str, field: Field) -> str:
    try:
        return datetime.strptime(text, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        raise ValueError(
            f"{field.label} is a date written 'YYYY-MM-DD', not {text!r}."
        ) from None


def _one_of(text: str, field: Field) -> str:
    # Matched without case so a dropdown and a typed answer agree.
    for option in field.options:
        if option.lower() == text.lower():
            return option

    raise ValueError(f"{field.label} is one of {list(field.options)}, not {text!r}.")
