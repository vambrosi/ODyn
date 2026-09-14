"""
The entry form, built from the annotation registry rather than written out.

Two things matter here. The form has to follow the registry, so that adding a
key is all it takes to collect something new. And a typed-in value has to be
read as the type the registry declared, or refused with a message someone can
act on -- the alternative is a string where a number belongs, found months
later during analysis.
"""

import pytest

from odyn import Database
from odyn.app.fields import Field, form_fields, missing_required


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path, project="test")


# --------------------------------------------------------------------------- #
# Following the registry
# --------------------------------------------------------------------------- #


def test_the_form_is_built_from_the_registry(db):
    fields = form_fields(db, "session")
    keys = {field.key for field in fields}

    assert "mouse_weight_g" in keys
    assert "goal" in keys

    # Experiment keys belong on the experiment form, not this one.
    assert "fov_depth_um" not in keys


def test_a_new_key_appears_without_touching_the_app(db):
    """That is the point of the registry: collecting something new is a row."""
    db.add_annotation_key(
        applies_to="session",
        key="room_temperature_c",
        label="Room temperature",
        value_type="real",
        description="Temperature in the rig room.",
        unit="C",
    )

    field = _by_key(form_fields(db, "session"), "room_temperature_c")

    assert field.prompt == "Room temperature (C)"
    assert field.value_type == "real"


def test_a_retired_key_is_not_offered(db):
    """Still readable in the database, but nothing new goes under it."""
    db.retire_annotation_key(applies_to="session", key="headplate")

    assert _by_key(form_fields(db, "session"), "headplate") is None


def test_required_carries_through(db):
    fields = form_fields(db, "session")

    assert _by_key(fields, "mouse_weight_g").required
    assert not _by_key(fields, "goal").required


def test_a_note_asks_for_a_box_rather_than_a_line(db):
    """`long_text` is stored like text; it only says how much room it needs."""
    fields = form_fields(db, "session")

    assert _by_key(fields, "note").value_type == "long_text"
    assert _by_key(fields, "goal").value_type == "text"
    assert _by_key(fields, "flag").value_type == "boolean"


def test_enum_options_come_from_the_registry(db):
    field = _by_key(form_fields(db, "experiment"), "depth_class")

    assert set(field.options) == {"superficial", "deep"}


def test_a_target_with_no_keys_is_an_empty_form(db):
    assert form_fields(db, "nothing_like_this") == []


def _by_key(fields, key):
    return next((field for field in fields if field.key == key), None)


# --------------------------------------------------------------------------- #
# Reading what was typed
# --------------------------------------------------------------------------- #


def test_a_blank_box_means_not_recorded(db):
    """Not an empty value: nobody answered, and that is worth distinguishing."""
    field = _by_key(form_fields(db, "session"), "mouse_weight_g")

    assert field.parse("") is None
    assert field.parse("   ") is None
    assert field.parse(None) is None


@pytest.mark.parametrize(
    "value_type, written, expected",
    [
        ("text", " a note ", "a note"),
        ("integer", "10", 10),
        ("real", "25.1", 25.1),
        ("real", "25", 25.0),
        ("boolean", "yes", True),
        ("boolean", "N", False),
        ("date", "2026-07-06", "2026-07-06"),
    ],
)
def test_values_are_read_as_their_type(value_type, written, expected):
    field = Field("k", "Field", value_type, "")

    assert field.parse(written) == expected


@pytest.mark.parametrize(
    "value_type, written, says",
    [
        ("integer", "10.5", "whole number"),
        ("integer", "abc", "whole number"),
        ("real", "heavy", "a number"),
        ("boolean", "right side down slightly", "yes or no"),
        ("date", "06/07/2026", "YYYY-MM-DD"),
        ("date", "2026-07-32", "YYYY-MM-DD"),
    ],
)
def test_what_cannot_be_read_is_refused_by_name(value_type, written, says):
    """The message goes straight on screen, so it names the field and the text."""
    field = Field("k", "Mouse weight", value_type, "")

    with pytest.raises(ValueError, match=says):
        field.parse(written)

    with pytest.raises(ValueError, match="Mouse weight"):
        field.parse(written)


def test_an_enum_is_matched_without_case():
    """A dropdown and a typed answer have to agree."""
    field = Field("k", "Depth class", "enum", "", options=("superficial", "deep"))

    assert field.parse("DEEP") == "deep"

    with pytest.raises(ValueError, match="one of"):
        field.parse("middling")


# --------------------------------------------------------------------------- #
# Nudging rather than enforcing
# --------------------------------------------------------------------------- #


def test_missing_required_lists_what_is_still_needed(db):
    fields = form_fields(db, "session")

    missing = missing_required(fields, {"goal": "10x pre/post ket/xyl"})

    assert "mouse_weight_g" in {field.key for field in missing}


def test_nothing_is_missing_once_it_is_filled_in(db):
    fields = form_fields(db, "session")
    values = {field.key: "anything" for field in fields if field.required}

    assert missing_required(fields, values) == []
