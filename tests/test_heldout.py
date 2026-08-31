"""The held-out guard.

Small surface, disproportionate consequence: every figure this project
publishes is measured on volume 2, and real-crop harvesting is the first
thing here that could quietly turn those pages into training data. The
failure would be silent -- nothing errors, the numbers just start
improving for the wrong reason.
"""

from __future__ import annotations

import pytest

from tetrak_hy_trainer import heldout

VOLUME_2 = "Ինդեքս:Հայկական Սովետական Հանրագիտարան (Soviet Armenian Encyclopedia) 2.djvu"
VOLUME_1 = "Ինդեքս:Հայկական Սովետական Հանրագիտարան (Soviet Armenian Encyclopedia) 1.djvu"
VOLUME_12 = "Ինդեքս:Հայկական Սովետական Հանրագիտարան (Soviet Armenian Encyclopedia) 12.djvu"


class TestIsHeldOut:
    def test_volume_2_is_held_out(self) -> None:
        assert heldout.is_held_out(VOLUME_2)

    def test_the_training_volumes_are_not(self) -> None:
        for volume in (1, 3, 4, 5, 6):
            title = VOLUME_1.replace("1.djvu", f"{volume}.djvu")
            assert not heldout.is_held_out(title), title

    def test_volume_12_is_not_mistaken_for_volume_2(self) -> None:
        """A substring check on "2.djvu" would match "12.djvu"; the
        suffix must be matched against the whole volume number."""
        assert not heldout.is_held_out(VOLUME_12)

    def test_surrounding_whitespace_does_not_smuggle_it_past(self) -> None:
        assert heldout.is_held_out(f"  {VOLUME_2}\n")


class TestAssertNotHeldOut:
    def test_a_training_volume_passes_quietly(self) -> None:
        assert heldout.assert_not_held_out(VOLUME_1) is None

    def test_volume_2_raises(self) -> None:
        with pytest.raises(heldout.HeldOutDataError):
            heldout.assert_not_held_out(VOLUME_2)

    def test_the_message_says_which_pages_and_where_it_came_from(self) -> None:
        """The operator needs to know what they pointed at and why it is
        refused, not merely that something failed."""
        with pytest.raises(heldout.HeldOutDataError) as caught:
            heldout.assert_not_held_out(VOLUME_2, source="runs/eval/ase-vol2/manifest.json")
        message = str(caught.value)
        assert "105-114" in message
        assert "runs/eval/ase-vol2/manifest.json" in message

    def test_the_message_does_not_suggest_working_around_it(self) -> None:
        with pytest.raises(heldout.HeldOutDataError) as caught:
            heldout.assert_not_held_out(VOLUME_2)
        assert "do not work around" in str(caught.value)
