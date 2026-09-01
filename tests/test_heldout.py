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


class TestWorkRegistry:
    """Brief 012's widening: per-work held-out pages for non-ASE sources."""

    def test_a_work_with_no_entry_has_no_held_out_pages(self) -> None:
        assert heldout.held_out_pages("Ինդեքս:Faustus of Byzantium.djvu") is None

    def test_registered_pages_are_held_out(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(heldout.WORK_PAGES, "Faustus of Byzantium", frozenset({40, 41}))
        title = "Ինդեքս:Faustus of Byzantium, History of Armenia, 1968.djvu"
        assert heldout.page_is_held_out(title, 40)
        assert not heldout.page_is_held_out(title, 42)

    def test_every_ase_vol2_page_is_held_out_without_an_entry(self) -> None:
        title = "Ինդեքս:Հայկական Սովետական Հանրագիտարան (Soviet Armenian Encyclopedia) 2.djvu"
        assert heldout.page_is_held_out(title, 1)
        assert heldout.page_is_held_out(title, 700)

    def test_volume_2_of_another_work_is_not_condemned(self) -> None:
        """The whole-volume rule is the ASE's, not a rule about the digit 2."""
        assert not heldout.is_held_out("Ինդեքս:Nar-Dos, Collected works, vol. 2.djvu")

    def test_new_ase_volumes_are_not_held_out(self) -> None:
        for volume in (7, 9, 13):
            title = (
                "Ինդեքս:Հայկական Սովետական Հանրագիտարան "
                f"(Soviet Armenian Encyclopedia) {volume}.djvu"
            )
            assert not heldout.page_is_held_out(title, 100)
