import json

import pytest

from eval_pipeline.judge import LLMJudge, PrecomputedJudge
from eval_pipeline.reward_hacking import (
    GamingExperiment,
    GamingResult,
    LengthDisclosedJudge,
    LengthNormalizedJudge,
    _length_disclosure_note,
    _truncate_to,
    pad_response,
    run_gaming_experiment,
)
from eval_pipeline.schema import Comparison


def row(row_id, response_a="a real answer", response_b="another real answer"):
    return Comparison(row_id=row_id, prompt="q", response_a=response_a, response_b=response_b)


def _section(user_prompt, label):
    # Must split on the full "[The Start of ...]" marker, not just the
    # trailing "Assistant X's Answer]\n" — that shorter pattern also matches
    # the *End* marker (which is immediately followed by "\n\n[The Start" for
    # slot A), silently appending a few stray characters onto slot A's
    # section and making an exact length comparison compare unequal boundary
    # artifacts rather than the two responses themselves.
    start = f"[The Start of Assistant {label}'s Answer]\n"
    end = f"\n[The End of Assistant {label}'s Answer]"
    return user_prompt.split(start)[1].split(end)[0]


def length_sensitive_backend(system_prompt, user_prompt):
    """A judge that (wrongly) rewards whichever response is longer."""
    a, b = _section(user_prompt, "A"), _section(user_prompt, "B")
    return "[[A]]" if len(a) > len(b) else ("[[B]]" if len(b) > len(a) else "[[C]]")


def content_based_backend(system_prompt, user_prompt):
    """A judge that ignores length and only looks for a 'CORRECT' marker."""
    a, b = _section(user_prompt, "A"), _section(user_prompt, "B")
    a_good, b_good = "CORRECT" in a, "CORRECT" in b
    if a_good == b_good:
        return "[[C]]"
    return "[[A]]" if a_good else "[[B]]"


# --- pad_response ----------------------------------------------------------

def test_pad_response_preserves_the_original_content():
    original = "the real substantive answer"
    padded = pad_response(original)
    assert original in padded
    assert len(padded) > len(original)


# --- GamingResult / GamingExperiment arithmetic -----------------------------

def test_flipped_to_padded_requires_a_genuine_flip():
    flipped = GamingResult("c0", baseline_winner="B", baseline_unstable=False,
                            padded_winner="A", padded_unstable=False)
    already_a = GamingResult("c1", baseline_winner="A", baseline_unstable=False,
                              padded_winner="A", padded_unstable=False)
    still_b = GamingResult("c2", baseline_winner="B", baseline_unstable=False,
                            padded_winner="B", padded_unstable=False)
    assert flipped.flipped_to_padded is True
    assert already_a.flipped_to_padded is False  # already won, not a flip
    assert still_b.flipped_to_padded is False


def test_unstable_results_are_excluded_from_usable():
    stable = GamingResult("c0", "B", False, "A", False)
    unstable = GamingResult("c1", "B", False, "A", True)
    experiment = GamingExperiment([stable, unstable])
    assert experiment.usable == [stable]


def test_padding_effect_and_flip_rate_on_a_worked_example():
    results = [
        GamingResult("c0", baseline_winner="B", baseline_unstable=False, padded_winner="A", padded_unstable=False),  # flips
        GamingResult("c1", baseline_winner="B", baseline_unstable=False, padded_winner="B", padded_unstable=False),  # doesn't
        GamingResult("c2", baseline_winner="A", baseline_unstable=False, padded_winner="A", padded_unstable=False),  # already A
        GamingResult("c3", baseline_winner="tie", baseline_unstable=False, padded_winner="A", padded_unstable=False),  # flips
    ]
    experiment = GamingExperiment(results)
    assert experiment.baseline_a_rate == pytest.approx(1 / 4)
    assert experiment.padded_a_rate == pytest.approx(3 / 4)
    assert experiment.padding_effect == pytest.approx(0.5)
    # eligible (baseline != A): c0, c1, c3 -> 2 of 3 flip
    assert experiment.flip_rate == pytest.approx(2 / 3)


def test_experiment_with_no_usable_results():
    experiment = GamingExperiment([GamingResult("c0", "B", True, "A", False)])
    assert experiment.usable == []
    assert experiment.baseline_a_rate is None
    assert experiment.padding_effect is None
    assert "no usable comparisons" in experiment.summary()


# --- run_gaming_experiment, against fake but content-driven judges ---------

def test_length_sensitive_judge_shows_a_real_padding_effect():
    judge = LLMJudge(length_sensitive_backend)
    rows = [row(f"c{i}") for i in range(6)]  # default response_b is the longer string
    experiment = run_gaming_experiment(judge, rows)

    assert experiment.baseline_a_rate == pytest.approx(0.0)  # B starts longer -> never "A" at baseline
    assert experiment.padded_a_rate == pytest.approx(1.0)    # padding makes A far longer -> always wins
    assert experiment.padding_effect == pytest.approx(1.0)
    assert experiment.flip_rate == pytest.approx(1.0)


def test_content_based_judge_shows_no_padding_effect():
    # Neither response contains "CORRECT" -> always a tie, padding changes nothing.
    judge = LLMJudge(content_based_backend)
    rows = [row(f"c{i}") for i in range(6)]
    experiment = run_gaming_experiment(judge, rows)

    assert experiment.padding_effect == pytest.approx(0.0)


def test_content_based_judge_is_not_fooled_even_when_padded_side_has_the_correct_answer():
    # response_a is genuinely correct; padding it shouldn't change that it wins,
    # and un-padded it already wins -> no *flip* (it was never a gaming case).
    judge = LLMJudge(content_based_backend)
    rows = [row("c0", response_a="the CORRECT answer", response_b="a wrong answer")]
    experiment = run_gaming_experiment(judge, rows)
    assert experiment.baseline_a_rate == pytest.approx(1.0)
    assert experiment.padded_a_rate == pytest.approx(1.0)
    assert experiment.flip_rate is None  # no eligible (non-A-baseline) cases


# --- _truncate_to ------------------------------------------------------------

def test_truncate_to_is_a_noop_when_already_short_enough():
    assert _truncate_to("short text", 100) == "short text"


def test_truncate_to_cuts_at_a_word_boundary_and_marks_truncation():
    text = "one two three four five six seven eight nine ten"
    truncated = _truncate_to(text, 20)
    assert truncated.endswith("...")
    assert len(truncated) <= 24  # 20 + "..." plus boundary slack
    assert not truncated[:-3].endswith(" ")  # trailing space stripped before "..."


# --- LengthNormalizedJudge: the mitigation actually mitigates --------------

def test_length_normalized_judge_neutralises_the_padding_effect():
    inner = LLMJudge(length_sensitive_backend)
    naive_experiment = run_gaming_experiment(inner, [row(f"c{i}") for i in range(6)])
    assert naive_experiment.padding_effect == pytest.approx(1.0)  # fully gameable

    mitigated = LengthNormalizedJudge(inner, tolerance=1.15)
    mitigated_experiment = run_gaming_experiment(mitigated, [row(f"c{i}") for i in range(6)])
    # Once truncated to near-equal length, the length-sensitive judge has
    # nothing to key off and should fall back to its length tie-break ("[[C]]").
    assert mitigated_experiment.padding_effect == pytest.approx(0.0)


def test_length_normalized_judge_leaves_the_shorter_response_untouched():
    judge = LengthNormalizedJudge(PrecomputedJudge.from_records([{"id": "c0", "winner": "A"}]))
    comparison = Comparison(row_id="c0", prompt="q", response_a="short", response_b="a " * 200)
    # PrecomputedJudge.verdict doesn't look at response text, so this only
    # tests that construction/wrapping doesn't crash on an extreme length gap.
    result = judge.verdict(comparison)
    assert result.winner == "A"


def test_length_normalized_judge_preserves_order_invariance_from_inner():
    order_invariant = PrecomputedJudge.from_records([{"id": "c0", "winner": "A"}])
    assert LengthNormalizedJudge(order_invariant).is_order_invariant is True

    not_invariant = LLMJudge(length_sensitive_backend)
    assert LengthNormalizedJudge(not_invariant).is_order_invariant is False


# --- _length_disclosure_note ------------------------------------------------

def test_no_note_when_lengths_are_within_threshold():
    assert _length_disclosure_note(100, 110, threshold=1.15) == ""


def test_note_fires_and_states_exact_lengths_past_the_threshold():
    note = _length_disclosure_note(100, 200, threshold=1.15)
    assert "100 characters" in note
    assert "200 characters" in note
    assert "not better for being longer" in note


def test_note_threshold_is_a_strict_boundary():
    # Exactly at the threshold: no note. Just past it: a note.
    assert _length_disclosure_note(100, 115, threshold=1.15) == ""
    assert _length_disclosure_note(100, 116, threshold=1.15) != ""


def test_note_handles_a_zero_length_response_without_crashing():
    assert _length_disclosure_note(0, 50, threshold=1.15) == ""


# --- LengthDisclosedJudge: content is never touched, only the prompt ------

def test_length_disclosed_judge_never_modifies_either_response():
    captured = {}

    def capturing_backend(system_prompt, user_prompt):
        captured["user_prompt"] = user_prompt
        return "[[A]]"

    original = Comparison(row_id="c0", prompt="q", response_a="short", response_b="a real answer " * 50)
    inner = LLMJudge(capturing_backend)
    LengthDisclosedJudge(inner, threshold=1.15).verdict(original)

    # The full, untruncated text of both responses must appear verbatim in
    # what the judge actually saw — this is the core guarantee that
    # distinguishes this mitigation from LengthNormalizedJudge.
    assert original.response_a in captured["user_prompt"]
    assert original.response_b in captured["user_prompt"]


def test_length_disclosed_judge_prepends_the_note_to_the_prompt_when_it_fires():
    captured = {}

    def capturing_backend(system_prompt, user_prompt):
        captured["user_prompt"] = user_prompt
        return "[[A]]"

    comparison = Comparison(row_id="c0", prompt="the actual question", response_a="x", response_b="y" * 100)
    LengthDisclosedJudge(LLMJudge(capturing_backend), threshold=1.15).verdict(comparison)

    assert "substantial length difference" in captured["user_prompt"]
    assert "the actual question" in captured["user_prompt"]
    # The note comes before the question in the rendered prompt.
    assert captured["user_prompt"].index("substantial length") < captured["user_prompt"].index("the actual question")


def test_length_disclosed_judge_adds_nothing_when_lengths_are_similar():
    captured = {}

    def capturing_backend(system_prompt, user_prompt):
        captured["user_prompt"] = user_prompt
        return "[[A]]"

    comparison = Comparison(row_id="c0", prompt="q", response_a="answer one", response_b="answer two")
    LengthDisclosedJudge(LLMJudge(capturing_backend), threshold=1.15).verdict(comparison)
    assert "substantial length difference" not in captured["user_prompt"]


def test_length_disclosed_judge_preserves_order_invariance_from_inner():
    order_invariant = PrecomputedJudge.from_records([{"id": "c0", "winner": "A"}])
    assert LengthDisclosedJudge(order_invariant).is_order_invariant is True

    not_invariant = LLMJudge(length_sensitive_backend)
    assert LengthDisclosedJudge(not_invariant).is_order_invariant is False


def test_length_disclosed_judge_a_fake_that_reads_the_note_can_neutralise_the_padding_effect():
    # Proves the wiring end-to-end: a judge that actually respects the
    # disclosed lengths (unlike length_sensitive_backend, which never reads
    # instructions) is no longer fooled by padding once wrapped. Whether a
    # *real* model reads the note is a separate, unverified question this
    # fake cannot answer — see REWARD_HACKING.md.
    def instruction_following_backend(system_prompt, user_prompt):
        if "substantial length difference" in user_prompt:
            return "[[C]]"  # declines to pick a side once warned of a length gap
        return "[[A]]" if len(user_prompt) > 0 else "[[B]]"  # otherwise arbitrary but stable

    inner = LLMJudge(instruction_following_backend)
    rows = [row(f"c{i}") for i in range(6)]
    disclosed = LengthDisclosedJudge(inner, threshold=1.15)
    experiment = run_gaming_experiment(disclosed, rows)
    # Padding inflates response_a well past the threshold, so the note fires
    # on every padded call and the fake backend ties rather than rewarding it.
    assert experiment.padded_a_rate == pytest.approx(0.0)


# --- main() entrypoint: refusal path only, never a real spend -------------

def test_main_refuses_a_run_it_cannot_confirm(monkeypatch, tmp_path):
    from eval_pipeline import reward_hacking

    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    policy_outputs = tmp_path / "policy_outputs.jsonl"
    policy_outputs.write_text(
        '{"id": "c0", "prompt": "q", "response_a": "a", "response_b": "b"}\n',
        encoding="utf-8",
    )
    exit_code = reward_hacking.main(["--policy-outputs", str(policy_outputs)])
    assert exit_code == 1


def test_main_prints_the_true_call_count_it_actually_makes(monkeypatch, tmp_path, capsys):
    # Regression test for a real bug found via code review: the printed
    # cost estimate said `len(sample) * 4` calls, undercounting because it
    # didn't account for every judge condition main() actually runs. main()
    # now runs run_gaming_experiment against THREE judges (naive,
    # length-normalized, length-disclosed), each judging every comparison
    # twice (baseline, padded) with 2 backend calls each (position swap) —
    # the true count is `* 12`. This asserts the printed number always
    # matches the real number of backend calls made, so that class of bug
    # can't recur silently in the one place in this codebase whose job is
    # showing a user a cost before spending their money.
    from eval_pipeline import providers, reward_hacking

    call_count = {"n": 0}

    def fake_backend_factory(model=None, cache=None):
        def fake_backend(system_prompt, user_prompt):
            call_count["n"] += 1
            return "[[A]]"
        return fake_backend

    monkeypatch.setattr(providers, "anthropic_judge_backend", fake_backend_factory)

    n = 3
    policy_outputs = tmp_path / "policy_outputs.jsonl"
    records = [
        {"id": f"c{i}", "prompt": f"q{i}", "response_a": "a", "response_b": "b"}
        for i in range(n)
    ]
    policy_outputs.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

    exit_code = reward_hacking.main([
        "--policy-outputs", str(policy_outputs), "--n", str(n), "--cache", "", "--yes",
    ])
    assert exit_code == 0

    printed = capsys.readouterr().out
    printed_n_calls = int(printed.split(" calls,")[0].split(", ")[-1])
    assert printed_n_calls == call_count["n"]  # the estimate must match reality
    assert call_count["n"] == n * 12  # the actual, ground-truth call count
