from eval_pipeline.prompts import (
    MULTI_TURN_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    render_prompt,
    render_system_prompt,
)
from eval_pipeline.schema import Comparison


def test_single_turn_prompt_shows_both_answers_plainly():
    comparison = Comparison(row_id="c0", prompt="what is 2+2?", response_a="4", response_b="five")
    text = render_prompt(comparison)
    assert "what is 2+2?" in text
    assert "[The Start of Assistant A's Answer]\n4" in text
    assert "[The Start of Assistant B's Answer]\nfive" in text
    assert "Conversation with User" not in text  # single-turn template only


def test_multi_turn_prompt_renders_each_sides_own_prior_answer():
    comparison = Comparison(
        row_id="c0", prompt="now about autumn", response_a="A's autumn haiku",
        response_b="B's autumn haiku",
        context_a=(("user", "write a haiku"), ("assistant", "A's haiku")),
        context_b=(("user", "write a haiku"), ("assistant", "B's haiku")),
    )
    text = render_prompt(comparison)
    assert "<|The Start of Assistant A's Conversation with User|>" in text
    assert "A's haiku" in text and "B's haiku" in text
    assert "now about autumn" in text
    assert text.index("Assistant A's Conversation") < text.index("Assistant B's Conversation")


def test_system_prompt_instructs_against_position_and_length_bias():
    assert "position" in SYSTEM_PROMPT.lower()
    assert "length" in SYSTEM_PROMPT.lower()
    assert "[[A]]" in SYSTEM_PROMPT and "[[B]]" in SYSTEM_PROMPT and "[[C]]" in SYSTEM_PROMPT


def test_multi_turn_system_prompt_focuses_the_judge_on_the_second_question():
    # The one substantive difference from SYSTEM_PROMPT, per MT-Bench's own
    # published "pair-v2-multi-turn" judge prompt: without this sentence, a
    # judge shown a multi-turn item has no instruction to isolate turn-2
    # quality from turn-1 quality.
    assert "second user question" in MULTI_TURN_SYSTEM_PROMPT
    assert "second user question" not in SYSTEM_PROMPT
    # Otherwise it carries the same guarantees as the single-turn prompt.
    assert "position" in MULTI_TURN_SYSTEM_PROMPT.lower()
    assert "length" in MULTI_TURN_SYSTEM_PROMPT.lower()
    assert "[[A]]" in MULTI_TURN_SYSTEM_PROMPT and "[[C]]" in MULTI_TURN_SYSTEM_PROMPT


def test_render_system_prompt_selects_by_whether_context_is_present():
    single_turn = Comparison(row_id="c0", prompt="q", response_a="a", response_b="b")
    assert render_system_prompt(single_turn) == SYSTEM_PROMPT

    multi_turn = Comparison(
        row_id="c0", prompt="q", response_a="a", response_b="b",
        context_a=(("user", "prior q"), ("assistant", "prior a")),
    )
    assert render_system_prompt(multi_turn) == MULTI_TURN_SYSTEM_PROMPT
