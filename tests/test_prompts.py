from eval_pipeline.prompts import SYSTEM_PROMPT, render_prompt
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
