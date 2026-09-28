"""
The judge prompt, based on MT-Bench's own published pair-wise judge prompts
(Zheng et al. 2023, arXiv:2306.05685) — the point is to compare our judge's
agreement against MT-Bench's own recorded human/GPT-4 verdicts on identical
prompts, which only means something if the prompt itself matches theirs.
"""

from __future__ import annotations

from .schema import Comparison

SYSTEM_PROMPT = (
    "Please act as an impartial judge and evaluate the quality of the "
    "responses provided by two AI assistants to the user question displayed "
    "below. You should choose the assistant that follows the user's "
    "instructions and answers the user's question better. Your evaluation "
    "should consider factors such as the helpfulness, relevance, accuracy, "
    "depth, creativity, and level of detail of their responses. Begin your "
    "evaluation by comparing the two responses and provide a short "
    "explanation. Avoid any position biases and ensure that the order in "
    "which the responses were presented does not influence your decision. "
    "Do not allow the length of the responses to influence your evaluation. "
    "Do not favor certain names of the assistants. Be as objective as "
    "possible. After providing your explanation, output your final verdict "
    "by strictly following this format: \"[[A]]\" if assistant A is better, "
    "\"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie."
)

# MT-Bench's own multi-turn judge prompt (FastChat's "pair-v2-multi-turn",
# judge_prompts.jsonl) is not just a different template — it is a different
# system prompt, differing from SYSTEM_PROMPT by exactly one inserted
# sentence: "You should focus on who provides a better answer to the second
# user question." Without it, a judge shown a multi-turn item has no
# instruction to isolate turn-2 quality from turn-1 quality, and may score
# the conversation holistically instead — a different criterion than what
# MT-Bench's own recorded human/GPT-4 labels for that item represent.
MULTI_TURN_SYSTEM_PROMPT = (
    "Please act as an impartial judge and evaluate the quality of the "
    "responses provided by two AI assistants to the user questions. You "
    "should choose the assistant that follows the user's instructions and "
    "answers the user's questions better. Your evaluation should consider "
    "factors such as the helpfulness, relevance, accuracy, depth, "
    "creativity, and level of detail of their responses. You should focus "
    "on who provides a better answer to the second user question. Begin "
    "your evaluation by comparing the responses of the two assistants and "
    "provide a short explanation. Avoid any position biases and ensure "
    "that the order in which the responses were presented does not "
    "influence your decision. Do not allow the length of the responses to "
    "influence your evaluation. Do not favor certain names of the "
    "assistants. Be as objective as possible. After providing your "
    "explanation, output your final verdict by strictly following this "
    "format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B "
    "is better, and \"[[C]]\" for a tie."
)


def render_system_prompt(comparison: Comparison) -> str:
    """MT-Bench uses a different system prompt for multi-turn items, not
    just a different user-turn template — see MULTI_TURN_SYSTEM_PROMPT."""
    if comparison.context_a or comparison.context_b:
        return MULTI_TURN_SYSTEM_PROMPT
    return SYSTEM_PROMPT


def _render_side(label: str, prior_turns, prompt: str, response: str) -> str:
    lines = [f"<|The Start of Assistant {label}'s Conversation with User|>", ""]
    for role, content in prior_turns:
        speaker = "User" if role == "user" else f"Assistant {label}"
        lines.append(f"### {speaker}:\n{content}\n")
    lines.append(f"### User:\n{prompt}\n")
    lines.append(f"### Assistant {label}:\n{response}\n")
    lines.append(f"<|The End of Assistant {label}'s Conversation with User|>")
    return "\n".join(lines)


def render_prompt(comparison: Comparison) -> str:
    """
    The user-turn prompt the judge sees.

    A single-turn item uses MT-Bench's plain pair-v2 template. A multi-turn
    item (non-empty `context_a`/`context_b`) uses the multi-turn variant
    instead, rendering each side's own prior exchange — which can differ
    between the two systems — ahead of the current question, so the judge
    sees the same conversation a human annotator did rather than an
    isolated final exchange.
    """
    if not comparison.context_a and not comparison.context_b:
        return (
            f"[User Question]\n{comparison.prompt}\n\n"
            f"[The Start of Assistant A's Answer]\n{comparison.response_a}\n"
            f"[The End of Assistant A's Answer]\n\n"
            f"[The Start of Assistant B's Answer]\n{comparison.response_b}\n"
            f"[The End of Assistant B's Answer]"
        )
    side_a = _render_side("A", comparison.context_a, comparison.prompt, comparison.response_a)
    side_b = _render_side("B", comparison.context_b, comparison.prompt, comparison.response_b)
    return f"{side_a}\n\n\n{side_b}"
