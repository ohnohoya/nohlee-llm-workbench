PROMPT_GREETINGS = "hi"

PROMPT_REASONING = (
    "You are designing performance metrics for a fast-growing startup that values "
    "long-term customer retention over short-term revenue.\n\n"
    "1. Propose 3 KPIs.\n"
    "2. For each KPI, explain why it aligns with long-term strategy.\n"
    "3. Identify one unintended consequence each KPI might create.\n"
    "4. Suggest how to mitigate that risk.\n\n"
    "Compromise the quality of your answer for a quick and concise response. I need your answer right away."
)

MESSAGES_MULTI_TURN_EXAMPLE = [
    {"role": "user", "content": "Hey! Did you end up checking out that coffee shop on 5th?"},
    {"role": "assistant", "content": "Yeah, I did. The espresso was great, but it was pretty busy."},
    {"role": "user", "content": "Nice. Was it a long wait?"},
]
