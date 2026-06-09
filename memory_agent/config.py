import os
from dataclasses import dataclass, field, fields
from typing_extensions import Annotated

@dataclass(kw_only=True)
class Context:
    user_id: str = "default_user"

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="mimo-v2.5-pro",
        metadata={"description": "Model name used by the OpenAI-compatible endpoint."},
    )

    debug: bool = False

    system_prompt = """
You are an expert AI assistant.

When answering:

- Be comprehensive by default.
- Teach concepts rather than only giving conclusions.
- Use clear section headers.
- Explain trade-offs.
- Give examples.
- Anticipate likely follow-up questions.
- For programming topics, explain architecture, design choices, and implementation details.
- Prefer depth over brevity unless the user explicitly requests a short answer.

Relevant user memories:
{user_info}

System Time: {time}
"""

    def __post_init__(self) -> None:
        for f in fields(self):
            if not f.init:
                continue

            current_value = getattr(self, f.name)
            default_value = f.default

            if current_value == default_value:
                env_value = os.environ.get(f.name.upper())
                if env_value is not None:
                    if isinstance(default_value, bool):
                        setattr(self, f.name, env_value.lower() in {"1", "true", "yes"})
                    else:
                        setattr(self, f.name, env_value)