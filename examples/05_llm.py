"""Structured output: run a prompt over the transcript."""

import json

from nexara import Nexara

client = Nexara(api_key="mock-key")

# A prompt alone gives free-form text back.
plain = client.transcriptions.create(
    url="https://example.com/call.mp3",
    prompt="Summarise this call in one sentence.",
)
print(plain.llm_output)
print(plain.transcription.text)

# Add a schema to get an object instead.
structured = client.transcriptions.create(
    url="https://example.com/call.mp3",
    prompt="Summarise this call and judge its sentiment.",
    json_schema={
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "sentiment": {"enum": ["positive", "neutral", "negative"]},
        },
        "required": ["summary", "sentiment"],
    },
)
print(json.dumps(structured.llm_output, ensure_ascii=False, indent=2))
