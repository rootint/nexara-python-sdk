"""Word-level timestamps — and the trap in the default.

verbose_json alone does NOT give you words: the default granularity is
"segment", and the server strips words from the response. You have to ask.
"""

from nexara import Nexara

client = Nexara(api_key="mock-key")

# The default: verbose, but wordless.
default = client.transcriptions.create(
    url="https://example.com/audio.mp3",
    response_format="verbose_json",
)
print("words with default granularity:", default.words)  # None

# Ask for words explicitly.
detailed = client.transcriptions.create(
    url="https://example.com/audio.mp3",
    response_format="verbose_json",
    timestamp_granularities=["word"],
)
assert detailed.words is not None
for word in detailed.words:
    print(f"{word.start:5.2f}–{word.end:5.2f}  {word.word}  (prob={word.prob})")

# Sentence granularity replaces `segments` with `sentences` — segments go away.
by_sentence = client.transcriptions.create(
    url="https://example.com/audio.mp3",
    response_format="verbose_json",
    timestamp_granularities=["sentence"],
)
print("segments:", by_sentence.segments)  # None
assert by_sentence.sentences is not None
for sentence in by_sentence.sentences:
    print(f"{sentence.start:5.2f}–{sentence.end:5.2f}  {sentence.text}")
