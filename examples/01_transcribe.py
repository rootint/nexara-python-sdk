"""Basic transcription: audio in, text out."""

from nexara import Nexara

client = Nexara(api_key="mock-key")

result = client.transcriptions.create(url="https://example.com/audio.mp3")
print(result.text)

# Other output formats come back as plain strings, not models.
srt = client.transcriptions.create(url="https://example.com/audio.mp3", response_format="srt")
print(srt)
