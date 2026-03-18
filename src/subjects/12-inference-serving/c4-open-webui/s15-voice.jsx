import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function Voice() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Voice Input/Output (STT & TTS)</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Open WebUI supports both speech-to-text (STT) for voice input and text-to-speech (TTS)
        for spoken responses. This enables hands-free interaction with LLMs, making the interface
        accessible on mobile devices and for voice-first workflows.
      </p>

      <DefinitionBlock
        title="STT & TTS in Open WebUI"
        definition="Speech-to-text converts audio input to text before sending to the LLM. Text-to-speech converts the LLM's text response to audio. Open WebUI supports browser-native speech APIs, OpenAI Whisper/TTS, and self-hosted alternatives."
        id="def-voice"
      />

      <PythonCode
        title="Terminal"
        code={`# Configure voice with OpenAI APIs
docker run -d -p 3000:8080 \\
    -e AUDIO_STT_ENGINE=openai \\
    -e AUDIO_STT_MODEL=whisper-1 \\
    -e AUDIO_TTS_ENGINE=openai \\
    -e AUDIO_TTS_MODEL=tts-1 \\
    -e AUDIO_TTS_VOICE=nova \\
    -e OPENAI_API_KEY=sk-your-key \\
    -v open-webui:/app/backend/data \\
    --name open-webui \\
    ghcr.io/open-webui/open-webui:main

# Self-hosted STT with local Whisper
# -e AUDIO_STT_ENGINE=openai
# -e AUDIO_STT_OPENAI_API_BASE_URL=http://localhost:8000/v1
# Uses faster-whisper or whisper.cpp server

# Browser-native speech (no server needed, limited quality)
# -e AUDIO_STT_ENGINE=web

# Available TTS voices (OpenAI): alloy, echo, fable, onyx, nova, shimmer`}
        id="code-config"
      />

      <PythonCode
        title="self_hosted_whisper.py"
        code={`# Set up a self-hosted Whisper server for STT
# Option 1: faster-whisper-server
# pip install faster-whisper
# Provides an OpenAI-compatible Whisper API

import subprocess
import requests
import tempfile
import wave
import struct
import math

# Start faster-whisper server (run separately)
# python -m faster_whisper_server --model large-v3 --port 8000

# Test STT by sending audio to the local Whisper server
WHISPER_URL = "http://localhost:8000/v1/audio/transcriptions"

# Generate a test audio file (sine wave)
def create_test_audio(filename, duration=2, freq=440):
    sample_rate = 16000
    n_samples = int(sample_rate * duration)
    with wave.open(filename, 'w') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for i in range(n_samples):
            value = int(32767 * math.sin(2 * math.pi * freq * i / sample_rate))
            wav.writeframes(struct.pack('<h', value))

# In practice, record real audio or use a file
# Here we just demonstrate the API call
audio_file = "test.wav"
create_test_audio(audio_file)

with open(audio_file, "rb") as f:
    resp = requests.post(
        WHISPER_URL,
        files={"file": ("audio.wav", f, "audio/wav")},
        data={"model": "large-v3"},
    )
    print(f"Transcription: {resp.json().get('text', 'N/A')}")

# For TTS, use Piper (fast, local, open-source)
# pip install piper-tts
# piper --model en_US-lessac-medium.onnx --output_file output.wav
print("Local STT + TTS configured for Open WebUI")`}
        id="code-whisper"
      />

      <ExampleBlock
        title="Voice Integration Options"
        problem="Compare STT and TTS options for Open WebUI."
        steps={[
          { formula: 'Browser Web Speech API: free, no setup, basic quality', explanation: 'Uses the browser built-in speech recognition. Works offline in Chrome.' },
          { formula: 'OpenAI Whisper + TTS: best quality, cloud API costs', explanation: 'Whisper for STT, TTS-1 for speech. ~$0.006/min STT, $0.015/1K chars TTS.' },
          { formula: 'Self-hosted Whisper + Piper: free, runs locally', explanation: 'faster-whisper for STT, Piper for TTS. Needs GPU for real-time speed.' },
        ]}
        id="example-options"
      />

      <NoteBlock
        type="tip"
        title="Mobile Voice Chat"
        content="Open WebUI's voice features work well on mobile browsers. Press and hold the microphone button to record, release to send. Combined with TTS, this creates a voice assistant experience entirely running on your own hardware."
        id="note-mobile"
      />

      <WarningBlock
        title="STT Accuracy"
        content="Speech recognition accuracy depends heavily on the model and audio quality. Background noise, accents, and technical terminology can reduce accuracy. For critical inputs, always review the transcribed text before sending to the LLM."
        id="warning-accuracy"
      />
    </div>
  )
}
