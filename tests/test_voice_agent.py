import asyncio
from types import SimpleNamespace
from pathlib import Path
from typing import cast

import numpy as np

import local_voice_agent.voice_agent as va


def test_build_chat_model_with_and_without_base_url(monkeypatch):
    created = {}

    class FakeChat:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr(va, "ChatOpenAI", FakeChat)

    settings = SimpleNamespace(model_name="m", api_key="k", base_url=None)
    _ = va._build_chat_model(cast(va.Settings, settings))
    assert "model_name" in created
    assert created["model_name"] == "m"

    created.clear()
    settings2 = SimpleNamespace(model_name="m2", api_key="k2", base_url="http://x")
    _ = va._build_chat_model(cast(va.Settings, settings2))
    assert created["model_name"] == "m2"
    assert "openai_api_base" in created


def test_init_tts_paths_and_return(monkeypatch, tmp_path, capsys):
    # success path
    class FakeTTS:
        def __init__(self, *args, **kwargs):
            pass

        def _load_models(self):
            return True

    monkeypatch.setattr(va, "TextToSpeech", FakeTTS)
    tts, ok = va._init_tts(tmp_path, tmp_path / "out")
    assert ok is True
    assert isinstance(tts, FakeTTS)

    # failure path
    class BadTTS:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(va, "TextToSpeech", BadTTS)
    tts2, ok2 = va._init_tts(tmp_path, tmp_path / "out")
    assert tts2 is None and ok2 is False


def test_print_usage_outputs(capsys=None):
    # tts_available True
    va._print_usage(True)
    # tts_available False
    va._print_usage(False)
    # No assertion on full content, just ensure function runs (manual inspection possible)


def test_record_and_transcribe_success_and_none(monkeypatch, tmp_path):
    # control working dir so debug_dir is predictable
    monkeypatch.chdir(tmp_path)

    class FakeRecorder:
        def record_manual(self):
            return np.array([0, 1], dtype=np.int16)

        def save_to_wav(self, audio, path):
            # create file to simulate save
            Path(path).write_bytes(b"RIFF")

    class FakeRecognizer:
        def transcribe_audio_file(self, path):
            return "hello"

    out = asyncio.run(
        va._record_and_transcribe(
            cast(va.AudioRecorder, FakeRecorder()), cast(va.SpeechRecognizer, FakeRecognizer())
        )
    )
    assert out == "hello"

    # when no audio recorded
    class EmptyRecorder:
        def record_manual(self):
            return None

    out2 = asyncio.run(
        va._record_and_transcribe(
            cast(va.AudioRecorder, EmptyRecorder()), cast(va.SpeechRecognizer, FakeRecognizer())
        )
    )
    assert out2 is None


def test_init_tts_when_load_models_false(monkeypatch, tmp_path):
    class FakeTTS:
        def __init__(self, *a, **k):
            pass

        def _load_models(self):
            return False

    monkeypatch.setattr(va, "TextToSpeech", FakeTTS)
    tts, ok = va._init_tts(tmp_path, tmp_path / "out")
    assert isinstance(tts, FakeTTS)
    assert ok is False


def test_init_speech_recognizer_load_failure(monkeypatch, tmp_path):
    class FakeSR:
        def __init__(self, *a, **k):
            pass

        def load_model(self):
            return False

    monkeypatch.setattr(va, "SpeechRecognizer", FakeSR)
    recognizer = va._init_speech_recognizer(tmp_path)
    assert isinstance(recognizer, FakeSR)


def test_async_main_flow(monkeypatch, tmp_path):
    # sequence of user inputs to drive the loop and exit
    inputs = ["tts:on", "voice:女声", "s", "再见"]

    async def fake_ainput(prompt):
        return inputs.pop(0)

    class FakeAgent:
        def __init__(self, *a, **k):
            self.calls = 0

        async def ainvoke(self, *args, **kwargs):
            self.calls += 1
            return {"messages": [SimpleNamespace(content="AI reply")]}

    class FakeTTS:
        def __init__(self, *a, **k):
            self.calls = 0

        def synthesize_long_text(self, text):
            self.calls += 1
            return str(tmp_path / "reply.wav")
        def set_voice(self, v):
            return True

    class FakeAudioPlayer:
        def __init__(self, *a, **k):
            self.played = []

        def play_wav_file(self, path):
            self.played.append(path)

    # monkeypatch many pieces of async_main
    monkeypatch.setattr(va, "ainput", fake_ainput)
    monkeypatch.setattr(va, "_build_chat_model", lambda settings: "fake_model")
    monkeypatch.setattr(va, "create_agent", lambda model, tools, system_prompt: FakeAgent())
    monkeypatch.setattr(va, "_init_speech_recognizer", lambda model_dir: SimpleNamespace(load_model=lambda: True, transcribe_audio_file=lambda p: "recognized"))
    monkeypatch.setattr(va, "_init_tts", lambda model_dir, out_dir: (FakeTTS(), True))
    monkeypatch.setattr(va, "AudioPlayer", FakeAudioPlayer)

    async def fake_record_and_transcribe(audio_recorder, speech_recognizer):
        return "recognized"

    monkeypatch.setattr(va, "_record_and_transcribe", fake_record_and_transcribe)

    # Settings should be simple and not validate env
    monkeypatch.setattr(va, "Settings", lambda: SimpleNamespace(model_name="m", api_key="k", base_url=None))

    # run async_main (will exit after inputs exhausted)
    import asyncio

    asyncio.run(va.async_main())

    # If we reach here without error, basic flow executed. No direct asserts on prints.

