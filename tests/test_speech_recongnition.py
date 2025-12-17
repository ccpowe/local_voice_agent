from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from local_voice_agent.speech_recognition import SpeechRecognizer


@dataclass
class FakeInfo:
    language: str = "zh"
    language_probability: float = 0.99


@dataclass
class FakeSegment:
    text: str
    start: float = 0.0
    end: float = 0.0


def test_load_model_uses_download_root(tmp_path, monkeypatch):
    import local_voice_agent.speech_recognition as sr_module

    captured = {}

    class FakeWhisperModel:
        def __init__(
            self,
            model_size_or_path: str,
            device: str = "auto",
            compute_type: str = "default",
            download_root: str | None = None,
            **kwargs,
        ):
            captured["model_size_or_path"] = model_size_or_path
            captured["device"] = device
            captured["compute_type"] = compute_type
            captured["download_root"] = download_root

        def transcribe(self, *args, **kwargs):
            raise AssertionError("Not used in this test")

    monkeypatch.setattr(sr_module, "WhisperModel", FakeWhisperModel)

    recognizer = SpeechRecognizer(
        model_size="small",
        device="auto",
        compute_type="float16",
        model_cache_dir=str(tmp_path),
    )
    assert recognizer.model_path == Path(tmp_path) / "whisper"
    assert recognizer.model_path.exists()

    assert recognizer.load_model() is True
    assert captured["model_size_or_path"] == "small"
    assert captured["device"] == "auto"
    assert captured["compute_type"] == "float16"
    assert captured["download_root"] == str(Path(tmp_path) / "whisper")


def test_transcribe_audio_file_loads_model_when_missing(tmp_path, monkeypatch):
    wav_path = Path("tests/test_data/voice_zh.wav")
    assert wav_path.exists()

    recognizer = SpeechRecognizer(model_cache_dir=str(tmp_path))

    called = {"load_model": 0}

    class FakeModel:
        def transcribe(self, audio_file_path: str, **kwargs):
            return [FakeSegment("hello")], FakeInfo(language="en")

    def fake_load_model():
        called["load_model"] += 1
        recognizer.model = FakeModel()
        return True

    monkeypatch.setattr(recognizer, "load_model", fake_load_model)
    monkeypatch.setattr(
        "local_voice_agent.speech_recognition.convert",
        lambda text, _: text,
        raising=True,
    )

    text = recognizer.transcribe_audio_file(str(wav_path))
    assert called["load_model"] == 1
    assert text == "hello"


def test_transcribe_audio_file_combines_segments_and_converts(tmp_path, monkeypatch):
    wav_path = Path("tests/test_data/voice_zh.wav")
    assert wav_path.exists()

    recognizer = SpeechRecognizer(model_cache_dir=str(tmp_path))

    class FakeModel:
        def transcribe(self, audio_file_path: str, **kwargs):
            assert kwargs["language"] == recognizer.language
            assert kwargs["beam_size"] == 5
            assert kwargs["best_of"] == 5
            assert kwargs["temperature"] == 0.0
            assert kwargs["condition_on_previous_text"] is False
            return [FakeSegment("繁體"), FakeSegment("中文")], FakeInfo(language="zh")

    recognizer.model = FakeModel()

    monkeypatch.setattr(
        "local_voice_agent.speech_recognition.convert",
        lambda text, _: text.replace("繁體", "繁体"),
        raising=True,
    )

    assert recognizer.transcribe_audio_file(str(wav_path)) == "繁体中文"


def test_transcribe_audio_file_returns_none_for_empty_text(tmp_path, monkeypatch):
    wav_path = Path("tests/test_data/HEARME_en.wav")
    assert wav_path.exists()

    recognizer = SpeechRecognizer(model_cache_dir=str(tmp_path))

    class FakeModel:
        def transcribe(self, audio_file_path: str, **kwargs):
            return [FakeSegment("   ")], FakeInfo(language="en")

    recognizer.model = FakeModel()

    monkeypatch.setattr(
        "local_voice_agent.speech_recognition.convert",
        lambda text, _: text,
        raising=True,
    )

    assert recognizer.transcribe_audio_file(str(wav_path)) is None
