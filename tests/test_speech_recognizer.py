from types import SimpleNamespace


from local_voice_agent import speech_recognition


def test__convert_to_simplified_uses_zhconv(monkeypatch):
    sr = speech_recognition.SpeechRecognizer()

    # monkeypatch the convert function to ensure behavior is exercised
    monkeypatch.setattr(speech_recognition, "convert", lambda text, _: "简体示例")
    out = sr._convert_to_simplified("繁體示例")
    assert out == "简体示例"


def test_load_model_and_transcribe_with_mock(monkeypatch, tmp_path):
    sr = speech_recognition.SpeechRecognizer(model_cache_dir=tmp_path)

    # Fake WhisperModel class to avoid heavy model download/initialization
    class FakeWhisperModel:
        def __init__(self, *args, **kwargs):
            # pretend initialization is fine
            pass

        def transcribe(self, path, **kwargs):
            # return a single-segment fake result and an info object
            seg = SimpleNamespace(text="测试文本", start=0.0, end=1.0)
            info = SimpleNamespace(language="zh", language_probability=0.99)
            return [seg], info

    monkeypatch.setattr(speech_recognition, "WhisperModel", FakeWhisperModel)

    # load_model should succeed
    assert sr.load_model() is True
    assert sr.model is not None

    # create a fake wav file
    wav_file = tmp_path / "test.wav"
    wav_file.write_bytes(b"RIFF....WAVEfmt ")  # minimal bytes; model is mocked so content not used

    # Now transcribe should call our FakeWhisperModel.transcribe and return simplified text
    monkeypatch.setattr(speech_recognition, "convert", lambda t, _: "测试文本")
    result = sr.transcribe_audio_file(str(wav_file))
    assert result == "测试文本"

