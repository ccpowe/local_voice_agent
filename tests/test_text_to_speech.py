from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import local_voice_agent.text_to_speech as tts_module


def test_resolve_device_respects_cuda(monkeypatch):
    monkeypatch.setattr(tts_module.torch.cuda, "is_available", lambda: True)
    t = tts_module.TextToSpeech(device="auto", output_dir=Path.cwd())
    dev = t._resolve_device()
    assert str(dev).startswith("cuda")

    monkeypatch.setattr(tts_module.torch.cuda, "is_available", lambda: False)
    t2 = tts_module.TextToSpeech(device="auto", output_dir=Path.cwd())
    dev2 = t2._resolve_device()
    assert str(dev2).startswith("cpu")


def test_preprocess_and_calculate_speed():
    t = tts_module.TextToSpeech(output_dir=Path.cwd())
    raw = "  这是  测试。\n第二行。\r\n  "
    processed = t._preprocess_text(raw)
    assert "\n" not in processed and "\r" not in processed
    assert "  " not in processed

    assert t._calculate_speed(10) == t.speed
    assert pytest.approx(t._calculate_speed(75), rel=1e-3) == t.speed * 0.95
    assert pytest.approx(t._calculate_speed(150), rel=1e-3) == t.speed * 0.9
    assert pytest.approx(t._calculate_speed(500), rel=1e-3) == t.speed * 0.85


def test_split_long_text_splits_sentences():
    t = tts_module.TextToSpeech(output_dir=Path.cwd())
    text = "第一句。这是第二句很长很长很长的句子，它会被拆分，继续，继续。短句。"
    segments = t._split_long_text(text, max_chars=10)
    # All segments should be non-empty and not exceed max_chars
    assert all(seg for seg in segments)
    assert all(len(seg) <= 10 for seg in segments)


def test_get_available_voices_and_set_voice(tmp_path):
    # create local model structure
    repo_dir = tmp_path / "kokoro" / tts_module.DEFAULT_MODEL_REPO_ID.replace("/", "--")
    voices_dir = repo_dir / "voices"
    voices_dir.mkdir(parents=True)
    (voices_dir / "zf_001.pt").write_bytes(b"")
    (voices_dir / "custom.pt").write_bytes(b"")

    t = tts_module.TextToSpeech(model_cache_dir=str(tmp_path), output_dir=tmp_path)
    voices = t.get_available_voices()
    assert "zf_001" in voices
    assert "custom" in voices

    # set by file path
    custom_file = voices_dir / "custom.pt"
    assert t.set_voice(str(custom_file)) is True
    assert str(custom_file) in t.voice

    # set by name
    assert t.set_voice("zf_001") is True
    # set unavailable voice
    assert t.set_voice("does_not_exist") is False


def test_load_models_local_and_remote(tmp_path, monkeypatch):
    # prepare local model structure
    repo_dir = tmp_path / "kokoro" / tts_module.DEFAULT_MODEL_REPO_ID.replace("/", "--")
    repo_dir.mkdir(parents=True)
    (repo_dir / "config.json").write_text("{}")
    model_name = "local_model.bin"
    (repo_dir / model_name).write_bytes(b"")

    # Fake KModel and KPipeline
    created = {}

    class FakeKModel:
        MODEL_NAMES = {tts_module.DEFAULT_MODEL_REPO_ID: model_name}

        def __init__(self, repo_id=None, config=None, model=None):
            created["repo_id"] = repo_id
            created["config"] = config
            created["model"] = model

        def to(self, device):
            return self

        def eval(self):
            return self

    class FakeKPipeline:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            # generator that yields an object with `.audio`
            yield SimpleNamespace(audio=np.zeros(4))

    monkeypatch.setattr(tts_module, "KModel", FakeKModel)
    monkeypatch.setattr(tts_module, "KPipeline", FakeKPipeline)

    t = tts_module.TextToSpeech(model_cache_dir=str(tmp_path), output_dir=tmp_path)
    assert t._local_model_dir is not None
    ok = t._load_models()
    assert ok is True
    assert isinstance(t.model, FakeKModel)
    assert isinstance(t.en_pipeline, FakeKPipeline)
    assert isinstance(t.zh_pipeline, FakeKPipeline)


def test_load_models_remote_when_no_local(monkeypatch, tmp_path):
    created = {}

    class FakeKModel2:
        def __init__(self, repo_id=None, config=None, model=None):
            created["repo_id"] = repo_id
            created["config"] = config
            created["model"] = model

        def to(self, device):
            return self

        def eval(self):
            return self

    class FakeKPipeline2:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            yield SimpleNamespace(audio=np.zeros(2))

    monkeypatch.setattr(tts_module, "KModel", FakeKModel2)
    monkeypatch.setattr(tts_module, "KPipeline", FakeKPipeline2)

    t = tts_module.TextToSpeech(output_dir=tmp_path)
    # ensure no local model dir
    assert t._local_model_dir is None
    ok = t._load_models()
    assert ok is True
    assert created.get("repo_id") == t.repo_id


def test_synthesize_text_success_and_empty_input(monkeypatch, tmp_path):
    t = tts_module.TextToSpeech(output_dir=tmp_path)

    # mock models loaded
    monkeypatch.setattr(t, "_load_models", lambda: True)

    # fake zh_pipeline that yields an object with `.audio`
    def fake_pipeline(text, voice=None, speed=None):
        def gen():
            yield SimpleNamespace(audio=np.array([0.1, -0.1], dtype=float))

        return gen()

    t.zh_pipeline = fake_pipeline

    # avoid actual file writes by patching _save_audio_file
    monkeypatch.setattr(t, "_save_audio_file", lambda output, audio: str(tmp_path / "out.wav"))

    out = t.synthesize_text("  测试文本  ")
    assert out is not None
    assert out.endswith("out.wav")

    # empty/whitespace-only input should return None
    res = t.synthesize_text("   \n  ")
    assert res is None


def test_synthesize_paragraphs_merging_and_silence(monkeypatch, tmp_path):
    t = tts_module.TextToSpeech(output_dir=tmp_path)
    monkeypatch.setattr(t, "_load_models", lambda: True)

    # set small sample_rate and silence to make silence arrays short
    t.sample_rate = 10
    t.silence_duration = 0.2  # silence_samples = 2

    # fake pipeline: return different audio arrays per paragraph
    def make_pipeline(audio_array):
        def pipeline(text, voice=None, speed=None):
            def gen():
                yield SimpleNamespace(audio=np.array(audio_array, dtype=float))

            return gen()

        return pipeline

    # for two paragraphs produce arrays [1,1] and [2,2]
    pipelines = {"p1": make_pipeline([1.0, 1.0]), "p2": make_pipeline([2.0, 2.0])}

    def zh_pipeline(text, voice=None, speed=None):
        return pipelines["p1" if "第一" in text or "p1" in text else "p2"](text, voice, speed)

    t.zh_pipeline = zh_pipeline

    captured = {}

    def fake_save(output, audio):
        captured["audio"] = audio
        return str(tmp_path / "out.wav")

    monkeypatch.setattr(t, "_save_audio_file", fake_save)

    paragraphs = ["p1", "p2"]
    out = t.synthesize_paragraphs(paragraphs, output_file=None)
    assert out is not None
    assert "audio" in captured
    combined = captured["audio"]
    # silence_samples = int(0.2 * 10) = 2
    expected = np.concatenate([np.array([1.0, 1.0]), np.zeros(2), np.array([2.0, 2.0])])
    assert np.allclose(combined, expected)


def test_split_long_text_complex_cases():
    t = tts_module.TextToSpeech(output_dir=Path.cwd())
    text = "这是第一句，很长很长，包含许多子句，需要被拆分。第二句也比较长，需要再拆分，继续，继续。短句。"
    segments = t._split_long_text(text, max_chars=10)
    assert isinstance(segments, list)
    assert len(segments) > 1
    assert all(len(seg) <= 10 for seg in segments)


def test_synthesize_long_text_short_and_split(monkeypatch, tmp_path):
    t = tts_module.TextToSpeech(output_dir=tmp_path)
    monkeypatch.setattr(t, "_load_models", lambda: True)

    # short path: _split_long_text returns single segment -> synthesize_text called
    monkeypatch.setattr(t, "_split_long_text", lambda text, max_chars: [text])
    monkeypatch.setattr(t, "synthesize_text", lambda text, output_file=None: "short.wav")
    out = t.synthesize_long_text("short text", max_chars=10)
    assert out == "short.wav"

    # long path: split into multiple segments -> synthesize_paragraphs called
    monkeypatch.setattr(t, "_split_long_text", lambda text, max_chars: ["a", "b"])
    monkeypatch.setattr(t, "synthesize_paragraphs", lambda paragraphs, output_file=None: "paras.wav")
    out2 = t.synthesize_long_text("long text", max_chars=5)
    assert out2 == "paras.wav"


def test_save_audio_file_default_and_custom(monkeypatch, tmp_path):
    t = tts_module.TextToSpeech(output_dir=tmp_path)

    written = {}

    def fake_write(path, audio, sample_rate):
        # create a file to simulate write
        Path(path).write_bytes(b"ok")
        written["path"] = path
        written["audio_shape"] = getattr(audio, "shape", None)

    monkeypatch.setattr(tts_module, "sf", SimpleNamespace(write=fake_write))

    audio = np.zeros(5)
    out_path = t._save_audio_file(None, audio)
    assert "tts_output_" in out_path
    assert Path(out_path).exists()

    # custom path ensures parent dir created
    custom = tmp_path / "sub" / "o.wav"
    out2 = t._save_audio_file(str(custom), audio)
    assert out2 == str(custom.resolve())
    assert Path(out2).exists()


def test_load_models_failure_and_en_callable(monkeypatch, tmp_path):
    # KModel raising should cause _load_models to return False
    class BadKModel:
        def __init__(self, *a, **k):
            raise RuntimeError("bad")

    monkeypatch.setattr(tts_module, "KModel", BadKModel)
    t = tts_module.TextToSpeech(output_dir=tmp_path)
    assert t._load_models() is False

    # Now test en_callable behavior when KPipeline present
    created = {}

    class FakeKModel3:
        MODEL_NAMES = {}

        def __init__(self, *a, **k):
            pass

        def to(self, device):
            return self

        def eval(self):
            return self

    class FakeKPipeline3:
        last_en_callable = None

        def __init__(self, *args, **kwargs):
            self.lang_code = kwargs.get("lang_code")
            if "en_callable" in kwargs:
                FakeKPipeline3.last_en_callable = kwargs.get("en_callable")

        def __call__(self, text, *a, **k):
            if self.lang_code == "a":
                # english pipeline yields object with .phonemes
                yield SimpleNamespace(phonemes="PH")
            else:
                yield SimpleNamespace(audio=np.zeros(1))

    monkeypatch.setattr(tts_module, "KModel", FakeKModel3)
    monkeypatch.setattr(tts_module, "KPipeline", FakeKPipeline3)

    t2 = tts_module.TextToSpeech(output_dir=tmp_path)
    assert t2._load_models() is True
    # en_callable special-case
    en_callable = FakeKPipeline3.last_en_callable
    assert callable(en_callable)
    assert en_callable("kokoro") == "kˈOkəɹO"
    # other text delegates to en_pipeline phonemes
    assert en_callable("hello") == "PH"

