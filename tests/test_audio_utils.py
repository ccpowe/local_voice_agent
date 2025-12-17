import builtins
import io
import wave

import numpy as np

from local_voice_agent.audio_utils import AudioRecorder


def test_save_to_wav_writes_valid_wav(tmp_path):
    recorder = AudioRecorder(sample_rate=16000, channels=1)
    audio_float = np.array([0.0, 0.5, -0.5], dtype=np.float32)
    out_file = tmp_path / "out.wav"

    recorder.save_to_wav(audio_float, str(out_file))

    with wave.open(str(out_file), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 16000
        assert wf.getsampwidth() == 2
        frames = wf.readframes(wf.getnframes())

    samples = np.frombuffer(frames, dtype=np.int16)
    assert samples.dtype == np.int16
    assert samples.tolist() == [0, int(0.5 * 32767), int(-0.5 * 32767)]


def test_get_wav_bytes_is_readable_wav():
    recorder = AudioRecorder(sample_rate=8000, channels=1)
    audio_int16 = np.array([0, 1, -1, 32767, -32768], dtype=np.int16)

    wav_bytes = recorder.get_wav_bytes(audio_int16)
    assert isinstance(wav_bytes, (bytes, bytearray))

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 8000
        assert wf.getsampwidth() == 2
        frames = wf.readframes(wf.getnframes())

    samples = np.frombuffer(frames, dtype=np.int16)
    assert samples.tolist() == audio_int16.tolist()


def test_record_manual_returns_int16(monkeypatch):
    recorder = AudioRecorder(sample_rate=16000, channels=1)

    # Auto-advance the "press Enter" prompts
    monkeypatch.setattr(builtins, "input", lambda *args, **kwargs: "")

    # Fake InputStream that feeds two chunks into callback.
    import local_voice_agent.audio_utils as audio_utils_module

    class FakeInputStream:
        def __init__(self, samplerate, channels, callback, dtype):
            self.callback = callback

        def __enter__(self):
            self.callback(np.array([[0.0], [0.25]], dtype=np.float32), 2, None, None)
            self.callback(np.array([[-0.25]], dtype=np.float32), 1, None, None)
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(audio_utils_module.sd, "InputStream", FakeInputStream)

    audio_int16 = recorder.record_manual()
    assert audio_int16 is not None
    assert audio_int16.dtype == np.int16
    assert audio_int16.tolist() == [0, int(0.25 * 32767), int(-0.25 * 32767)]
