
import numpy as np

from local_voice_agent import audio_utils


def test__to_pcm16_converts_float32_and_passthrough_int16():
    float_data = np.array([0.0, 0.5, -0.5], dtype=np.float32)
    int16_from_float = audio_utils._to_pcm16(float_data)
    assert int16_from_float.dtype == np.int16
    assert int16_from_float.tolist() == [
        0,
        int(0.5 * np.iinfo(np.int16).max),
        int(-0.5 * np.iinfo(np.int16).max),
    ]

    int16_data = np.array([0, 1, -1], dtype=np.int16)
    passthrough = audio_utils._to_pcm16(int16_data)
    assert passthrough.dtype == np.int16
    assert passthrough.tolist() == int16_data.tolist()


def test_audioplayer_play_audio_converts_int16_and_calls_sounddevice(monkeypatch):
    player = audio_utils.AudioPlayer(sample_rate=16000)

    called = {"play": False, "wait": False}

    def fake_play(arr, samplerate):
        # play should receive float32 in range [-1, 1]
        assert arr.dtype == np.float32
        # int16 min value (-32768) maps to slightly less than -1 when divided by 32767,
        # allow a small tolerance equal to 1 / max_int16.
        max_abs = float(np.max(np.abs(arr)))
        eps = 1.0 + (1.0 / np.iinfo(np.int16).max)
        assert max_abs <= eps
        assert samplerate == 16000
        called["play"] = True

    def fake_wait():
        called["wait"] = True

    monkeypatch.setattr(audio_utils.sd, "play", fake_play)
    monkeypatch.setattr(audio_utils.sd, "wait", fake_wait)

    int16_audio = np.array([0, 32767, -32768], dtype=np.int16)
    player.play_audio(int16_audio)

    assert called["play"] is True
    assert called["wait"] is True


def test_play_wav_file_reads_and_plays(tmp_path, monkeypatch):
    recorder = audio_utils.AudioRecorder(sample_rate=8000, channels=1)
    # create a small int16 buffer and write to file
    samples = np.array([0, 1, -1, 32767, -32768], dtype=np.int16)
    wav_bytes = recorder.get_wav_bytes(samples)

    out_file = tmp_path / "tmp.wav"
    out_file.write_bytes(wav_bytes)

    played = {"called": False}

    def fake_play(arr, samplerate):
        # When play_wav_file reads, it produces int16 array which play_audio converts to float
        assert isinstance(arr, np.ndarray)
        played["called"] = True

    def fake_wait():
        return None

    monkeypatch.setattr(audio_utils.AudioPlayer, "play_audio", lambda self, a: fake_play(a, self.sample_rate))
    monkeypatch.setattr(audio_utils.sd, "wait", fake_wait)

    player = audio_utils.AudioPlayer(sample_rate=8000)
    player.play_wav_file(str(out_file))

    assert played["called"] is True

