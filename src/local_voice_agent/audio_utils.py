"""
音频录制和播放工具模块
"""

import io
import logging
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

_PCM16_MAX = float(np.iinfo(np.int16).max)


def _to_pcm16(audio_data: np.ndarray) -> np.ndarray:
    if audio_data.dtype == np.int16:
        return audio_data
    if audio_data.dtype == np.float32:
        return (audio_data * _PCM16_MAX).astype(np.int16)
    return audio_data.astype(np.int16)


class AudioRecorder:
    """音频工具类，提供音频文件保存功能"""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
    ):
        self.sample_rate = sample_rate
        self.channels = channels

    def save_to_wav(self, audio_data: np.ndarray, filename: str | Path) -> None:
        """保存音频数据到WAV文件"""
        audio_data = _to_pcm16(audio_data)

        with wave.open(str(Path(filename).resolve()), "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16 bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())

    def get_wav_bytes(self, audio_data: np.ndarray) -> bytes:
        """将音频数据转换为WAV格式的字节流"""
        audio_data = _to_pcm16(audio_data)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())
        return buffer.getvalue()

    def record_manual(self) -> np.ndarray | None:
        """
        手动控制录音：按 Enter 开始，再按 Enter 停止。

        Returns:
            一维 int16 音频数据；未录到音频则返回 None
        """
        print("🎤 准备开始录制...")
        input("按 Enter 开始录制: ")

        print("🔴 录制中... 按 Enter 停止录制")

        recording = True
        audio_chunks: list[np.ndarray] = []

        def audio_callback(indata, frames, time, status):
            if recording:
                audio_chunks.append(indata.copy())

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=audio_callback,
            dtype=np.float32,
        ):
            input()

        recording = False
        print("⏹️ 录制停止!")

        if not audio_chunks:
            logger.warning("未录制到音频数据")
            return None

        audio_float = np.concatenate(audio_chunks, axis=0)
        audio_int16 = (audio_float * _PCM16_MAX).astype(np.int16).flatten()
        return audio_int16


class AudioPlayer:
    """音频播放器"""

    def __init__(self, sample_rate: int = 24000):  # 24000是TTS默认采样率
        self.sample_rate = sample_rate

    def play_audio(self, audio_data: np.ndarray) -> None:
        """播放音频数据"""
        # 确保音频数据是float32格式，范围在[-1, 1]
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / _PCM16_MAX

        sd.play(audio_data, samplerate=self.sample_rate)
        sd.wait()  # 等待播放完成

    def play_wav_file(self, filename: str | Path) -> None:
        """播放WAV文件"""
        with wave.open(str(Path(filename).resolve()), "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            audio_data = np.frombuffer(frames, dtype=np.int16)
            self.play_audio(audio_data)
