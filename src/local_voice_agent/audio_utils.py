"""
音频录制和播放工具模块
"""

import io
import wave

import numpy as np
import sounddevice as sd


class AudioRecorder:
    """音频工具类，提供音频文件保存功能"""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
    ):
        self.sample_rate = sample_rate
        self.channels = channels

    def save_to_wav(self, audio_data: np.ndarray, filename: str) -> None:
        """保存音频数据到WAV文件"""
        # 确保音频数据是int16格式
        if audio_data.dtype != np.int16:
            # 如果是float32格式，需要转换
            if audio_data.dtype == np.float32:
                audio_data = (audio_data * 32767).astype(np.int16)
            else:
                audio_data = audio_data.astype(np.int16)

        with wave.open(filename, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16 bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())

    def get_wav_bytes(self, audio_data: np.ndarray) -> bytes:
        """将音频数据转换为WAV格式的字节流"""
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())
        return buffer.getvalue()


class AudioPlayer:
    """音频播放器"""

    def __init__(self, sample_rate: int = 24000):  # 24000是TTS默认采样率
        self.sample_rate = sample_rate

    def play_audio(self, audio_data: np.ndarray) -> None:
        """播放音频数据"""
        # 确保音频数据是float32格式，范围在[-1, 1]
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32767.0

        sd.play(audio_data, samplerate=self.sample_rate)
        sd.wait()  # 等待播放完成

    def play_wav_file(self, filename: str) -> None:
        """播放WAV文件"""
        with wave.open(filename, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            audio_data = np.frombuffer(frames, dtype=np.int16)
            self.play_audio(audio_data)
