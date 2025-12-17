"""
语音识别模块 - 基于faster-whisper
"""

import logging
import tempfile
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from faster_whisper import WhisperModel

from .audio_utils import AudioRecorder

# 尝试导入繁简转换库
try:
    from zhconv import convert

    ZHCONV_AVAILABLE = True
except ImportError:
    ZHCONV_AVAILABLE = False
    logging.warning("zhconv库未安装，无法进行繁简转换")


WHISPER_CONFIG = {
    "model_size": "small",
    "device": "auto",
    "compute_type": "float16",
    "language": "zh",
}
AUDIO_CONFIG = {"sample_rate": 16000, "channels": 1}
PATH_CONFIG = {"model_cache_dir": Path.cwd() / "data" / "model"}
LOG_CONFIG = {"level": "INFO"}

# 配置日志
logging.basicConfig(level=getattr(logging, LOG_CONFIG.get("level", "INFO")))
logger = logging.getLogger(__name__)


class SpeechRecognizer:
    """语音识别器，使用faster-whisper进行中文语音识别"""

    def __init__(
        self,
        model_size: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        language: Optional[str] = None,
        model_cache_dir: Optional[str] = None,
    ):
        """
        初始化语音识别器

        Args:
            model_size: 模型大小 ("tiny", "base", "small", "medium", "large")
            device: 设备类型 ("cpu", "cuda", "auto")
            compute_type: 计算类型 ("float16", "float32", "int8")
            language: 语言代码 ("zh"表示中文)
            model_cache_dir: 指定模型下载目录（默认: ./data/model）
        """
        # 使用配置文件的默认值
        self.model_size = model_size or WHISPER_CONFIG.get("model_size", "small")
        self.device = device or WHISPER_CONFIG.get("device", "auto")
        self.compute_type = compute_type or WHISPER_CONFIG.get(
            "compute_type", "float32"
        )
        self.language = language or WHISPER_CONFIG.get("language", "zh")
        self.model = None

        # 初始化音频录制器，使用配置参数
        self.audio_recorder = AudioRecorder(
            sample_rate=AUDIO_CONFIG.get("sample_rate", 16000),
            channels=AUDIO_CONFIG.get("channels", 1),
        )

        # 模型下载路径
        base_cache_dir = (
            Path(model_cache_dir) if model_cache_dir else PATH_CONFIG["model_cache_dir"]
        )
        self.model_path = base_cache_dir / "whisper"
        self.model_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"初始化语音识别器: 模型={self.model_size}, 设备={self.device}")

    def _convert_to_simplified(self, text: str) -> str:
        """
        将繁体中文转换为简体中文

        Args:
            text: 输入文本

        Returns:
            简体中文文本
        """
        if not ZHCONV_AVAILABLE or not text:
            return text

        try:
            # 使用zhconv进行繁简转换
            simplified = convert(text, "zh-cn")
            if simplified != text:
                logger.info(f"繁简转换: {text} -> {simplified}")
            return simplified
        except Exception as e:
            logger.warning(f"繁简转换失败: {e}")
            return text

    def load_model(self) -> bool:
        """加载Whisper模型"""
        try:
            logger.info(f"正在加载Whisper模型: {self.model_size}")

            # 如果设备是auto，自动选择
            if self.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"自动选择设备: {device}")
            else:
                device = self.device

            self.model = WhisperModel(
                model_size_or_path=self.model_size,
                device=device,
                compute_type=self.compute_type,
                download_root=str(self.model_path),
            )

            logger.info("Whisper模型加载成功")
            return True

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False

    def transcribe_audio_data(
        self, audio_data: np.ndarray, sample_rate: int = 16000
    ) -> Optional[str]:
        """
        转录音频数据

        Args:
            audio_data: 音频数据 (numpy array)
            sample_rate: 采样率

        Returns:
            识别的文本，失败返回None
        """
        if self.model is None:
            if not self.load_model():
                return None

        try:
            # 将音频数据保存为临时文件
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_filename = temp_file.name
                self.audio_recorder.save_to_wav(audio_data, temp_filename)

            # 使用faster-whisper进行转录
            segments, info = self.model.transcribe(
                temp_filename,
                language=self.language,
                beam_size=5,
                best_of=5,
                temperature=0.0,
                condition_on_previous_text=False,
            )

            # 合并所有段落的文本
            full_text = ""
            for segment in segments:
                full_text += segment.text

            # 清理临时文件
            Path(temp_filename).unlink(missing_ok=True)

            # 清理文本并进行繁简转换
            text = full_text.strip()
            if text:
                # 进行繁简转换
                simplified_text = self._convert_to_simplified(text)
                logger.info(f"识别结果: {simplified_text}")
                return simplified_text
            else:
                logger.warning("未识别到有效文本")
                return None

        except Exception as e:
            logger.error(f"语音识别失败: {e}")
            return None

    def record_and_transcribe(self) -> Optional[str]:
        """
        录制音频并进行语音识别 - 使用手动控制模式

        Returns:
            识别的文本，失败返回None
        """
        try:
            import numpy as np
            import sounddevice as sd

            print("🎤 准备开始录制...")
            input("按 Enter 开始录制: ")

            print("🔴 录制中... 按 Enter 停止录制")

            # 录制参数
            sample_rate = 16000
            channels = 1
            recording = True
            audio_data = []

            def audio_callback(indata, frames, time, status):
                if recording:
                    audio_data.append(indata.copy())

            # 开始录制流
            with sd.InputStream(
                samplerate=sample_rate,
                channels=channels,
                callback=audio_callback,
                dtype=np.float32,
            ):
                input()  # 等待用户按Enter停止

            recording = False
            print("⏹️ 录制停止!")

            if not audio_data:
                logger.warning("未录制到音频数据")
                return None

            # 合并音频数据并转换格式
            audio_float = np.concatenate(audio_data, axis=0)
            audio_int16 = (audio_float * 32767).astype(np.int16).flatten()

            duration = len(audio_int16) / sample_rate
            logger.info(f"录制完成，音频长度: {duration:.2f}秒")

            # 保存录音文件用于调试
            import datetime

            debug_dir = Path.cwd() / "data" / "voice" / "debug_recordings"
            debug_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_filename = debug_dir / f"recording_{timestamp}.wav"

            try:
                # 添加调试信息
                logger.info(
                    f"音频数据类型: {audio_int16.dtype}, 形状: {audio_int16.shape}, 范围: [{audio_int16.min()}, {audio_int16.max()}]"
                )

                with wave.open(debug_filename, "wb") as wf:
                    wf.setnchannels(channels)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(sample_rate)
                    wf.writeframes(audio_int16.tobytes())

                logger.info(f"🎵 录音已保存到: {debug_filename}")

                # 验证保存的文件大小
                file_size = debug_filename.stat().st_size
                logger.info(f"📁 文件大小: {file_size} 字节")
            except Exception as e:
                logger.warning(f"保存录音文件失败: {e}")
                import traceback

                logger.warning(f"详细错误: {traceback.format_exc()}")

            # 进行语音识别
            return self.transcribe_audio_data(audio_int16, sample_rate)

        except Exception as e:
            logger.error(f"录制和识别失败: {e}")
            import traceback

            logger.error(f"详细错误: {traceback.format_exc()}")
            return None

    def transcribe_audio_file(self, audio_file_path: str) -> Optional[str]:
        """
        转录音频文件

        Args:
            audio_file_path: 音频文件路径

        Returns:
            识别的文本，失败返回None
        """
        try:
            # 确保模型已加载
            if self.model is None:
                if not self.load_model():
                    return None

            if self.model is None:
                logger.error("模型加载失败，无法进行转录")
                return None

            # 直接使用Whisper处理音频文件，支持多种格式
            segments, info = self.model.transcribe(
                audio_file_path,
                language=self.language,
                beam_size=5,
                best_of=5,
                temperature=0.0,
                condition_on_previous_text=False,
            )

            # 记录语言检测信息
            logger.info(
                f"检测到语言: {info.language} (置信度: {info.language_probability:.2f})"
            )

            # 合并所有段落的文本
            full_text = ""
            segment_count = 0
            for segment in segments:
                full_text += segment.text
                segment_count += 1
                logger.debug(
                    f"段落 {segment_count}: [{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}"
                )

            logger.info(f"转录完成，共 {segment_count} 个段落")

            # 清理文本并进行繁简转换
            text = full_text.strip()
            if text:
                # 进行繁简转换
                simplified_text = self._convert_to_simplified(text)
                logger.info(f"识别结果: {simplified_text}")
                return simplified_text
            else:
                logger.warning("未识别到有效文本")
                return None

        except Exception as e:
            logger.error(f"音频文件转录失败: {e}")
            return None

    def __del__(self):
        """析构函数，清理资源"""
        if hasattr(self, "model") and self.model is not None:
            del self.model
