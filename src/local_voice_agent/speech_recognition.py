"""语音识别模块 - 基于faster-whisper"""

import logging
from pathlib import Path

from faster_whisper import WhisperModel
from zhconv import convert  # 处理繁体简体转换

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
        model_size: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
        language: str | None = None,
        model_cache_dir: str | Path | None = None,
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

        # 模型下载路径
        base_cache_dir = (
            Path(model_cache_dir)
            if model_cache_dir
            else PATH_CONFIG["model_cache_dir"]
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

        try:
            # 使用zhconv进行繁简转换
            simplified = convert(text, "zh-cn")
            if simplified != text:
                logger.info(f"繁简转换: {text} -> {simplified}")
            return simplified
        except Exception as exc:
            logger.warning(f"繁简转换失败: {exc}")
            return text

    def load_model(self) -> bool:
        """加载Whisper模型"""
        try:
            logger.info(f"正在加载Whisper模型: {self.model_size}")

            device = self.device

            self.model = WhisperModel(
                model_size_or_path=self.model_size,
                device=device,
                compute_type=self.compute_type,
                download_root=str(self.model_path),
            )

            logger.info("Whisper模型加载成功")
            return True

        except Exception as exc:
            logger.error(f"模型加载失败: {exc}")
            return False

    def transcribe_audio_file(self, audio_file_path: str | Path) -> str | None:
        """
        转录音频文件

        Args:
            audio_file_path: 音频文件路径

        Returns:
            识别的文本，失败返回None
        """

        # 确保模型已加载
        if self.model is None and not self.load_model():
            return None
        if self.model is None:
            logger.error("模型未加载，无法进行转录")
            return None

        # 直接使用Whisper处理音频文件，支持多种格式
        segments, info = self.model.transcribe(
            str(audio_file_path),
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

        segment_texts: list[str] = []
        segment_count = 0
        for segment in segments:
            segment_texts.append(segment.text)
            segment_count += 1
            logger.debug(
                f"段落 {segment_count}: [{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}"
            )

        logger.info(f"转录完成，共 {segment_count} 个段落")

        # 清理文本并进行繁简转换
        text = "".join(segment_texts).strip()
        if text:
            # 进行繁简转换
            simplified_text = self._convert_to_simplified(text)
            logger.info(f"识别结果: {simplified_text}")
            return simplified_text
        else:
            logger.warning("未识别到有效文本")
            return None

    def __del__(self):
        """析构函数，清理资源"""
        if hasattr(self, "model") and self.model is not None:
            del self.model
