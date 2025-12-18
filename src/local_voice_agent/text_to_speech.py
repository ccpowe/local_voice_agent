"""
文本转语音模块 - 基于Kokoro TTS
"""

import logging
import os
import re
import time
from pathlib import Path
from typing import Final, List, Optional

import numpy as np
import soundfile as sf
import torch
from kokoro import KModel, KPipeline

DEFAULT_MODEL_REPO_ID: Final[str] = "hexgrad/Kokoro-82M-v1.1-zh"
DEFAULT_SAMPLE_RATE: Final[int] = 24000
DEFAULT_VOICE: Final[str] = "zf_001"
DEFAULT_SPEED: Final[float] = 1.0
DEFAULT_DEVICE: Final[str] = "auto"
DEFAULT_SILENCE_DURATION: Final[float] = 0.2

PATH_CONFIG = {"tts_output_dir": Path.cwd() / "data" / "voice" / "tts_out"}
LOG_LEVEL_NAME: Final[str] = "INFO"

# 配置日志
logging.basicConfig(level=getattr(logging, LOG_LEVEL_NAME))
logger = logging.getLogger(__name__)


class TextToSpeech:
    """文本转语音器，使用Kokoro TTS进行中文语音合成"""

    def __init__(
        self,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        device: Optional[str] = None,
        model_cache_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
    ):
        """
        初始化文本转语音器

        Args:
            voice: 语音模型 (zf_xxx表示女声, zm_xxx表示男声)
            speed: 语音速度倍率
            device: 设备类型 ("cpu", "cuda", "auto")
            model_cache_dir: 指定模型目录（HuggingFace Hub 缓存/下载目录）；不指定则使用 HuggingFace 默认缓存目录
            output_dir: 指定输出目录（默认: ./data/voice/tts_out）
        """

        self.voice: str = voice if voice is not None else DEFAULT_VOICE
        self.speed: float = float(speed) if speed is not None else DEFAULT_SPEED
        self.device: str = device if device is not None else DEFAULT_DEVICE
        self.sample_rate: int = DEFAULT_SAMPLE_RATE
        self.silence_duration: float = DEFAULT_SILENCE_DURATION
        self.repo_id: str = DEFAULT_MODEL_REPO_ID
        self.model_cache_dir = (
            str(Path(model_cache_dir).resolve()) if model_cache_dir else None
        )
        self._local_model_dir: Path | None = (
            Path(self.model_cache_dir) / "kokoro" / self.repo_id.replace("/", "--")
            if self.model_cache_dir
            else None
        )

        # 输出目录
        resolved_output_dir = (
            Path(output_dir).resolve()
            if output_dir
            else Path(PATH_CONFIG["tts_output_dir"]).resolve()
        )
        self.output_dir = resolved_output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化模型和管道
        self.model: KModel | None = None
        self.zh_pipeline: KPipeline | None = None
        self.en_pipeline: KPipeline | None = None

        logger.info(
            f"初始化TTS: 语音={self.voice}, 速度={self.speed}, 设备={self.device}"
        )

        # 可选：指定本地缓存目录（不指定则走 HuggingFace 默认缓存目录）
        self._setup_model_cache()

    def _resolve_device(self) -> torch.device:
        if self.device == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_str = self.device
        return torch.device(device_str)

    def _setup_model_cache(self):
        """配置本地模型目录（可选）"""
        if not self.model_cache_dir:
            logger.info("未指定 model_cache_dir，将使用 Kokoro/HuggingFace 默认缓存目录")
            return

        Path(self.model_cache_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"使用指定的模型/缓存目录: {self.model_cache_dir}")

    def _resolve_voice_arg(self, voice: str) -> str:
        if voice.endswith(".pt") or os.path.exists(voice):
            return voice
        local_model_dir = self._local_model_dir
        if local_model_dir is not None:
            local_voice_path = local_model_dir / "voices" / f"{voice}.pt"
            if local_voice_path.exists():
                return str(local_voice_path)
        return voice

    def _load_models(self) -> bool:
        """加载TTS模型"""
        try:
            if self.model is not None:
                return True

            logger.info(f"正在加载Kokoro TTS模型: {self.repo_id}")

            device = self._resolve_device()

            config_path: str | None = None
            model_path: str | None = None
            local_model_dir = self._local_model_dir
            if local_model_dir is not None:
                local_config = local_model_dir / "config.json"
                model_name = KModel.MODEL_NAMES.get(self.repo_id)
                local_model = (local_model_dir / model_name) if model_name else None
                if (
                    local_config.exists()
                    and local_model is not None
                    and local_model.exists()
                ):
                    config_path = str(local_config)
                    model_path = str(local_model)
                    logger.info(f"从本地目录加载 Kokoro 模型文件: {local_model_dir}")

            if config_path and model_path:
                self.model = (
                    KModel(repo_id=self.repo_id, config=config_path, model=model_path)
                    .to(device)
                    .eval()
                )
            else:
                self.model = KModel(repo_id=self.repo_id).to(device).eval()

            logger.info(f"模型加载成功: {self.repo_id}")

            # 英语管道（用于处理英文单词）
            self.en_pipeline = KPipeline(
                lang_code="a", repo_id=self.repo_id, model=False
            )

            # 中文管道
            def en_callable(text: str) -> str:
                """处理英文单词的发音"""
                if text.lower() == "kokoro":
                    return "kˈOkəɹO"
                en_pipeline = self.en_pipeline
                if en_pipeline is None:
                    raise RuntimeError("English pipeline is not initialized")
                return next(en_pipeline(text)).phonemes

            self.zh_pipeline = KPipeline(
                lang_code="z",
                repo_id=self.repo_id,
                model=self.model,
                en_callable=en_callable,
            )

            logger.info("Kokoro TTS模型加载成功")
            return True

        except Exception as e:
            logger.error(f"TTS模型加载失败: {e}")
            return False

    def _calculate_speed(self, text_length: int) -> float:
        """
        根据文本长度动态调整语音速度
        避免长文本语音过快的问题
        """
        base_speed = self.speed

        # 根据文本长度调整速度
        if text_length <= 50:
            return base_speed
        elif text_length <= 100:
            return base_speed * 0.95
        elif text_length <= 200:
            return base_speed * 0.9
        else:
            return base_speed * 0.85

    def _preprocess_text(self, text: str) -> str:
        """
        预处理文本，去除可能导致TTS中断的字符

        Args:
            text: 原始文本

        Returns:
            处理后的文本
        """
        if not text:
            return ""

        # 去除所有换行符和回车符
        text = text.replace("\n", " ").replace("\r", " ")

        # 将多个连续空格合并为一
        text = re.sub(r"\s+", " ", text)

        # 去除首尾空格
        text = text.strip()

        return text

    def _save_audio_file(
        self, output_file: Optional[str], audio_data: np.ndarray
    ) -> str:
        # 生成输出文件路径 （似乎能提取成一个）
        if output_file is None:
            timestamp = int(time.time())
            output_path = self.output_dir / f"tts_output_{timestamp}.wav"
        else:
            # 确保输出目录存在
            output_path = Path(output_file).resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存音频文件
        sf.write(str(output_path), audio_data, self.sample_rate)
        logger.info(f"语音合成完成: {output_path}")
        return str(output_path)

    def synthesize_text(
        self, text: str, output_file: Optional[str] = None
    ) -> Optional[str]:
        """
        将文本转换为语音

        Args:
            text: 要转换的文本
            output_file: 输出文件路径，如果为None则自动生成

        Returns:
            生成的音频文件路径，失败返回None
        """
        try:
            if not self._load_models():
                return None

            processed_text = self._preprocess_text(text)
            if not processed_text.strip():
                logger.warning("输入文本为空")
                return None

            logger.info(f"正在合成语音: {processed_text[:50]}...")

            # 计算动态语音速度
            speed = self._calculate_speed(len(processed_text))

            # 生成语音
            zh_pipeline = self.zh_pipeline
            if zh_pipeline is None:
                raise RuntimeError("Chinese pipeline is not initialized")
            voice_arg = self._resolve_voice_arg(self.voice)
            generator = zh_pipeline(processed_text, voice=voice_arg, speed=speed)
            result = next(generator)
            audio_data = result.audio
            audio_output_path = self._save_audio_file(output_file, audio_data)
            return audio_output_path

        except Exception as e:
            logger.error(f"语音合成失败: {e}")
            return None

    def synthesize_paragraphs(
        self, paragraphs: List[str], output_file: Optional[str] = None
    ) -> Optional[str]:
        """
        将多个段落转换为语音，段落之间插入静音

        Args:
            paragraphs: 段落列表
            output_file: 输出文件路径

        Returns:
            生成的音频文件路径，失败返回None
        """

        try:
            if not self._load_models():
                return None

            if not paragraphs or all(not p.strip() for p in paragraphs):
                logger.warning("输入段落为空")
                return None

            logger.info(f"正在合成 {len(paragraphs)} 个段落的语音")

            # 预处理所有段落
            processed_paragraphs = [self._preprocess_text(p) for p in paragraphs]
            processed_paragraphs = [
                p for p in processed_paragraphs if p.strip()
            ]  # 过滤空段落

            if not processed_paragraphs:
                logger.warning("预处理后没有有效的段落")
                return None

            zh_pipeline = self.zh_pipeline
            if zh_pipeline is None:
                raise RuntimeError("Chinese pipeline is not initialized")

            audio_segments: list[np.ndarray] = []
            silence_samples = int(self.silence_duration * self.sample_rate)

            for i, paragraph in enumerate(processed_paragraphs):
                logger.info(
                    f"处理第 {i + 1}/{len(processed_paragraphs)} 段: {paragraph[:50]}..."
                )

                # 计算动态语音速度
                speed = self._calculate_speed(len(paragraph))

                # 生成当前段落的语音
                voice_arg = self._resolve_voice_arg(self.voice)
                generator = zh_pipeline(paragraph, voice=voice_arg, speed=speed)
                result = next(generator)
                audio_data = result.audio

                # 添加段落间的静音（除了第一段）
                if i > 0 and silence_samples > 0:
                    silence = np.zeros(silence_samples)
                    audio_segments.append(silence)

                audio_segments.append(audio_data)

            # 合并所有音频段
            if audio_segments:
                combined_audio = np.concatenate(audio_segments)
                audio_output_path = self._save_audio_file(output_file, combined_audio)
                return audio_output_path
            else:
                logger.warning("没有生成任何音频内容")
                return None

        except Exception as e:
            logger.error(f"多段落语音合成失败: {e}")
            return None

    def synthesize_long_text(
        self, text: str, max_chars: int = 150, output_file: Optional[str] = None
    ) -> Optional[str]:
        """
        将长文本自动分割并转换为语音

        Args:
            text: 要转换的长文本
            max_chars: 每段最大字符数，默认150（留有余量）
            output_file: 输出文件路径，如果为None则自动生成

        Returns:
            生成的音频文件路径，失败返回None
        """
        if not self._load_models():
            return None

        try:
            # 预处理文本
            processed_text = self._preprocess_text(text)
            if not processed_text.strip():
                logger.warning("输入文本为空")
                return None

            # 自动分割长文本
            segments = self._split_long_text(processed_text, max_chars)

            if len(segments) > 1:
                logger.info(
                    f"长文本自动分割为 {len(segments)} 段，每段约 {max_chars} 字符"
                )
                return self.synthesize_paragraphs(segments, output_file)
            else:
                # 文本较短，直接合成
                return self.synthesize_text(processed_text, output_file)

        except Exception as e:
            logger.error(f"长文本合成失败: {e}")
            return None

    def _split_long_text(self, text: str, max_chars: int = 150) -> List[str]:
        """
        将长文本按语义分割为合适长度的段落

        Args:
            text: 要分割的文本
            max_chars: 每段最大字符数

        Returns:
            分割后的段落列表
        """
        if len(text) <= max_chars:
            return [text]

        # 按句子结束符分割
        sentences = re.split(r"([。！？\.\!\?])", text)

        # 重建句子（包含结束符）
        sentences = [
            sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else "")
            for i in range(0, len(sentences), 2)
            if sentences[i].strip()
        ]

        segments = []
        current_segment = ""

        for sentence in sentences:
            # 如果单个句子就超过限制，按逗号分割
            if len(sentence) > max_chars:
                clauses = re.split(r"([，,])", sentence)
                clauses = [
                    clauses[i] + (clauses[i + 1] if i + 1 < len(clauses) else "")
                    for i in range(0, len(clauses), 2)
                    if clauses[i].strip()
                ]

                for clause in clauses:
                    if len(current_segment) + len(clause) <= max_chars:
                        current_segment += clause
                    else:
                        if current_segment.strip():
                            segments.append(current_segment.strip())
                        current_segment = clause
            else:
                # 正常句子处理
                if len(current_segment) + len(sentence) <= max_chars:
                    current_segment += sentence
                else:
                    if current_segment.strip():
                        segments.append(current_segment.strip())
                    current_segment = sentence

        # 添加最后一段
        if current_segment.strip():
            segments.append(current_segment.strip())

        # 如果还是太大，按字符强制分割
        final_segments = []
        for segment in segments:
            if len(segment) <= max_chars:
                final_segments.append(segment)
            else:
                # 强制按字符分割，避免截断单词
                for i in range(0, len(segment), max_chars):
                    chunk = segment[i : i + max_chars].strip()
                    if chunk:
                        final_segments.append(chunk)

        return final_segments

    def get_available_voices(self) -> List[str]:
        """
        获取可用的语音列表

        Returns:
            语音名称列表
        """
        # 兼容：如果用户把 voices 以平铺方式放到 model_cache_dir/voices 下，则从本地枚举
        if self.model_cache_dir:
            local_model_dir = self._local_model_dir
            if local_model_dir is not None:
                local_voices_dir = local_model_dir / "voices"
                if local_voices_dir.exists():
                    voices = [p.stem for p in local_voices_dir.glob("*.pt")]
                    if voices:
                        return sorted(voices)

            local_voices_dir = Path(self.model_cache_dir) / "voices"
            if local_voices_dir.exists():
                voices = [p.stem for p in local_voices_dir.glob("*.pt")]
                if voices:
                    return sorted(voices)

        # 默认返回Kokoro-82M-v1.1-zh模型的常用语音（实际可用性由Kokoro仓库决定）
        return [
            "zf_001",
            "zf_002",
            "zf_003",
            "zf_004",
            "zf_005",
            "zm_010",
            "zm_011",
            "zm_012",
            "zm_013",
            "zm_014",
            "af_maple",
            "af_sol",
            "bf_vale",
        ]

    def set_voice(self, voice: str) -> bool:
        """
        设置语音模型

        Args:
            voice: 语音名称

        Returns:
            设置是否成功
        """
        available_voices = self.get_available_voices()
        if voice in available_voices:
            self.voice = voice
            logger.info(f"语音已切换到: {voice}")
            return True
        else:
            logger.warning(f"语音 {voice} 不可用。可用语音: {available_voices[:5]}...")
            return False

    def __del__(self):
        """析构函数，清理资源"""
        if hasattr(self, "model") and self.model is not None:
            del self.model
        if hasattr(self, "zh_pipeline") and self.zh_pipeline is not None:
            del self.zh_pipeline
        if hasattr(self, "en_pipeline") and self.en_pipeline is not None:
            del self.en_pipeline
