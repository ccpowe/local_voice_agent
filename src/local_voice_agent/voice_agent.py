from __future__ import annotations

import asyncio
import datetime
import logging
from pathlib import Path

from aioconsole import ainput
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from .audio_utils import AudioPlayer, AudioRecorder
from .settings import Settings
from .speech_recognition import SpeechRecognizer
from .text_to_speech import TextToSpeech


def _build_chat_model(settings: Settings) -> ChatOpenAI:
    if settings.base_url:
        return ChatOpenAI(
            model_name=settings.model_name,
            openai_api_key=settings.api_key,
            openai_api_base=settings.base_url,
        )
    return ChatOpenAI(model_name=settings.model_name, openai_api_key=settings.api_key)


def _init_speech_recognizer(model_dir: Path) -> SpeechRecognizer:
    print("🔄 正在初始化语音识别器...")
    recognizer = SpeechRecognizer(
        model_size="small",
        device="auto",
        language="zh",
        model_cache_dir=model_dir,
    )

    print("📥 正在加载语音识别模型，请稍候...")
    if recognizer.load_model():
        print("✅ 语音识别模型加载成功")
    else:
        print("❌ 语音识别模型加载失败")
    return recognizer


def _init_tts(model_dir: Path, tts_out_dir: Path) -> tuple[TextToSpeech | None, bool]:
    print("🔄 正在初始化TTS语音合成器...")
    try:
        tts_synthesizer = TextToSpeech(
            voice="zf_001",
            speed=1.0,
            device="auto",
            model_cache_dir=model_dir,
            output_dir=tts_out_dir,
        )

        print("📥 正在加载TTS模型，请稍候...")
        if tts_synthesizer._load_models():
            print("✅ TTS语音合成器初始化成功")
            return tts_synthesizer, True

        print("❌ TTS模型加载失败")
        return tts_synthesizer, False
    except Exception as exc:
        print(f"⚠️  TTS初始化失败: {exc}")
        return None, False


def _print_usage(tts_available: bool) -> None:
    print("🎤 语音地理分析助手已启动!")
    print("💡 使用说明:")
    print("   - 输入 's' 或 'speech' 开始语音输入")
    print("   - 直接输入文字进行文字对话")
    if tts_available:
        print("   - 输入 'tts:on' 开启语音输出，'tts:off' 关闭语音输出")
        print("   - 输入 'voice:女声' 或 'voice:男声' 切换语音类型")
    print("   - 输入 '再见' 退出程序")
    print("-" * 50)


async def _record_and_transcribe(
    audio_recorder: AudioRecorder, speech_recognizer: SpeechRecognizer
) -> str | None:
    print("\n🎤 语音输入模式")
    print("💡 操作说明: 按Enter开始录制，说话后再按Enter停止")

    audio_data = audio_recorder.record_manual()
    if audio_data is None:
        print("❌ 未录制到音频，请重试")
        return None

    debug_dir = Path.cwd() / "data" / "voice" / "audio_cache"
    debug_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = debug_dir / f"recording_{timestamp}.wav"
    audio_recorder.save_to_wav(audio_data, wav_path)

    return speech_recognizer.transcribe_audio_file(wav_path)


async def async_main() -> None:
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = Settings()

    audio_recorder = AudioRecorder(sample_rate=16000, channels=1)
    audio_player = AudioPlayer()

    data_dir = Path.cwd() / "data"
    model_dir = data_dir / "model"
    tts_out_dir = data_dir / "voice" / "tts_out"

    model = _build_chat_model(settings)
    agent = create_agent(model=model, tools=[], system_prompt="健康助手小v")

    speech_recognizer = _init_speech_recognizer(model_dir)
    tts_synthesizer, tts_available = _init_tts(model_dir, tts_out_dir)

    _print_usage(tts_available)

    tts_enabled = False

    while True:
        user_input = await ainput("请输入你的问题 (或输入's'进行语音输入): ")
        normalized = user_input.strip().lower()

        match normalized:
            case "exit" | "quit" | "再见":
                break
            case "tts:on" if tts_available:
                tts_enabled = True
                print("🔊 语音输出已开启")
                continue
            case "tts:off" if tts_available:
                tts_enabled = False
                print("🔇 语音输出已关闭")
                continue
            case "voice:女声" | "voice:zf" if tts_available:
                if tts_synthesizer and tts_synthesizer.set_voice("zf_001"):
                    print("🎵 已切换到女声")
                continue
            case "voice:男声" | "voice:zm" if tts_available:
                if tts_synthesizer and tts_synthesizer.set_voice("zm_010"):
                    print("🎵 已切换到男声")
                continue
            case "s" | "speech" | "语音":
                transcribed = await _record_and_transcribe(
                    audio_recorder=audio_recorder,
                    speech_recognizer=speech_recognizer,
                )
                if not transcribed:
                    print("❌ 语音识别失败，请重试")
                    continue
                print(f"📝 识别结果: {transcribed}")
                user_input = transcribed

        print("AI: ", end="", flush=True)
        result = await agent.ainvoke(
            {"messages": [{"role": "human", "content": user_input}]},
            config={"configurable": {"thread_id": "1"}, "recursion_limit": 100},
        )

        ai_response = result["messages"][-1].content
        print(ai_response)
        print()

        if not (tts_available and tts_enabled):
            continue

        if not tts_synthesizer:
            print("TTS 合成器不可用")
            continue

        if not (ai_text := str(ai_response).strip()):
            continue

        print("🔊 正在生成语音回复...")
        try:
            audio_file = tts_synthesizer.synthesize_long_text(ai_text)
            if not audio_file:
                print("❌ 语音生成失败")
                continue
            print(f"🎵 语音回复已生成: {audio_file}")
            audio_player.play_wav_file(audio_file)
        except Exception as exc:
            print(f"❌ TTS错误: {exc}")

    print("👋 再见!")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
