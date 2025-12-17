import asyncio
import datetime
import logging
import os
from pathlib import Path

from aioconsole import ainput
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from .audio_utils import AudioPlayer, AudioRecorder
from .speech_recognition import SpeechRecognizer
from .text_to_speech import TextToSpeech

# 禁用httpx的INFO日志
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()

audio_recorder = AudioRecorder(sample_rate=16000, channels=1)
audio_player = AudioPlayer()


# 初始化模型
model = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),  # type:ignore
    api_key=os.getenv("API_KEY"),  # type:ignore
    base_url=os.getenv("BASE_URL"),  # type:ignore
)


# 初始化检查点
checkpoint = InMemorySaver()

agent = create_agent(model, tools=[], system_prompt="健康助手小v")

# 统一数据目录
DATA_DIR = Path.cwd() / "data"
MODEL_DIR = DATA_DIR / "model"
VOICE_DIR = DATA_DIR / "voice"
TTS_OUT_DIR = VOICE_DIR / "tts_out"

# 初始化语音识别器
print("🔄 正在初始化语音识别器...")
speech_recognizer = SpeechRecognizer(
    model_size="small",  # 可选: "tiny", "base", "small", "medium", "large"
    device="auto",  # 自动选择CPU或GPU
    language="zh",  # 中文识别
    model_cache_dir=str(MODEL_DIR),
)

# 立即加载语音识别模型
print("📥 正在加载语音识别模型，请稍候...")
if speech_recognizer.load_model():
    print("✅ 语音识别模型加载成功")
else:
    print("❌ 语音识别模型加载失败")

# 初始化TTS语音合成器
print("🔄 正在初始化TTS语音合成器...")
try:
    tts_synthesizer = TextToSpeech(
        voice="zf_001",  # 默认女声，可选: zf_xxx(女声), zm_xxx(男声)
        speed=1.0,  # 语音速度
        device="auto",  # 自动选择设备
        model_cache_dir=str(MODEL_DIR),
        output_dir=str(TTS_OUT_DIR),
    )
    # 立即加载TTS模型
    print("📥 正在加载TTS模型，请稍候...")
    if tts_synthesizer._load_models():
        TTS_AVAILABLE = True
        print("✅ TTS语音合成器初始化成功")
    else:
        TTS_AVAILABLE = False
        print("❌ TTS模型加载失败")
except Exception as e:
    tts_synthesizer = None
    TTS_AVAILABLE = False
    print(f"⚠️  TTS初始化失败: {e}")


async def main():
    print("🎤 语音地理分析助手已启动!")
    print("💡 使用说明:")
    print("   - 输入 's' 或 'speech' 开始语音输入")
    print("   - 直接输入文字进行文字对话")
    if TTS_AVAILABLE:
        print("   - 输入 'tts:on' 开启语音输出，'tts:off' 关闭语音输出")
        print("   - 输入 'voice:女声' 或 'voice:男声' 切换语音类型")
    print("   - 输入 '再见' 退出程序")
    print("-" * 50)

    user_input = await ainput("请输入你的问题 (或输入's'进行语音输入):")

    # TTS状态控制
    tts_enabled = False

    while user_input.lower() != "再见":
        # 检查TTS控制命令
        if TTS_AVAILABLE and user_input.lower() == "tts:on":
            tts_enabled = True
            print("🔊 语音输出已开启")
            user_input = await ainput("请输入你的问题 (或输入's'进行语音输入):")
            continue
        elif TTS_AVAILABLE and user_input.lower() == "tts:off":
            tts_enabled = False
            print("🔇 语音输出已关闭")
            user_input = await ainput("请输入你的问题 (或输入's'进行语音输入):")
            continue
        elif TTS_AVAILABLE and user_input.lower() in ["voice:女声", "voice:zf"]:
            if tts_synthesizer and tts_synthesizer.set_voice("zf_001"):
                print("🎵 已切换到女声")
            user_input = await ainput("请输入你的问题 (或输入's'进行语音输入):")
            continue
        elif TTS_AVAILABLE and user_input.lower() in ["voice:男声", "voice:zm"]:
            if tts_synthesizer and tts_synthesizer.set_voice("zm_010"):
                print("🎵 已切换到男声")
            user_input = await ainput("请输入你的问题 (或输入's'进行语音输入):")
            continue

        # 检查是否使用语音输入
        if user_input.lower() in ["s", "speech", "语音"]:
            print("\n🎤 语音输入模式")
            print("💡 操作说明: 按Enter开始录制，说话后再按Enter停止")

            audio_data = audio_recorder.record_manual()
            file_path = None
            if audio_data is not None:
                debug_dir = Path.cwd() / "data" / "voice" / "audio_cache"
                debug_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = debug_dir / f"recording_{timestamp}.wav"
                audio_recorder.save_to_wav(audio_data, str(file_path))

            recognized_text = (
                speech_recognizer.transcribe_audio_file(str(file_path))
                if file_path is not None
                else None
            )

            if recognized_text:
                print(f"📝 识别结果: {recognized_text}")
                user_input = recognized_text
            else:
                print("❌ 语音识别失败，请重试")
                user_input = await ainput("请输入你的问题 (或输入's'进行语音输入):")
                continue

        # 处理用户输入（文字或语音识别结果）
        print("AI: ", end="", flush=True)

        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config={"configurable": {"thread_id": "1"}, "recursion_limit": 100},
        )
        Ai_response = result["messages"][-1].content
        print(Ai_response)
        print()  # 在回答结束后换行

        # 如果启用了TTS，将AI回复转为语音
        if TTS_AVAILABLE and tts_enabled and Ai_response.strip():
            print("🔊 正在生成语音回复...")
            try:
                if tts_synthesizer:
                    audio_file = tts_synthesizer.synthesize_long_text(
                        Ai_response.strip()
                    )
                    if audio_file:
                        print(f"🎵 语音回复已生成: {audio_file}")
                        audio_player.play_wav_file(audio_file)
                    else:
                        print("❌ 语音生成失败")
                else:
                    print("TTS 合成器不可用")
            except Exception as e:
                print(f"❌ TTS错误: {e}")

        user_input = await ainput("请输入你的问题 (或输入's'进行语音输入):")


if __name__ == "__main__":
    asyncio.run(main())
