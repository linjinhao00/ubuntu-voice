from setuptools import find_packages, setup


setup(
    name="bytecli",
    version="1.1.6",
    description="Local voice-to-text dictation tool for Ubuntu/Linux",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    author="ByteCLI Contributors",
    packages=find_packages(include=["bytecli", "bytecli.*"]),
    include_package_data=True,
    package_data={"bytecli": ["data/*.css", "data/*.xml", "i18n/*.json"]},
    python_requires=">=3.10",
    install_requires=[
        "faster-whisper",
        "openai-whisper",
        "python-xlib",
        "sounddevice",
        "pulsectl",
        "numpy",
        "dbus-python",
        "PyGObject",
    ],
    extras_require={
        "sensevoice": ["funasr-onnx"],
        "funasr": ["funasr>=1.3.3", "torchaudio==2.5.1"],
        "qwen": ["qwen-asr>=0.0.6", "Pillow>=10"],
        "eval": ["jiwer>=4.0.0", "soundfile", "webrtcvad"],
        "silero": ["silero-vad"],
        "glm": ["transformers", "torch"],
    },
    entry_points={
        "console_scripts": [
            "bytecli-service=bytecli.service.main:main",
            "bytecli-indicator=bytecli.indicator.main:main",
            "bytecli-settings=bytecli.settings.main:main",
            "bytecli-asr-eval=bytecli.eval.asr_eval:main",
        ],
    },
)
