from setuptools import setup, find_packages

setup(
    name="dream-rsi",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pyyaml>=5.4.1",
        "click>=8.0.0",
    ],
    entry_points={
        "console_scripts": [
            "dream-rsi = dream_rsi.cli:main",
            "dream-rsi-mcp = dream_rsi.mcp_server:main",
        ],
    },
)
