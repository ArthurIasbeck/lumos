from setuptools import setup, find_packages

setup(
    name="lumos",
    version="0.0.1",
    packages=find_packages(),
    description="Pacote empregado na resolução de problemas de otimização por meio do emprego de AGs.",
    author="Arthur Iasbeck",
    author_email="arthuriasbeck@ufu.br",
    url="https://github.com/ArthurIasbeck/lumos",
    install_requires=[
        "contourpy==1.2.1",
        "cycler==0.12.1",
        "fonttools==4.53.0",
        "kiwisolver==1.4.5",
        "loguru==0.7.2",
        "matplotlib==3.9.0",
        "numpy==2.0.0",
        "packaging==24.1",
        "pillow==10.3.0",
        "pyparsing==3.1.2",
        "PyQt5==5.15.10",
        "PyQt5-Qt5==5.15.2",
        "PyQt5-sip==12.13.0",
        "python-dateutil==2.9.0.post0",
        "six==1.16.0",
        "toml==0.10.2",
    ],
)
