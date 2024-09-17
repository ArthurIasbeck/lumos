from setuptools import setup, find_packages

with open("requirements.txt") as f:
    REQUIRED = f.read().splitlines()

setup(
    name="lumos",
    version="0.0.1",
    packages=find_packages(),
    description="Pacote empregado na resolução de problemas de otimização por meio do emprego de AGs.",
    author="Arthur Iasbeck",
    author_email="arthuriasbeck@ufu.br",
    url="https://github.com/ArthurIasbeck/lumos",
    install_requires=REQUIRED,
)
