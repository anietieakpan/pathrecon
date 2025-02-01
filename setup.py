# setup.py

from setuptools import setup, find_packages

setup(
    name="ketu",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'flask',
        'sqlalchemy',
        'psycopg2-binary',
        'python-dotenv',
        'influxdb-client',
        'opencv-python',
        'numpy',
        'nomeroff-net'
    ],
    extras_require={
        'dev': [
            'pytest',
            'pytest-cov',
            'flake8',
            'black',
            'mypy'
        ]
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="License plate detection and analysis system",
    keywords="license-plate, detection, analysis, computer-vision",
    python_requires='>=3.7',
)