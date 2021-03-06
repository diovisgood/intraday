import setuptools

with open('README.md', 'r') as file:
    long_description = file.read()

setuptools.setup(
    name='intraday',
    version='0.0.236',
    author='Pavel B. Chernov',
    author_email='pavel.b.chernov@gmail.com',
    description='Exchange/Broker Simulation Environment for Intraday Trading Models',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords='exchange broker trading gym environment simulation episode',
    url='https://github.com/diovisgood/intraday',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: Creative Commons',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
    install_requires=[
        'gym>=0.17.2',
        'numpy>=1.18.1',
        'arrow>=0.13.1',
        'feather-format>=0.4.1',
],
    extras_require={
        'pyglet': ['pyglet>=1.5.16'],
    },
)
