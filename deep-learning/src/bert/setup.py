from setuptools import setup
setup(name='bert',
    version='0.0.1',
    description='load bug reports datasets',
    author='Luiz Alberto',
    author_email='gomes.luiz@gmail.com',
    packages=['data', 'features'],
    install_requires=['pandas'],
    zip_false=False
)
