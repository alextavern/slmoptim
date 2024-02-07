from setuptools import setup

setup(name='slmoptim',
      version='0.1',
      description='A collection of packages for an optimization problem using a Spatial Light Modulator',
      author='Alexandros Tavernarakis',
      author_email='alexandre.tavernarakis@universite-paris-saclay.fr',
      packages=['initializer', 'loader', 'optimizer', 'utils'],
      py_modules=['.slmpy'],
      install_requires=['numpy', 'pandas', 'scipy'],
      scripts=["scripts/speckle_analysis.py", "scripts/transmission_matrix.py"]
      )

