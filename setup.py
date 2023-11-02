from setuptools import setup

setup(name='slmOptim',
      version='0.1',
      description='A collection of packages for an optimization problem using a Spatial Light Modulator',
      author='Alexandros Tavernarakis',
      author_email='alexandre.tavernarakis@universite-paris-saclay.fr',
      packages=['optimization', 'patternSLM', 'zeluxPy'],
      py_modules=['.slmpy'],
      install_requires=['numpy', 'pandas', 'scipy'],
      scripts=["speckle_analysis.py", "transmission_matrix.py", "optimizataion.py"]
      )

