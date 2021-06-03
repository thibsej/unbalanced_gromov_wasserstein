from setuptools import find_packages, setup

setup(
    name="unbalanced_gromov_wasserstein",
    version="0.1.0",
    description="Computation of Unbalanced Gromov-Wasserstein distances using entropic regularization",
    author='Thibault Sejourne',
    author_email='thibault.sejourne@ens.fr',
    url='https://github.com/thibsej/unbalanced_gromov_wasserstein',
    packages=find_packages(),
    install_requires=[
              'numpy',
              'torch'
          ],
    license="MIT",
)
