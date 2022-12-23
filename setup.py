from setuptools import setup

setup(
    name="unbalancedgw",
    distname="",
    version='0.2.0',
    description="Computation of Unbalanced Gromov-Wasserstein distances using "
                "entropic regularization",
    author='Thibault Sejourne',
    author_email='thibault.sejourne@ens.fr',
    url='https://github.com/thibsej/unbalanced_gromov_wasserstein',
    packages=['unbalancedgw', 'unbalancedgw.tests'],
    install_requires=[
              'numpy',
              'torch'
          ],
    license="MIT",
)
