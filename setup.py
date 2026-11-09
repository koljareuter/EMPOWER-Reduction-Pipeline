#!/usr/bin/env python
import os, sys
import numpy
from os.path import join as pjoin
import shutil
import glob

try:
    from setuptools import setup, Extension, Command
    from setuptools.command.build_ext import build_ext as _build_ext
    from setuptools.command.build import build
except ImportError:
    from distutils.core import setup, Extension, Command
    from distutils.command.build_ext import build_ext as _build_ext
    from distutils.command.build import build

class CleanCommand(Command):
    """Custom distutils command to clean the .so and .pyc files."""

    user_options = [("all", "a", "")]

    def initialize_options(self):
        self.all = True
        self._clean_me = []
        self._clean_trees = []
        self._clean_exclude = []

        for root, dirs, files in list(os.walk('pyspark')):
            for f in files:
                if f in self._clean_exclude:
                    continue
                if os.path.splitext(f)[-1] in ('.pyc', '.so', '.o',
                                               '.pyo',
                                               '.pyd', '.c', '.orig'):
                    self._clean_me.append(pjoin(root, f))
            for d in dirs:
                if d == '__pycache__':
                    self._clean_trees.append(pjoin(root, d))

        for d in ('build', 'dist', ):
            if os.path.exists(d):
                self._clean_trees.append(d)

    def finalize_options(self):
        pass

    def run(self):
        for clean_me in self._clean_me:
            try:
                os.unlink(clean_me)
            except Exception:
                pass
        for clean_tree in self._clean_trees:
            try:
                import shutil
                shutil.rmtree(clean_tree)
            except Exception:
                pass


try:
    from Cython.Build import cythonize
except ImportError:
    raise ImportError("pyspark requires cython to install")


class build_ext(_build_ext):
    def build_extension(self, ext):
        _build_ext.build_extension(self, ext)
        

if __name__ == "__main__":

    include_dirs = ["include",
                    numpy.get_include(),
                   ]

    cmodules = []
    cmodules += [Extension("pyspark.utils.interpolation", 
                           ["pyspark/utils/interpolation.pyx"], 
                           include_dirs=include_dirs)]
    ext_modules = cythonize(cmodules)

    scripts = ['scripts/'+file for file in os.listdir('scripts/')]  

    cmdclass = {'clean': CleanCommand,
                'build_ext': build_ext}

  
    with open('pyspark/_version.py') as f:
        exec(f.read())

    setup(
        name = "pyspark",
        url="https://github.com/koljareuter/EMPOWER-Reduction-Pipeline",
        version= __version__,
        author="Trevor Mendel",
        author_email="trevor.mendel@anu.edu.au",
        ext_modules = ext_modules,
        cmdclass = cmdclass,
        scripts = scripts, 
        packages=["pyspark",
                  "pyspark.detector",
                  "pyspark.utils",
                  "pyspark.calibration",
                  "pyspark.cubes",
                  "pyspark.sky",
                  "pyspark.combine"],
        license="LICENSE",
        description="Reduction routines for KMOS",
        package_data={"": ["README.md", "LICENSE"],
                      "pyspark": ["data/superdark_coeffs_mean.fits.fz", 
                                  "data/superbad_60s.fits.fz",
                                  "data/ZAPv2_SVD_IZ_n500004.fits",
                                  "data/ZAPv2_SVD_IZ_n500004_nomean.fits",
                                  "data/ZAPv2_SVD_YJ_n500004.fits",
                                  "data/ZAPv2_SVD_H_n500004.fits",
                                  "data/ZAPv2_SVD_H_n500004_nomean.fits",
                                  "data/ZAPv2_SVD_K_n500004.fits",
                                  "data/ZAPv1_SVD_YJ_n500004.fits",
                                  "data/ZAPv1_SVD_H_n500004.fits",
                                  "data/ZAPv1_SVD_K_n500004.fits",
                                  "data/ZAPv2_SVD_YJ_n9504_e600.fits",
                                  "data/RESPONSE_IZIZIZ.fits",
                                  "data/RESPONSE_YJYJYJ.fits",
                                  "data/RESPONSE_HHH.fits",
                                  "data/RESPONSE_HKHKHK.fits",
                                  "data/RESPONSE_KKK.fits",
                                  ]},
        include_package_data=True,
        zip_safe=False,
    )

