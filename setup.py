from setuptools import setup

setup(name='torchseal',
      version='1.0',
      description='A utility for finding memory leaks in pytorch',
      author='Ben Heil',
      author_email='ben.jer.heil@gmail.com',
      url='https://github.com/ben-heil/torchseal',
      packages=['torchseal'],
      install_requires=['torch'],
      extras_require={
          'tests': ['torch',
                    'torchvision',
                    'pytest'
                    ]
      },
      )
