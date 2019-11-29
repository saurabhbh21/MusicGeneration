from setuptools import setup

with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()


setup(
   name='MusicGeneration',
   version='1.0',
   description='Generate Music with GAN',
   license="",
   packages=['musicgeneration'],  #same as name
   install_requires=[required], #external packages as dependencies from requirements.txt
   scripts=[
            'scripts/directory.sh',
            
           ]
)