from os import path
import setuptools

this_directory = path.abspath(path.dirname(__file__))

with open(path.join(this_directory, 'requirements.txt'),
          encoding='utf-8') as f:
    requirements = f.read().splitlines()

    
    
# get __version__ from _version.py
ver_file = path.join(this_directory, 'eptk', '__version__.py')
with open(ver_file) as f:
    exec(f.read())

# readme path
readme_file = path.join(this_directory,'README.md')    

setuptools.setup(
    name = "eptk",
    version = __version__,
    url = "https://github.com/samy101/eptk",
    author = "eptk developers",
    author_email = "-",
    keywords = ['energy prediction'],
    description = "A Python Tool Kit for Energy Prediction. Authors : Pandarasamy Arjunan, Hardik Prabhu Email: pandarasamya@iiitd.ac.in,  hardik.prabhu@gmail.com",
    long_description = open(readme_file).read(),
    license = "Apache 2.0",
    packages = setuptools.find_packages(),      
    install_requires = requirements,
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    include_package_data=True,
    package_data={'': ['*.json']}
    
)
