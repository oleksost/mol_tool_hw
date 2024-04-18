from setuptools import setup, find_packages

def load_requirements(filename='requirements.txt'):
    with open(filename, 'r') as f:
        lineiter = (line.strip() for line in f)
        return [line for line in lineiter if line and not line.startswith("#")]

setup(
    name="mol_tools",
    version="0.1.0",
    package_dir={"": "src"},  # This tells setuptools that package modules are under src
    packages=find_packages(where="src"),  # This finds package directories under src
    install_requires=load_requirements(),
    python_requires='>=3.9'
)