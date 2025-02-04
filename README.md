# Obscure
Say goodbye to boring camera backgrounds with Obscure!

## Installation
Install opencv 4.11 \
Get the latest release (built with opencv 4.11): [Download](https://github.com/lef-fan/obscure/releases/latest)

## Development
Install latest opencv. \
Clone the repo with the required submodules:
```
git clone --recursive git@github.com:lef-fan/obscure.git
```
or if you have already cloned the project without the submodules needed, then:
```
git submodule update --init --recursive
```
Update the submodules (necessary for libtorch library):
```
sh update_submodules.sh
```
Install required libs for python:
```
pip install -r python/requirements.txt
```
Convert the targeted model:
```
python python/model_conversion.py
```
Build the plugin:
```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```
This will install the plugin under:
```
$ENV{HOME}/.config/obs-studio/plugins
```
Open OBS and choose Obscure under camera effect filters.

## Usage
...

## Documentation
Work in progress...

## Contributions
🌟 We'd love your contribution! Please submit your changes via pull request to join in the fun! 🚀

## Disclaimer
...

## Acknowledgments
...

## License Information

### ❗ Important Note:
While this project is licensed under GNU AGPLv3, the usage of some of the components it depends on might not and they will be listed below: