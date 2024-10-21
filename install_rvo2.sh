#!/bin/bash



os_type=$(uname)
if [[ "$os_type" == "Linux" ]]; then
    echo "This is a Linux OS."
    
elif [[ "$os_type" == "Darwin" ]]; then
    echo "This is macOS."
    osx_version=$(sw_vers -productVersion)
    export MACOSX_DEPLOYMENT_TARGET=$osx_version
else
    echo "Cannot support installation of rvo2 in Win or Other OS"
fi


echo $MACOSX_DEPLOYMENT_TARGET

git clone https://github.com/sybrenstuvel/Python-RVO2.git
cd Python-RVO2
python setup.py build


# If this command fails to execute, please type this command directly in the Python-RVO2 directory.
python setup.py install


