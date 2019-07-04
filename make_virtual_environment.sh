# Create a Python virtual environment in ./venv/
virtualenv venv
# Activate the virtual environment
source venv/bin/activate
# Need to install NumPy and SciPy before running through requirements file
pip install numpy==1.13.1
pip install scipy==0.19.1
pip install -r requirements.txt
