echo "Building virtualenv..."
rm -rf venv
pip install virtualenv
python -m virtualenv venv
source venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .

bash ./scripts/download_models.sh
bash ./scripts/download_sample_data.sh
mkdir data/plotting_data/cache