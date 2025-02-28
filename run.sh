python -m venv venv

source ./venv/bin/activate

pip install -r requirements.txt

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

python train.py
