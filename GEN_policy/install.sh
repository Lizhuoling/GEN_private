conda create -n GEN python=3.11
conda activate GEN

# Move to the ROOT of this repo
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
#pip install torch==2.7.0 torchvision==0.22.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch==2.3.1 torchvision==0.18.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install sonata
cd GEN/utils/sonata
pip install spconv-cu124 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.1+cu121.html
#pip install flash-attn==2.8.3 --no-build-isolation
MAX_JOBS=4 pip install flash-attn --no-build-isolation
pip install huggingface_hub==0.23.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
python setup.py develop
cd ../../..

# Install Concerto
cd GEN/utils/Concerto
python setup.py develop
cd ../../..

python setup.py develop
