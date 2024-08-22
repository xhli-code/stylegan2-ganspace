
# Environment Installation

Install the necessary packages using the following commands:

```bash
conda create -n gan_work python=3.7
conda activate gan_work
pip install torch==1.5.0+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install tensorflow-gpu==1.14.0
pip install protobuf==3.20 Pillow requests
pip install matplotlib
pip install scipy
pip install scikit-image
pip install boto3
pip install tqdm
pip install scikit-learn
pip install fbpca
pip install ganspace\windows\pycuda-2019.1.2+cuda101-cp37-cp37m-win_amd64.whl
pip install PyOpenGL
pip install pyopengltk
pip install glumpy

```

# Training

1. Prepare your data by placing it in a designated folder, then execute the following commands:
```bash
cd stylegan2-ganspace/stylegan2
python dataset_tool.py create_from_images ~/datasets/my-custom-dataset ~/my-custom-images
```

2. Once the data is prepared, begin training by running the following command:
```bash
python run_training.py --num-gpus=1 --data-dir=~/datasets --config=config-f --dataset=shdgzma --mirror-augment=true
```

# Model Conversion

To convert model weights from TensorFlow to PyTorch, execute the following command:
```bash
cd stylegan2-ganspace/ganspace/models/stylegan2/stylegan2-pytorch
python convert_weight.py --repo ~/stylegan2 network-snapshot.pkl
```

# Usage

1. Move the `.pt` file into the specified folder by running the following commands:
```bash
mkdir stylegan2-ganspace/ganspace/models/checkpoints
mv network-snapshot.pt stylegan2-ganspace/ganspace/models/checkpoints
```

2. To explore the model interactively, run the following commands:
```bash
cd stylegan2-ganspace/ganspace
python interactive.py --model=StyleGAN2 --class=shdgzma --layer=style --use_w -n=1_000_000 -b=10_000
```

# Related Projects

- [**StyleGAN2 - Official TensorFlow Implementation**](https://github.com/NVlabs/stylegan2)
- [**GANSpace: Discovering Interpretable GAN Controls**](https://github.com/harskish/ganspace)