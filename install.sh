sudo apt-get update
sudo apt-get install screen

pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu130
pip install transformers datasets wandb hydra-core omegaconf jaxtyping kornia
pip install -U Pillow
pip install pytest  # for the test suite (tests/)
sudo pip uninstall --yes flash_attn
