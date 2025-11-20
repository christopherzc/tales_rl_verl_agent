This is a README detailing the instructions to run the verl-agent code with the integrated TALES environements. 

Please see README_verl_agent.md for the original readme at the time of pulling.

1.) Install uv as our venv manager: 
curl -LsSf https://astral.sh/uv/install.sh | sh

2.) Check it worked:
source $HOME/.local/bin/env
uv--version

3.) Make the venv and activate it:

uv venv --python 3.12
source .venv/bin/activate

4.) Do the installs from the original readme but with uv:

uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install packaging
uv pip install wheel
uv pip install flash-attn==2.7.4.post1 --no-build-isolation
uv pip install -e .
uv pip install vllm==0.8.5
uv pip install gymnasium==0.29.1
uv pip install stable-baselines3==2.6.0
uv pip install "git+https://github.com/microsoft/tale-suite.git@tt_split"
uv pip install "peft==0.8.2"
(unpinned version causes an error)
apt-get update && apt-get install default-jre default-jdk
(For scienceworld)




docker run -it \
  --name trl \
  --gpus all \
  -v "$PWD":/workspace \
  -w /workspace \
  python:3.12 \
  bash