envs=Humanoid-v5
model=trpo
platform=cuda # or cpu
trials=1000
store_optuna=True

export JAX_PLATFORMS=$(platform)

default: install

train:
	@$(MAKE) fix
	@mkdir -p .logs
	@mkdir -p weights
	@. .venv/bin/activate && PYTHONPATH=. python train.py --model $(model) --envs $(envs) \
		--trials $(trials) --store-optuna $(store_optuna) \
		2>&1 | tee .logs/$(model)_$(envs)_$(platform)_$(shell date +'%Y%m%d').log

evaluate:
	@$(MAKE) fix
	@. .venv/bin/activate && PYTHONPATH=. python evaluate.py --model $(model) --envs $(envs)

install-ubuntu:
	@if [ "$$(uname -s)" = "Linux" ] && [ -f "/etc/os-release" ] && grep -iq "ubuntu" /etc/os-release; then \
		if ! python3.11 --version 2>/dev/null | grep -q "3.11.6"; then \
			echo "Installing Python 3.11.6..."; \
			sudo apt update; \
			sudo apt install -y wget build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libsqlite3-dev libreadline-dev libffi-dev curl libbz2-dev libnvinfer-dev libnvinfer-plugin-dev; \
			wget https://www.python.org/ftp/python/3.11.6/Python-3.11.6.tgz; \
			tar -xf Python-3.11.6.tgz; \
			cd Python-3.11.6 && ./configure --enable-optimizations && make -j $$(nproc) && sudo make altinstall; \
			cd ..; \
			sudo rm -rf Python-3.11.6 Python-3.11.6.tgz; \
		else \
			echo "Python 3.11.6 is already installed."; \
		fi; \
	else \
		echo "This target is only for Ubuntu or compatible Linux distributions."; \
		exit 1; \
	fi

install:
	@echo "Installing dependencies..."
	@if ! python3.11 --version; then echo "Please install python3.11.6" && exit 1; fi
	@if [ ! -d ".venv" ]; then python3.11 -m venv .venv; fi
	@. .venv/bin/activate && pip install --upgrade pip
	@. .venv/bin/activate && pip install -Ur "requirements/$(platform).txt"

install-ext:
	@echo "Reinstalling policy_gradients_jax from source..."
	@if [ "$(platform)" = "cuda" ]; then \
		. .venv/bin/activate && pip install --force-reinstall "policy_gradients_jax[cuda] @ git+https://github.com/pre63/policy_gradients_jax.git"; \
	else \
		. .venv/bin/activate && pip install --force-reinstall "policy_gradients_jax @ git+https://github.com/pre63/policy_gradients_jax.git"; \
	fi

fix:
	@. .venv/bin/activate && (isort --multi-line=0 --line-length=100 . && black --line-length 160 .)

clean:
	@echo "Cleaning up..."
	@rm -rf .venv
	@rm -rf __pycache__
	@rm -rf ./**/__pycache__
	@rm -rf .optuna
	@rm -rf .training_logs