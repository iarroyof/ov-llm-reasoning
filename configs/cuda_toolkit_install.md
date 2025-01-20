# CUDA Toolkit Installation

1. First, let's completely remove and reinstall the NVIDIA container stack:
```bash
# Remove existing installations
sudo apt remove --purge nvidia-container-toolkit nvidia-container-runtime nvidia-docker2
sudo apt autoremove

# Clean up any remaining configuration
sudo rm -rf /etc/nvidia-container-runtime/
sudo rm -rf /etc/docker/daemon.json

# Reinstall from scratch
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
```

2. Configure the runtime with more permissive settings:
```bash
# Configure the runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Create a new daemon.json with explicit settings
sudo tee /etc/docker/daemon.json <<EOF
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia",
    "exec-opts": ["native.cgroupdriver=systemd"],
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "100m"
    }
}
EOF
```

3. Create a new config.toml with modified settings:
```bash
sudo tee /etc/nvidia-container-runtime/config.toml <<EOF
disable-require = false
supported-driver-capabilities = "compute,utility,graphics,display,video,ngx"

[nvidia-container-cli]
load-kmods = true
no-cgroups = false
debug = "/var/log/nvidia-container-toolkit.log"

[nvidia-container-runtime]
debug = "/var/log/nvidia-container-runtime.log"
log-level = "debug"

[nvidia-container-runtime.modes.csv]
mount-spec-path = "/etc/nvidia-container-runtime/host-files-for-container.d"
EOF
```

4. Restart the services:
```bash
sudo systemctl restart docker
```

5. Try a test with a minimal container:
```bash
docker run --rm --gpus all ubuntu nvidia-smi
```

If this still doesn't work, could you please check:
1. The contents of `/var/log/nvidia-container-toolkit.log` (if it exists)
2. The output of `dmesg | grep nvidia`

This will help us identify if there are any kernel-level issues interfering with the container runtime.
