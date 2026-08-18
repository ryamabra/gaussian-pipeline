import modal

app = modal.App("hello")

image = modal.Image.debian_slim().pip_install("torch")

@app.function(gpu="A10G", image=image)
def check_gpu():
    import torch
    return torch.cuda.get_device_name(0)

@app.local_entrypoint()
def main():
    print(check_gpu.remote())
