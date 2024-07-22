import torch

if __name__ == '__main__':
    cuda_is_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count()

    print(f"Cuda available: {cuda_is_available}")
    if cuda_device_count < 1:
        print("No GPU found!")
    else:
        for i in range(cuda_device_count):
            print(torch.cuda.get_device_properties(i))
