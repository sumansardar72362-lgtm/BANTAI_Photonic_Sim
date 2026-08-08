# একদম PyTorch-এর মতো প্রফেশনাল ইম্পোর্ট!
import src.kecia as kc
from src.kecia import nn, optim

class PhotonicModel:
    def __init__(self):
        # এখন আর বড় নাম লিখতে হবে না, শুধু nn.Linear এবং nn.ReLU
        self.fc1 = nn.Linear(2, 4, device='photonic')
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 1, device='photonic')

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def test_api():
    print("\n==================================================")
    print(" 🌟 TESTING THE NEW CLEAN API")
    print("==================================================\n")
    
    # Tensor5D এখন শুধু kc.Tensor
    X = kc.Tensor([[0.0, 1.0], [1.0, 0.0]], device='photonic')
    
    model = PhotonicModel()
    
    print("\n🚀 Running Forward Pass with Clean API...")
    output = model.forward(X)
    print("\n🎯 Output:")
    print(output.data)
    
if __name__ == "__main__":
    test_api()