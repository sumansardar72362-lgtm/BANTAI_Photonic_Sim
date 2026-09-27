import numpy as np
from kecia.core.tensor5d import Tensor5D

class DataLoader:
    """
    BANTAI Kecia - Photonic DataLoader.
    বিশাল ডেটাসেটকে ছোট ছোট ব্যাচে (Batch) ভাগ করে ফোটোনিক চিপে ফিড করার জন্য।
    এটি মেমরি বটলনেক দূর করে এবং চিপের প্যারালাল প্রসেসিং (Batch Processing) স্পিড বাড়ায়।
    """
    def __init__(self, X, Y, batch_size=32, shuffle=True):
        # ডেটাকে ফাস্ট প্রসেস করার জন্য Numpy অ্যারেতে রাখা হচ্ছে
        self.X = np.array(X.data if isinstance(X, Tensor5D) else X, dtype=np.float32)
        self.Y = np.array(Y.data if isinstance(Y, Tensor5D) else Y, dtype=np.float32)
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = self.X.shape[0]
        
        # মোট কতগুলো ব্যাচ হবে তার হিসাব
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))
        
        print(f"📦 Photonic DataLoader Ready | Total Samples: {self.num_samples} | Batches: {self.num_batches} (Size: {self.batch_size})")

    def __iter__(self):
        """প্রতিটা Epoch-এর শুরুতে ডেটা রেন্ডমলি মিক্স (Shuffle) করে দেয় যাতে মডেল মুখস্থ না করে"""
        self.indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)
            
        self.current_batch = 0
        return self

    def __next__(self):
        """লুপ চলার সময় প্রতিবার একটি করে নতুন ব্যাচ রিটার্ন করবে"""
        if self.current_batch >= self.num_batches:
            raise StopIteration
            
        # ব্যাচের শুরু এবং শেষের ইনডেক্স বের করা
        start_idx = self.current_batch * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_samples)
        
        batch_indices = self.indices[start_idx:end_idx]
        
        # ইনডেক্স অনুযায়ী ডেটা আলাদা করা
        batch_X = self.X[batch_indices]
        batch_Y = self.Y[batch_indices]
        
        self.current_batch += 1
        
        # ⚡ ম্যাজিক: ডেটাগুলোকে সরাসরি Tensor5D তে কনভার্ট করে রিটার্ন করা হচ্ছে!
        # এর ফলে ট্রেইনিং লুপে আলাদা করে কনভার্ট করার কোনো ঝামেলা থাকবে না।
        return Tensor5D(batch_X, device='photonic'), Tensor5D(batch_Y, device='photonic')

    def __len__(self):
        """মোট ব্যাচ সংখ্যা জানার জন্য"""
        return self.num_batches