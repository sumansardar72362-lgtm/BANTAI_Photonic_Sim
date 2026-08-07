import numpy as np


class GlassMemoryMapper:
    """
    Translates standard computational data (like PyTorch/NumPy matrices) 
    into 5D Photonic Memory parameters: (X, Y, Z, Intensity, Phase)
    """
    
    def __init__(self, grid_x=256, grid_y=256, layers_z=10):
        # চিপের ফিজিক্যাল সাইজ
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.layers_z = layers_z
        
        # 5D সিমুলেটেড গ্রিড: X, Y, Z এবং শেষের ডাইমেনশনে Intensity(0) ও Phase(1)
        self.optical_grid = np.zeros((self.grid_x, self.grid_y, self.layers_z, 2))
        print(f"💎 Diamond-Glass Memory Initialized: {grid_x}x{grid_y} with {layers_z} Z-layers.")

    def encode_to_light(self, matrix, z_layer=0):
        """
        সাধারণ 2D ম্যাট্রিক্সকে আলোর Intensity এবং Phase-এ কনভার্ট করে।
        """
        matrix = np.array(matrix)
        rows, cols = matrix.shape
        
        if rows > self.grid_x or cols > self.grid_y:
            raise ValueError("Matrix is too large for the current chip grid size!")

        
        intensity = np.abs(matrix)
        
        
        phase = np.where(matrix < 0, np.pi, 0.0)
        
        # গ্লাসের গ্রিডে ডেটা বসানো হচ্ছে
        for i in range(rows):
            for j in range(cols):
                self.optical_grid[i, j, z_layer, 0] = intensity[i, j]
                self.optical_grid[i, j, z_layer, 1] = phase[i, j]
                
        print(f"⚡ Successfully mapped {rows}x{cols} matrix to Optical Layer Z={z_layer}")
        return intensity, phase

    def read_from_light(self, x_len, y_len, z_layer=0):
        """
        গ্লাসের আলো (Intensity ও Phase) থেকে আবার সাধারণ নম্বরে (Digital) কনভার্ট করে।
        """
        intensity = self.optical_grid[:x_len, :y_len, z_layer, 0]
        phase = self.optical_grid[:x_len, :y_len, z_layer, 1]
        
        # যদি Phase Pi (3.14) হয়, তার মানে ভ্যালু নেগেটিভ ছিল
        digital_matrix = intensity * np.cos(phase) 
        
        return np.round(digital_matrix, decimals=4)



    def save_state(self, filepath):
        """
        পুরো 5D অপটিক্যাল মেমোরি গ্রিডকে একটি .kecia ফাইলে সেভ করে।
        """
        import os
        import numpy as np
        
        # ফাইল সেভ করার ফোল্ডার না থাকলে তৈরি করে নেবে
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # এক্সটেনশন .kecia না থাকলে যোগ করে নেবে
        if not filepath.endswith('.kecia'):
            filepath += '.kecia'
            
        # 🛠️ FIX: ফাইল স্ট্রিম হিসেবে ওপেন করা হলো যাতে NumPy নিজে থেকে .npy বসাতে না পারে
        with open(filepath, 'wb') as f:
            np.save(f, self.optical_grid)
            
        print(f"💾 5D Photonic Memory saved securely to: {filepath}")

    def load_state(self, filepath):
        """
        একটি .kecia ফাইল থেকে 5D অপটিক্যাল মেমোরি রিস্টোর করে।
        """
        import os
        import numpy as np
        
        if not filepath.endswith('.kecia'):
            filepath += '.kecia'
            
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Memory file not found: {filepath}")
        
        # 🛠️ FIX: ফাইল স্ট্রিম হিসেবে রিড করা হলো
        with open(filepath, 'rb') as f:
            self.optical_grid = np.load(f)
            
        print(f"🔄 5D Photonic Memory successfully loaded from: {filepath}")