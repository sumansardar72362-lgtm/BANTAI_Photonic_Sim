import numpy as np
import pickle  # মেমরি সেভ এবং লোড করার জন্য নতুন লাইব্রেরি
import os

class KeciaLinearLayer:
    def __init__(self, input_features, output_features):
        self.input_features = input_features
        self.output_features = output_features
        
        # রেন্ডম মেমরি দিয়ে চিপ চালু হলো
        self.weights = np.random.rand(input_features, output_features)
        print(f"✅ [Kecia NN] Layer Initialized: {input_features} inputs -> {output_features} outputs")

    def forward(self, input_data):
        """এই ফাংশনটি ডেটা রিসিভ করে সরাসরি ফোটোনিক চিপে পাঠায়"""
        # ডামি অপটিক্যাল মাল্টিপ্লিকেশন সিমুলেশন
        optical_result = np.dot(input_data, self.weights)
        print("✅ [SDK] Forward Pass completed by Photonic Core!")
        return optical_result

    # ==========================================
    # 💾 নতুন ফিচার: মেমরি ম্যানেজমেন্ট
    # ==========================================
    def save_model(self, file_name="model.kecia"):
        """এআই-এর শেখা জ্ঞান বা Weights একটি .kecia ফাইলে সেভ করে রাখবে"""
        with open(file_name, 'wb') as f:
            pickle.dump(self.weights, f)
        print(f"\n💾 [Kecia Memory] AI Brain permanently saved to '{file_name}'!")

    def load_model(self, file_name="model.kecia"):
        """সেভ করা .kecia ফাইল থেকে মেমরি আবার চিপে লোড করবে"""
        if os.path.exists(file_name):
            with open(file_name, 'rb') as f:
                self.weights = pickle.load(f)
            print(f"\n📂 [Kecia Memory] AI Brain successfully loaded from '{file_name}'!")
        else:
            print(f"\n❌ [Error] '{file_name}' পাওয়া যায়নি! আগে চিপ ট্রেইন করে ডেটা সেভ করো।")