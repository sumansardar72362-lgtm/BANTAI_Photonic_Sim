from core.systolic_array import PhotonicSystolicArray

class Bantai:
    def __init__(self, size=2):
        """BANTAI ফ্রেমওয়ার্ক ইনিশিয়ালাইজ হচ্ছে"""
        print("\n🚀 Starting BANTAI AI Interface...")
        # এটি ব্যাকএন্ডে তোমার ফোটোনিক হার্ডওয়্যারকে কল করছে
        self.hardware_backend = PhotonicSystolicArray(size=size)
        print("✅ BANTAI SDK connected to Photonic Hardware.")

    def run_matmul(self, matrix_a, matrix_b):
        """ডেভেলপারদের জন্য সাধারণ ম্যাট্রিক্স গুণ ফাংশন"""
        print("--> Sending data to optical cores for zero-latency processing...")
        # পেছনের ফিজিক্স লজিক রান হচ্ছে
        output = self.hardware_backend.matmul(matrix_a, matrix_b)
        return output

# ডেভেলপার টেস্টিং ব্লক
if __name__ == "__main__":
    # একজন এআই ডেভেলপার ঠিক এভাবেই তোমার চিপ ব্যবহার করবে:
    
    # ১. চিপের সাথে কানেকশন তৈরি
    ai_chip = Bantai(size=2)
    
    # ২. এআই মডেলের ডেটা
    data_X = [[1, 2], [3, 4]]
    data_Y = [[5, 6], [7, 8]]
    
    # ৩. জাস্ট এক লাইনে পুরো ডেটা প্রসেসিং!
    result = ai_chip.run_matmul(data_X, data_Y)
    
    print("🎯 Final AI Output:")
    for row in result:
        print(row)