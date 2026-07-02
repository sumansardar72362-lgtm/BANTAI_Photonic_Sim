from core.mac_unit import PhotonicMAC

class PhotonicSystolicArray:
    def __init__(self, size):
        self.size = size
        # চিপের ভেতরে MAC ইউনিটের গ্রিড তৈরি হচ্ছে
        self.mac_grid = [[PhotonicMAC() for _ in range(size)] for _ in range(size)]
        print(f"🔥 BANTAI: {size}x{size} Systolic Array Initialized with {size*size} Optical Cores!")

    def matmul(self, matrix_a, matrix_b):
        """দুটি ম্যাট্রিক্সের অপটিক্যাল গুণফল বের করার লজিক"""
        # জিরো দিয়ে একটি খালি রেজাল্ট ম্যাট্রিক্স তৈরি করা
        result = [[0 for _ in range(self.size)] for _ in range(self.size)]
        
        # সিস্টোলিক ডেটা ফ্লো সিমুলেশন (Multiply and Accumulate)
        for i in range(self.size):
            for j in range(self.size):
                for k in range(self.size):
                    # MAC ইউনিটে আলো দিয়ে গুণ হচ্ছে
                    multiplied_val = self.mac_grid[i][j].multiply(matrix_a[i][k], matrix_b[k][j])
                    # রেজাল্টের সাথে যোগ হচ্ছে
                    result[i][j] += round(multiplied_val, 4)
                    
        return result

# টেস্টিং ব্লক: অ্যারে ঠিকঠাক কাজ করছে কি না তা চেক করার জন্য
if __name__ == "__main__":
    # একটি ২x২ সাইজের সিস্টোলিক অ্যারে তৈরি করা হলো (৪টি কোর)
    array = PhotonicSystolicArray(size=2)
    
    # ইনপুট ম্যাট্রিক্স A এবং B (এআই ডেটা)
    matrix_A = [[1, 2], 
                [3, 4]]
                
    matrix_B = [[5, 6], 
                [7, 8]]
                
    print("\nInput Matrix A:", matrix_A)
    print("Input Matrix B:", matrix_B)
    
    # BANTAI চিপের মাধ্যমে গুণ
    final_output = array.matmul(matrix_A, matrix_B)
    
    print("\n⚡ Optical Matrix Output:")
    for row in final_output:
        print(row)