class PhotonicMAC:
    def __init__(self):
  
        self.refractive_index = 2.4
        self.efficiency = 0.99  
        print("⚡ BANTAI Photonic MAC Unit Initialized. Status: Online")

    def electro_optic_modulator(self, value):
        """ইলেকট্রনিক ডেটাকে আলোর তীব্রতায় (Optical Intensity) রূপান্তর করে"""
        optical_signal = value * self.efficiency
        return optical_signal

    def photo_detector(self, optical_signal):
        """আলোর তীব্রতাকে পুনরায় সাধারণ আউটপুটে (Electronic Data) রূপান্তর করে"""

        return round(optical_signal, 4)

    def multiply(self, input_a, input_b):
        """অপটিক্যাল ইন্টারফারেন্সের মাধ্যমে গুণ করা (Zero-latency theory)"""

        light_a = self.electro_optic_modulator(input_a)
        light_b = self.electro_optic_modulator(input_b)
        

        optical_result = light_a * light_b
        
   
        final_result = self.photo_detector(optical_result)
        
        return final_result


if __name__ == "__main__":
   
    mac = PhotonicMAC()
    

    val1 = 5
    val2 = 10
    
    print(f"Input Data: {val1} and {val2}")
    result = mac.multiply(val1, val2)
    print(f"Optical Multiplication Result: {result}")