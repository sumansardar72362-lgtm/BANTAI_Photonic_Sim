import os
import pickle

def save_model(model, filepath):
    """
    BANTAI Kecia - মডেলের মেমরি (Weights & Biases) একটি .kecia ফাইলে সেভ করে।
    """
    # ফাইলের এক্সটেনশন চেক করা (না থাকলে .kecia বসিয়ে দেওয়া)
    if not filepath.endswith('.kecia'):
        filepath += '.kecia'
        
    model_state = []
    
    # প্রতিটি লেয়ার থেকে শুধু ডেটা (Numpy Array) কালেক্ট করা
    for layer in model.layers:
        layer_state = {}
        if hasattr(layer, 'weights'):
            layer_state['weights'] = layer.weights.data
        if hasattr(layer, 'bias') and layer.bias is not None:
            layer_state['bias'] = layer.bias.data
        model_state.append(layer_state)
        
    # ফোল্ডার না থাকলে তৈরি করে নেওয়া
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
        
    # ফাইল রাইট করা (Binary Mode)
    with open(filepath, 'wb') as f:
        pickle.dump(model_state, f)
        
    print(f"💾 Model memory successfully saved to: {filepath}")


def load_model(model, filepath):
    """
    BANTAI Kecia - .kecia ফাইল থেকে মেমরি (Weights & Biases) মডেলের ভেতর লোড করে।
    """
    if not filepath.endswith('.kecia'):
        filepath += '.kecia'
        
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ No model found at {filepath}")
        
    # ফাইল রিড করা (Binary Mode)
    with open(filepath, 'rb') as f:
        model_state = pickle.load(f)
        
    # মডেলের প্রতিটি লেয়ারে ডেটাগুলো আবার ইনজেক্ট (Inject) করা
    for i, layer in enumerate(model.layers):
        if i < len(model_state):
            state = model_state[i]
            if hasattr(layer, 'weights') and 'weights' in state:
                layer.weights.data = state['weights']
            if hasattr(layer, 'bias') and layer.bias is not None and 'bias' in state:
                layer.bias.data = state['bias']
                
    print(f"🧠 Model memory successfully loaded from: {filepath}")