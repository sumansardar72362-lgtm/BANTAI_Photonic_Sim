class Sequential:
    """
    BANTAI Kecia - Sequential Container.
    একাধিক লেয়ার এবং অ্যাক্টিভেশন ফাংশনকে একটি চেইনে যুক্ত করে।
    """
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x):
        # ইনপুট ডেটা পর্যায়ক্রমে প্রতিটি লেয়ারের ভেতর দিয়ে যাবে
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, grad_output):
        # চেইন রুল অনুযায়ী পেছনের দিক থেকে প্রতিটি লেয়ারের ভুল হিসাব হবে
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
        
    def get_parameters(self):
        # অপটিমাইজারের জন্য সব লেয়ারের ওয়েইট এবং বায়াস একসাথে কালেক্ট করা
        params = []
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                params.append(layer.weights)
            if hasattr(layer, 'bias') and layer.bias is not None:
                params.append(layer.bias)
        return params

    def __call__(self, x):
        return self.forward(x)