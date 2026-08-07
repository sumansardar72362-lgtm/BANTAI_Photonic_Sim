# Kecia Photonic Engine Initialized
# src/kecia/__init__.py
__version__ = "1.0.0"
print(f"🔥 BANTAI Kecia SDK v{__version__} Initialized (Photonic Backend Ready)")

# কোর মডিউলগুলোকে সহজে ব্যবহারযোগ্য করা হচ্ছে
from .core.tensor5d import Tensor5D as Tensor
from . import nn
from . import optim