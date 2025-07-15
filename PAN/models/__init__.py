from .pan_model import PanModel

def create_model(opt):
    """Factory wrapper so test/train scripts can do `from models import create_model`."""
    model = PanModel()
    model.initialize(opt)
    return model