from .explainer import ExplainerGenerator
from .product import ProductGenerator

VIDEO_TYPES: dict[str, type] = {
    "explainer": ExplainerGenerator,
    "product": ProductGenerator,
}
