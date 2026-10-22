"""Domain errors for the product pipeline with terminal-safe messages."""


class ProductPipelineError(Exception):
    exit_code = 1


class InvalidProductLinkError(ProductPipelineError):
    exit_code = 2


class SourceBlockedError(ProductPipelineError):
    exit_code = 3


class ProductNotFoundError(ProductPipelineError):
    exit_code = 4


class MediaProcessingError(ProductPipelineError):
    exit_code = 6


class ProviderError(ProductPipelineError):
    exit_code = 7
