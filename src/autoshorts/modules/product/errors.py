"""Domain errors for the product pipeline with terminal-safe messages."""


class ProductPipelineError(Exception):
    exit_code = 1


class InvalidProductLink(ProductPipelineError):
    exit_code = 2


class SourceBlocked(ProductPipelineError):
    exit_code = 3


class ProductNotFound(ProductPipelineError):
    exit_code = 4


class MediaProcessingError(ProductPipelineError):
    exit_code = 6


class ProviderError(ProductPipelineError):
    exit_code = 7
