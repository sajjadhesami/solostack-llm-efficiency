# Handle imports - try relative first, fall back to absolute
try:
    from .pipeline import build_pipeline
except ImportError:
    from inferencePipeline.pipeline import build_pipeline


def loadPipeline():
    """Entry point expected by the evaluator.

    Returns a callable that accepts a list of question dicts and returns a list
    of answer dicts in the expected format.
    """

    return build_pipeline()
