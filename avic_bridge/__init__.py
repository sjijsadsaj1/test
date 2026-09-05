"""AVIC-to-ApexNav process bridge.

The ApexNav process imports only the light-weight client/controller modules.
The SVC implementation is loaded lazily by :mod:`avic_bridge.service` inside
the separate ``avic`` Conda environment.
"""

__all__ = ["AvicController", "AvicControllerConfig", "AvicDecision"]


def __getattr__(name):
    """Load the OpenCV-dependent controller only when it is requested.

    Keeping package import lightweight lets protocol and Qwen preflight tools
    run in the isolated AVIC environment without importing ApexNav's vision
    stack as a side effect.
    """

    if name in __all__:
        from .controller import AvicController, AvicControllerConfig, AvicDecision

        return {
            "AvicController": AvicController,
            "AvicControllerConfig": AvicControllerConfig,
            "AvicDecision": AvicDecision,
        }[name]
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
