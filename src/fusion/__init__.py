"""
Fusion module registry.

Use :func:`get_fusion_module` to look up a fusion architecture by name.
The supported names map to:

============================  ====================================================
Name                          Class
============================  ====================================================
``"LMF"``                     :class:`baselines.LMF`
``"Early_Fusion"``            :class:`baselines.EarlyFusion`
``"Late_Fusion"``             :class:`baselines.LateFusion`
``"TFN"``                     :class:`baselines.TensorFusionNetwork`
``"MFB"``                     :class:`baselines.MultimodalFactorizedBilinear`
``"Cross_Modal_Attention"``   :class:`baselines.CrossModalAttention`
``"GMU"``                     :class:`baselines.GatedMultimodalUnit`
``"Multimodal_Transformer"``  :class:`baselines.MultimodalTransformer`
``"Tucker_Fusion"``           :class:`baselines.TuckerFusion`
``"BBFN"``                    :class:`bbfn.BiBimodalFusionNetwork`
``"Quaternion_Fusion"``       :class:`quaternion.QuaternionFusion`
``"ACGF"``                    :class:`proposed.AntiCorrelationGatedFusion`  (proposed)
``"TAC"``                     :class:`proposed.TinyAntiCorrelator`          (proposed)
============================  ====================================================
"""

from __future__ import annotations

from .baselines import (
    LMF,
    EarlyFusion,
    LateFusion,
    TensorFusionNetwork,
    MultimodalFactorizedBilinear,
    CrossModalAttention,
    GatedMultimodalUnit,
    MultimodalTransformer,
    TuckerFusion,
)
from .bbfn import BiBimodalFusionNetwork
from .quaternion import QuaternionFusion
from .proposed import AntiCorrelationGatedFusion, TinyAntiCorrelator


FUSION_REGISTRY = {
    # Baselines
    "LMF":                    LMF,
    "Early_Fusion":           EarlyFusion,
    "Late_Fusion":            LateFusion,
    "TFN":                    TensorFusionNetwork,
    "MFB":                    MultimodalFactorizedBilinear,
    "Cross_Modal_Attention":  CrossModalAttention,
    "GMU":                    GatedMultimodalUnit,
    "Multimodal_Transformer": MultimodalTransformer,
    "Tucker_Fusion":          TuckerFusion,
    "BBFN":                   BiBimodalFusionNetwork,
    "Quaternion_Fusion":      QuaternionFusion,
    # Proposed
    "ACGF":                   AntiCorrelationGatedFusion,
    "TAC":                    TinyAntiCorrelator,
}


def get_fusion_module(name: str, **kwargs):
    """Instantiate a fusion module by name.

    Parameters
    ----------
    name : str
        Key from ``FUSION_REGISTRY`` (case-sensitive).
    **kwargs
        Forwarded to the fusion module's constructor (e.g. ``hidden_dim=256``).

    Raises
    ------
    KeyError
        If ``name`` is not a registered fusion module.
    """
    if name not in FUSION_REGISTRY:
        raise KeyError(
            f"Unknown fusion module '{name}'. "
            f"Available: {sorted(FUSION_REGISTRY.keys())}"
        )
    return FUSION_REGISTRY[name](**kwargs)


__all__ = [
    "get_fusion_module",
    "FUSION_REGISTRY",
    "LMF",
    "EarlyFusion",
    "LateFusion",
    "TensorFusionNetwork",
    "MultimodalFactorizedBilinear",
    "CrossModalAttention",
    "GatedMultimodalUnit",
    "MultimodalTransformer",
    "TuckerFusion",
    "BiBimodalFusionNetwork",
    "QuaternionFusion",
    "AntiCorrelationGatedFusion",
    "TinyAntiCorrelator",
]
