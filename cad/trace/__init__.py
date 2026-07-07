"""
cad/trace/

Ground-up raster-to-vector tracing (Image Trace V2).  Replaces the OpenCV
dependency with Pillow (decode) + scipy.ndimage (blur/label) + internal
border-following and primitive fitting — all Qt-free and headless-testable.

Modules
-------
mask      — decode + grayscale + blur + threshold  → boolean foreground mask
contours  — findContours replacement: labeled border-following (Milestone A)
fit       — curvature-segmented line/arc fitting    (Milestone B)

See IMAGE_TRACE_V2_PLAN.md for the full plan and milestone split.
"""
