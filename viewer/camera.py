import numpy as np
from math import radians, degrees, asin, acos, atan2, sqrt, cos, sin, pi

# ---------------------------------------------------------------------------
# Quaternion helpers (w, x, y, z convention)
# ---------------------------------------------------------------------------

def _quat_identity():
    return np.array([1.0, 0.0, 0.0, 0.0])

def _quat_mul(a, b):
    """Hamilton product of two quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ])

def _quat_from_axis_angle(axis, angle_rad):
    """Build a quaternion from a unit axis and angle in radians."""
    s = sin(angle_rad / 2.0)
    c = cos(angle_rad / 2.0)
    return np.array([c, axis[0]*s, axis[1]*s, axis[2]*s])

def _quat_to_matrix(q):
    """Return a 3x3 rotation matrix from a unit quaternion."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])

def _quat_from_matrix(m):
    """Extract a unit quaternion from a 3x3 rotation matrix (Shepperd)."""
    trace = m[0,0] + m[1,1] + m[2,2]
    if trace > 0:
        s = 0.5 / sqrt(trace + 1.0)
        return np.array([0.25/s,
                         (m[2,1]-m[1,2])*s,
                         (m[0,2]-m[2,0])*s,
                         (m[1,0]-m[0,1])*s])
    elif m[0,0] > m[1,1] and m[0,0] > m[2,2]:
        s = 2.0 * sqrt(1.0 + m[0,0] - m[1,1] - m[2,2])
        return np.array([(m[2,1]-m[1,2])/s, 0.25*s,
                         (m[0,1]+m[1,0])/s, (m[0,2]+m[2,0])/s])
    elif m[1,1] > m[2,2]:
        s = 2.0 * sqrt(1.0 + m[1,1] - m[0,0] - m[2,2])
        return np.array([(m[0,2]-m[2,0])/s, (m[0,1]+m[1,0])/s,
                         0.25*s, (m[1,2]+m[2,1])/s])
    else:
        s = 2.0 * sqrt(1.0 + m[2,2] - m[0,0] - m[1,1])
        return np.array([(m[1,0]-m[0,1])/s, (m[0,2]+m[2,0])/s,
                         (m[1,2]+m[2,1])/s, 0.25*s])

def _screen_to_sphere(px, py, vw, vh):
    """
    Map a pixel coordinate to a point on the unit arcball sphere.
    Points outside the sphere are projected onto the rim (z=0 ring).
    Returns a normalised 3-vector.
    """
    # Normalise to [-1, 1] with y flipped
    x = (2.0 * px - vw) / min(vw, vh)
    y = (vh - 2.0 * py) / min(vw, vh)
    r2 = x*x + y*y
    if r2 <= 1.0:
        z = sqrt(1.0 - r2)
    else:
        # Outside sphere — project onto equator ring
        r = sqrt(r2)
        x, y, z = x/r, y/r, 0.0
    v = np.array([x, y, z])
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

class Camera:
    def __init__(self):
        self.target      = np.array([0.0, 0.0, 0.0])
        self._scene_extent = 1.0   # updated by fit_scene()
        self.distance    = 3.0
        self.ortho_scale = 1.5
        self.ortho       = True
        # Rotation stored as a quaternion (w, x, y, z)
        # Default: isometric-ish view with Z-up convention.
        # Azimuth around world Z, then tilt down toward the XY ground plane.
        q_az  = _quat_from_axis_angle(np.array([0,0,1], dtype=float), radians(45))
        q_el  = _quat_from_axis_angle(np.array([1,0,0], dtype=float), radians(60))
        self.rotation = _quat_mul(q_el, q_az)
        self.rotation /= np.linalg.norm(self.rotation)
        # Orbit drag state
        self._orbit_start_q  = None   # rotation snapshot at drag start
        self._orbit_start_v  = None   # sphere vector at drag start (arcball)
        self._orbit_last_pos = None   # last pixel position (trackball)
        self._vw = 1                  # viewport width  (updated each drag)
        self._vh = 1                  # viewport height
        # Pan drag state
        self._pan_last = None

    # ------------------------------------------------------------------
    # Derived geometry
    # ------------------------------------------------------------------

    def get_eye(self):
        """Camera position — target + distance along the rotation's +Z axis."""
        R = _quat_to_matrix(self.rotation)
        # Local +Z (forward-out) in world space
        forward = R[:, 2]
        return self.target + forward * self.distance

    def get_up(self):
        """Camera up — rotation's +Y axis in world space."""
        R = _quat_to_matrix(self.rotation)
        return R[:, 1]

    # ------------------------------------------------------------------
    # Orbit
    # ------------------------------------------------------------------

    def begin_orbit(self, pos, vw, vh, pivot=None):
        """
        Called on right-mouse-button press.
        vw, vh: current viewport pixel dimensions.
        pivot:  optional world-space 3-vector; if given, re-centre target there.
        """
        if pivot is not None:
            pivot = np.array(pivot, dtype=float)
            # Only re-centre the orbit target if the clicked point is far enough
            # away from the current target — prevents constant jumping when
            # inspecting small geometry with repeated right-click drags.
            _REPIVOT_THRESHOLD = 55.0  # mm
            if np.linalg.norm(pivot - self.target) > _REPIVOT_THRESHOLD:
                old_eye = self.get_eye()
                self.target   = pivot
                self.distance = float(np.linalg.norm(old_eye - pivot))
                self.distance = max(0.01, self.distance)
        self._vw = vw
        self._vh = vh
        self._orbit_start_q  = self.rotation.copy()
        self._orbit_start_v  = _screen_to_sphere(pos.x(), pos.y(), vw, vh)
        self._orbit_last_pos = (pos.x(), pos.y())

    def orbit(self, pos):
        """Called on right-drag mouse-move."""
        from cad.prefs import prefs
        if prefs.camera_mode == 'arcball':
            self._orbit_arcball(pos)
        else:
            self._orbit_trackball(pos)

    def _orbit_arcball(self, pos):
        """Sphere-projection arcball — accurate but has pole singularities."""
        if self._orbit_start_v is None:
            return
        v2 = _screen_to_sphere(pos.x(), pos.y(), self._vw, self._vh)
        v1 = self._orbit_start_v

        axis_cam = np.cross(v1, v2)
        axis_len = np.linalg.norm(axis_cam)
        if axis_len < 1e-10:
            return
        axis_cam /= axis_len

        # Transform screen-space axis into world space
        R = _quat_to_matrix(self._orbit_start_q)
        axis_world = R @ axis_cam

        dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
        angle = acos(dot)

        from cad.prefs import prefs
        angle *= prefs.camera_rotate_speed

        dq = _quat_from_axis_angle(axis_world, angle)
        self.rotation = _quat_mul(dq, self._orbit_start_q)
        self.rotation /= np.linalg.norm(self.rotation)

    def _orbit_trackball(self, pos):
        """Incremental trackball — no poles, rotates indefinitely in any direction."""
        if self._orbit_last_pos is None:
            return
        from cad.prefs import prefs
        dx = pos.x() - self._orbit_last_pos[0]
        dy = pos.y() - self._orbit_last_pos[1]
        self._orbit_last_pos = (pos.x(), pos.y())

        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return

        if prefs.camera_invert_yaw:
            dx = -dx
        if prefs.camera_invert_pitch:
            dy = -dy

        # Scale pixels to radians — pi radians across the short viewport dimension
        scale = prefs.camera_rotate_speed * pi / min(self._vw, self._vh)
        angle_yaw   = -dx * scale   # horizontal drag → yaw around world Z-up
        angle_pitch =  dy * scale   # vertical drag   → pitch around camera right

        R = _quat_to_matrix(self.rotation)
        right = R[:, 0]   # camera local X in world space
        up    = R[:, 1]   # camera local Y in world space

        # Yaw around the camera's own up axis — decoupled from world Z
        q_yaw   = _quat_from_axis_angle(up,    angle_yaw)
        # Pitch around the camera's own right axis
        q_pitch = _quat_from_axis_angle(right, angle_pitch)

        # Apply both from the same snapshot so order doesn't matter
        self.rotation = _quat_mul(q_pitch, _quat_mul(q_yaw, self.rotation))
        self.rotation /= np.linalg.norm(self.rotation)

    def end_orbit(self):
        self._orbit_start_q  = None
        self._orbit_start_v  = None
        self._orbit_last_pos = None

    # ------------------------------------------------------------------
    # Pan
    # ------------------------------------------------------------------

    def begin_pan(self, pos):
        self._pan_last = (pos.x(), pos.y())

    def pan(self, pos):
        if self._pan_last is None:
            return
        dx = pos.x() - self._pan_last[0]
        dy = pos.y() - self._pan_last[1]
        R = _quat_to_matrix(self.rotation)
        right = R[:, 0]   # camera local X
        up    = R[:, 1]   # camera local Y
        scale = self.ortho_scale if self.ortho else self.distance
        from cad.prefs import prefs
        self.target -= right * dx * 0.002 * scale * prefs.camera_pan_speed
        self.target += up    * dy * 0.002 * scale * prefs.camera_pan_speed
        self._pan_last = (pos.x(), pos.y())

    def end_pan(self):
        self._pan_last = None

    # ------------------------------------------------------------------
    # Scroll
    # ------------------------------------------------------------------

    def scroll(self, delta):
        """Zoom without cursor — kept for compatibility."""
        factor = 0.9 if delta > 0 else 1.1
        if self.ortho:
            self.ortho_scale *= factor
            self.ortho_scale = max(0.001 * self._scene_extent,
                                   min(10.0 * self._scene_extent, self.ortho_scale))
        else:
            self.distance *= factor
            self.distance = max(0.001 * self._scene_extent,
                                min(10.0 * self._scene_extent, self.distance))

    def scroll_at(self, delta: int, px: float, py: float, vw: int, vh: int):
        """
        Zoom toward/away from the world point under the cursor at (px, py).

        Converts the cursor to a normalised offset from the viewport centre,
        then shifts `target` so that world point stays fixed under the cursor
        as the scale changes.

        `delta` is Qt's angleDelta().y() in eighths-of-a-degree; one physical
        wheel notch is 120. Qt coalesces several notches into a single event
        with a proportionally larger delta, so the zoom step MUST scale with
        magnitude — a fixed per-event factor makes a coalesced 3-notch event
        zoom the same as one notch, which reads as "scroll notches ignored".
        """
        if delta == 0:
            return
        notches = abs(delta) / 120.0
        base = 0.9 if delta > 0 else 1.1
        factor = base ** notches

        R = _quat_to_matrix(self.rotation)
        right = R[:, 0]
        up    = R[:, 1]

        # Cursor offset from centre in NDC [-1, 1] with y flipped
        cx = (2.0 * px - vw) / vw
        cy = (vh - 2.0 * py) / vh

        if self.ortho:
            # World point under cursor before zoom
            # ortho_scale is half-width of the view in world units (matches renderer)
            aspect = vw / vh if vh else 1.0
            wx = cx * self.ortho_scale * aspect
            wy = cy * self.ortho_scale
            world_under = self.target + right * wx + up * wy

            self.ortho_scale *= factor
            self.ortho_scale = max(0.001 * self._scene_extent,
                                   min(10.0 * self._scene_extent, self.ortho_scale))

            # Where that world point would now project — shift target to compensate
            wx_new = cx * self.ortho_scale * aspect
            wy_new = cy * self.ortho_scale
            world_under_new = self.target + right * wx_new + up * wy_new
            self.target += world_under - world_under_new
        else:
            # Perspective: point under cursor is on the focal plane at `distance`
            # Field of view ~45° → tan(22.5°) ≈ 0.4142
            fov_half_tan = 0.4142
            aspect = vw / vh if vh else 1.0
            wx = cx * self.distance * fov_half_tan * aspect
            wy = cy * self.distance * fov_half_tan
            world_under = self.target + right * wx + up * wy

            self.distance *= factor
            self.distance = max(0.001 * self._scene_extent,
                                min(10.0 * self._scene_extent, self.distance))

            wx_new = cx * self.distance * fov_half_tan * aspect
            wy_new = cy * self.distance * fov_half_tan
            world_under_new = self.target + right * wx_new + up * wy_new
            self.target += world_under - world_under_new

    # ------------------------------------------------------------------
    # Snap to face normal
    # ------------------------------------------------------------------

    def snap_to_normal(self, nx, ny, nz, origin=(0.0, 0.0, 0.0),
                       x_dir=None, y_dir=None):
        """
        Orient camera to look straight down the face normal,
        centred on the face origin.

        If x_dir/y_dir are provided (the plane's own axes), use them as the
        camera right/up so that the view matches the sketch coordinate system.
        """
        length = sqrt(nx*nx + ny*ny + nz*nz)
        if length < 1e-9:
            return
        nx, ny, nz = nx/length, ny/length, nz/length

        self.target = np.array(origin, dtype=float)

        forward = np.array([nx, ny, nz])

        if x_dir is not None and y_dir is not None:
            right = np.array(x_dir, dtype=float)
            right /= np.linalg.norm(right)
            up = np.array(y_dir, dtype=float)
            up /= np.linalg.norm(up)
        else:
            # Fallback: derive axes from world up (world-plane sketches)
            world_up = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(forward, world_up)) > 0.99:
                world_up = np.array([0.0, 1.0, 0.0])
            right = np.cross(world_up, forward)
            right /= np.linalg.norm(right)
            up = np.cross(forward, right)
            up /= np.linalg.norm(up)

        # Assemble rotation matrix: columns are right, up, forward
        m = np.array([
            [right[0],   up[0],   forward[0]],
            [right[1],   up[1],   forward[1]],
            [right[2],   up[2],   forward[2]],
        ])
        self.rotation = _quat_from_matrix(m)
        self.rotation /= np.linalg.norm(self.rotation)

    # ------------------------------------------------------------------
    # Projection toggle
    # ------------------------------------------------------------------


    def fit_scene(self, bbox_min, bbox_max):
        """
        Position camera to frame the scene bounding box.
        Sets target to bbox centre, distance to fit everything in view.
        Called once after loading geometry.
        """
        import numpy as np
        mn = np.array(bbox_min, dtype=float)
        mx = np.array(bbox_max, dtype=float)
        centre = (mn + mx) * 0.5
        extent = float(np.linalg.norm(mx - mn))   # diagonal length

        self.target        = centre
        self.distance      = extent * 1.2
        self.ortho_scale   = extent * 0.6
        self._scene_extent = extent   # used by scroll() for adaptive limits

    def toggle_ortho(self):
        self.ortho = not self.ortho
