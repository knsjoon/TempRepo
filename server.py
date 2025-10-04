# server.py
from flask import Flask, jsonify, render_template, request
from urdfpy import URDF
import numpy as np
import os, threading, time, math

# ---------------- Flask setup ----------------
app = Flask(__name__, template_folder="templates", static_folder="static")

# ----- Adjust these to your paths -----
URDF_PATH    = "/home/seokjoonkim/Desktop/programming/ur10e_bundle/ur10e.urdf"
OBJ_BASE_URL = "/static/meshes/ur10e/visual/"
OBJ_FS_DIR   = "/home/seokjoonkim/Desktop/programming/babylonjs/static/meshes/ur10e/visual"

# ----------------------------------------------
class URDFVisualizer:
    def __init__(self, urdf_path: str):
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")
        self.robot = URDF.load(urdf_path)

        # Thread-safe joint state held by the instance
        self._lock = threading.Lock()
        self.current_cfg = {j.name: 0.0 for j in self.robot.joints}

        # Sim thread controls
        self._sim_thread: threading.Thread | None = None
        self._sim_running = False
        self._spin_name: str | None = None
        self._spin_rate_rad_s: float = math.radians(45.0)  # default 45°/s

    # ---------- Core FK helpers ----------
    def fk_by_name(self, cfg: dict[str, float]):
        link_fk = self.robot.link_fk(cfg=cfg)  # {LinkObj -> 4x4}
        return {link.name: T for link, T in link_fk.items()}

    def get_joint_positions(self, joint_cfg=None):
        if joint_cfg is None:
            joint_cfg = {j.name: 0.0 for j in self.robot.joints}
        fk = self.fk_by_name(joint_cfg)
        joints, edges = [], []
        for j in self.robot.joints:
            parent_name = j.parent.name if hasattr(j.parent, "name") else str(j.parent)
            child_name  = j.child.name  if hasattr(j.child,  "name")  else str(j.child)
            if child_name not in fk:
                continue
            child_pos = fk[child_name][:3, 3].tolist()
            joints.append({"name": j.name, "parent": parent_name, "child": child_name, "position": child_pos})
            if parent_name in fk:
                parent_pos = fk[parent_name][:3, 3].tolist()
                edges.append({"from": parent_name, "to": child_name, "p1": parent_pos, "p2": child_pos})
            elif parent_name == "world":
                edges.append({"from": "world", "to": child_name, "p1": [0,0,0], "p2": child_pos})
        return {"joints": joints, "edges": edges}

    @staticmethod
    def _mat3_to_quat_xyzw(R: np.ndarray):
        t = np.trace(R)
        if t > 0.0:
            s = np.sqrt(t + 1.0) * 2.0
            w = 0.25*s
            x = (R[2,1]-R[1,2])/s; y = (R[0,2]-R[2,0])/s; z = (R[1,0]-R[0,1])/s
        else:
            if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
                s = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
                w = (R[2,1]-R[1,2])/s; x = 0.25*s; y = (R[0,1]+R[1,0])/s; z = (R[0,2]+R[2,0])/s
            elif R[1,1] > R[2,2]:
                s = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
                w = (R[0,2]-R[2,0])/s; x = (R[0,1]+R[1,0])/s; y = 0.25*s; z = (R[1,2]+R[2,1])/s
            else:
                s = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
                w = (R[1,0]-R[0,1])/s; x = (R[0,2]+R[2,0])/s; y = (R[1,2]+R[2,1])/s; z = 0.25*s
        return [float(x), float(y), float(z), float(w)]

    @staticmethod
    def _dae_to_obj_name(src: str) -> str:
        base = os.path.basename(src)     # e.g., 'upperarm.dae'
        stem, _ = os.path.splitext(base) # 'upperarm'
        return f"{stem}.obj"

    def visuals_world_poses(self, joint_cfg=None):
        if joint_cfg is None:
            joint_cfg = {j.name: 0.0 for j in self.robot.joints}
        fk = self.fk_by_name(joint_cfg)
        visuals = []
        links_by_name = {L.name: L for L in self.robot.links}
        for lname, link_T in fk.items():
            link = links_by_name.get(lname)
            if not link or not getattr(link, "visuals", None):
                continue
            for vis in link.visuals:
                if not getattr(vis, "geometry", None) or not getattr(vis.geometry, "mesh", None) or not vis.geometry.mesh.filename:
                    continue
                V = vis.origin if vis.origin is not None else np.eye(4)
                world_T = link_T @ V
                pos = world_T[:3, 3]; R = world_T[:3, :3]
                quat = self._mat3_to_quat_xyzw(R)
                obj_name = self._dae_to_obj_name(vis.geometry.mesh.filename)
                if OBJ_FS_DIR and not os.path.exists(os.path.join(OBJ_FS_DIR, obj_name)):
                    print(f"⚠️ Missing OBJ on disk: {obj_name}")
                visuals.append({
                    "link": lname,
                    "mesh": obj_name,
                    "position": [float(pos[0]), float(pos[1]), float(pos[2])],
                    "quaternion": quat,
                    "scale": 1.0  # set 0.001 if your OBJs are in mm
                })
        return visuals

    # ---------- Joint config access ----------
    def get_current_cfg(self) -> dict[str, float]:
        with self._lock:
            return dict(self.current_cfg)

    def set_current_cfg(self, new_cfg: dict[str, float]):
        with self._lock:
            for name, val in new_cfg.items():
                if name in self.current_cfg:
                    try:
                        self.current_cfg[name] = float(val)
                    except (TypeError, ValueError):
                        pass

    # ---------- Simulation inside the class ----------
    def _pick_spinnable_joint(self) -> str | None:
        # Prefer typical names if present
        prefer = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint")
        names = [j.name for j in self.robot.joints]
        for n in prefer:
            if n in names:
                return n
        # Otherwise: first revolute/continuous
        for j in self.robot.joints:
            jtype = getattr(j, "joint_type", getattr(j, "type", None))
            if jtype in ("revolute", "continuous"):
                return j.name
        return None

    def start_simulation(self, joint_name: str | None = None, rate_deg_per_s: float = 45.0, hz: float = 20.0):
        """Spin one joint continuously at fixed angular velocity (inside this class)."""
        if self._sim_running:
            return  # already running

        self._spin_name = joint_name or self._pick_spinnable_joint()
        if not self._spin_name:
            print("[sim] No revolute/continuous joint found; simulation not started.")
            return

        self._spin_rate_rad_s = math.radians(rate_deg_per_s)
        self._sim_running = True

        def _loop():
            print(f"[sim] Spinning joint: {self._spin_name} at {rate_deg_per_s} deg/s")
            angle = 0.0
            t_prev = time.monotonic()
            period = 1.0 / max(hz, 1.0)
            while self._sim_running:
                t_now = time.monotonic()
                dt = t_now - t_prev
                t_prev = t_now

                angle = (angle + self._spin_rate_rad_s * dt) % (2.0 * math.pi)

                with self._lock:
                    # Zero all, set the spinner
                    for j in self.robot.joints:
                        self.current_cfg[j.name] = 0.0
                    self.current_cfg[self._spin_name] = angle

                time.sleep(period)

        self._sim_thread = threading.Thread(target=_loop, daemon=True)
        self._sim_thread.start()

    def stop_simulation(self):
        self._sim_running = False
        if self._sim_thread and self._sim_thread.is_alive():
            self._sim_thread.join(timeout=0.1)
        self._sim_thread = None

# ---------------- Instantiate & start sim ----------------
visualizer = URDFVisualizer(URDF_PATH)
visualizer.start_simulation(joint_name="shoulder_lift_joint", rate_deg_per_s=45.0, hz=20.0)  # 360° every 8s

# ---------------- Routes ----------------
@app.route("/")
def index():
    return render_template("index.html")

# Param-driven FK (kept as-is)
@app.route("/api/joints", methods=["GET"])
def api_joints():
    cfg = {j.name: 0.0 for j in visualizer.robot.joints}
    for j in visualizer.robot.joints:
        if j.name in request.args:
            try:
                cfg[j.name] = float(request.args[j.name])
            except ValueError:
                pass
    return jsonify(visualizer.get_joint_positions(joint_cfg=cfg))

@app.route("/api/visuals", methods=["GET"])
def api_visuals():
    cfg = {j.name: 0.0 for j in visualizer.robot.joints}
    for j in visualizer.robot.joints:
        if j.name in request.args:
            try:
                cfg[j.name] = float(request.args[j.name])
            except ValueError:
                pass
    visuals = visualizer.visuals_world_poses(joint_cfg=cfg)
    return jsonify({"baseUrl": OBJ_BASE_URL, "visuals": visuals})

# Live endpoints (read simulated cfg from the class)
@app.route("/api/joints_live", methods=["GET"])
def api_joints_live():
    cfg = visualizer.get_current_cfg()
    return jsonify(visualizer.get_joint_positions(joint_cfg=cfg))

@app.route("/api/visuals_live", methods=["GET"])
def api_visuals_live():
    cfg = visualizer.get_current_cfg()
    visuals = visualizer.visuals_world_poses(joint_cfg=cfg)
    return jsonify({"baseUrl": OBJ_BASE_URL, "visuals": visuals})

if __name__ == "__main__":
    app.run(debug=True)