from flask import Flask, jsonify, render_template, request
from urdfpy import URDF
import numpy as np
import os

# ---------------- Flask setup ----------------
app = Flask(__name__, template_folder="templates", static_folder="static")

# ----- Adjust these to your paths -----
URDF_PATH   = "/home/seokjoonkim/Desktop/programming/ur10e_bundle/ur10e.urdf"
OBJ_BASE_URL = "/static/meshes/ur10e/visual/"  # where OBJs are served in Flask
OBJ_FS_DIR   = "/home/seokjoonkim/Desktop/programming/babylonjs/static/meshes/ur10e/visual"  # optional disk check

# ----------------------------------------------
class URDFVisualizer:
    def __init__(self, urdf_path):
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")
        self.robot = URDF.load(urdf_path)

    def fk_by_name(self, cfg):
        """Return {link_name: 4x4 world transform} using urdfpy FK."""
        link_fk = self.robot.link_fk(cfg=cfg)  # {LinkObj -> 4x4}
        return {link.name: T for link, T in link_fk.items()}

    def get_joint_positions(self, joint_cfg=None):
        # default: all zeros
        if joint_cfg is None:
            joint_cfg = {j.name: 0.0 for j in self.robot.joints}
        
        fk = self.fk_by_name(joint_cfg)
        joints, edges = [], []

        for j in self.robot.joints:
            parent_name = j.parent.name if hasattr(j.parent, "name") else str(j.parent)
            child_name  = j.child.name  if hasattr(j.child,  "name")  else str(j.child)

            if child_name not in fk:
                continue

            child_pos = fk[child_name][:3, 3].tolist() # joint position = child link origin in world
            joints.append({"name": j.name, "parent": parent_name, "child": child_name, "position": child_pos})

            if parent_name in fk:
                parent_pos = fk[parent_name][:3, 3].tolist()
                edges.append({"from": parent_name, "to": child_name, "p1": parent_pos, "p2": child_pos})
            elif parent_name == "world":
                edges.append({"from": "world", "to": child_name, "p1": [0, 0, 0], "p2": child_pos})

        return {"joints": joints, "edges": edges}

    # ---------- Visuals at FK poses ----------
    @staticmethod
    def _mat3_to_quat_xyzw(R: np.ndarray):
        """3x3 rotation matrix -> quaternion [x, y, z, w]."""
        t = np.trace(R)
        if t > 0.0:
            s = np.sqrt(t + 1.0) * 2.0
            w = 0.25 * s
            x = (R[2,1] - R[1,2]) / s
            y = (R[0,2] - R[2,0]) / s
            z = (R[1,0] - R[0,1]) / s
        else:
            if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
                s = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
                w = (R[2,1] - R[1,2]) / s
                x = 0.25 * s
                y = (R[0,1] + R[1,0]) / s
                z = (R[0,2] + R[2,0]) / s
            elif R[1,1] > R[2,2]:
                s = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
                w = (R[0,2] - R[2,0]) / s
                x = (R[0,1] + R[1,0]) / s
                y = 0.25 * s
                z = (R[1,2] + R[2,1]) / s
            else:
                s = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
                w = (R[1,0] - R[0,1]) / s
                x = (R[0,2] + R[2,0]) / s
                y = (R[1,2] + R[2,1]) / s
                z = 0.25 * s
        return [float(x), float(y), float(z), float(w)]

    @staticmethod
    def _dae_to_obj_name(src: str) -> str:
        """Map a URDF .dae filename to your .obj filename (adjust if needed)."""
        base = os.path.basename(src)          # e.g., 'upperarm.dae'
        stem, _ = os.path.splitext(base)      # 'upperarm'
        # If any names differ, fix here:
        # name_map = {"upper_arm": "upperarm"}
        # stem = name_map.get(stem, stem)
        return f"{stem}.obj"

    def visuals_world_poses(self, joint_cfg=None):
        """For each link.visual: world_T = FK(link) @ visual.origin."""
        if joint_cfg is None:
            joint_cfg = {j.name: 0.0 for j in self.robot.joints}
    
        # first_joint_name = self.robot.joints[2].name
        # joint_cfg[first_joint_name] = np.deg2rad(30.0)
        # print (first_joint_name)
        # first_joint_name = self.robot.joints[3].name
        # joint_cfg[first_joint_name] = np.deg2rad(30.0)
        # print (first_joint_name)
        # first_joint_name = self.robot.joints[4].name
        # joint_cfg[first_joint_name] = np.deg2rad(30.0)
        # print (first_joint_name)
        fk = self.fk_by_name(joint_cfg)  # {link_name: 4x4}
        visuals = []

        # Walk links by name (to access .visuals)
        links_by_name = {L.name: L for L in self.robot.links}

        for lname, link_T in fk.items():
            link = links_by_name.get(lname)
            if not link or not getattr(link, "visuals", None):
                continue

            for vis in link.visuals:
                if not vis.geometry or not vis.geometry.mesh or not vis.geometry.mesh.filename:
                    continue
                
                V = vis.origin if vis.origin is not None else np.eye(4)
                world_T = link_T @ V
                pos = world_T[:3, 3]
                R = world_T[:3, :3]
                quat = self._mat3_to_quat_xyzw(R)

                obj_name = self._dae_to_obj_name(vis.geometry.mesh.filename)
                if OBJ_FS_DIR and not os.path.exists(os.path.join(OBJ_FS_DIR, obj_name)):
                    print(f"⚠️ Missing OBJ on disk: {obj_name}")

                visuals.append({
                    "link": lname,
                    "mesh": obj_name,  # filename only; client prepends OBJ_BASE_URL
                    "position": [float(pos[0]), float(pos[1]), float(pos[2])],
                    "quaternion": quat,
                    # URDF in meters; set to 0.001 if your OBJs are in millimeters
                    "scale": 1.0
                })
        return visuals


# Instantiate
visualizer = URDFVisualizer(URDF_PATH)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/joints")
def api_joints():
    # Optional joint angles via query params
    cfg = {}
   
    for j in visualizer.robot.joints:
        if j.name in request.args:
            try:
                cfg[j.name] = float(request.args[j.name])
            except ValueError:
                pass
    cfg = cfg or None
    return jsonify(visualizer.get_joint_positions(joint_cfg=cfg))

@app.route("/api/visuals")
def api_visuals():
    # Optional joint angles via query params
    cfg = {}
    for j in visualizer.robot.joints:
        if j.name in request.args:
            try:
                cfg[j.name] = float(request.args[j.name])
            except ValueError:
                pass
    cfg = cfg or None

    visuals = visualizer.visuals_world_poses(joint_cfg=cfg)
    return jsonify({"baseUrl": OBJ_BASE_URL, "visuals": visuals})

if __name__ == "__main__":
    app.run(debug=True)
