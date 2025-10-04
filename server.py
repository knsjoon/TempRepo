from flask import Flask, jsonify, render_template, request
from urdfpy import URDF
import numpy as np
import os

class URDFVisualizer:
    def __init__(self, urdf_path):
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")
        self.robot = URDF.load(urdf_path)

    def get_joint_positions(self, joint_cfg=None):
        # default all joint positions to 0.0 (radians / meters)
        if joint_cfg is None:
            joint_cfg = {j.name: 0.0 for j in self.robot.joints}

        # Forward kinematics: dict {LinkObj -> 4x4}
        link_fk = self.robot.link_fk(cfg=joint_cfg)

        # 🔑 Normalize to names so we never compare Link objects
        fk_by_name = {link.name: T for link, T in link_fk.items()}

        print("FK links:", sorted(fk_by_name.keys()))
        print("========================================")

        joints, edges = [], []

        for j in self.robot.joints:
            parent_name = j.parent.name if hasattr(j.parent, "name") else str(j.parent)
            child_name  = j.child.name  if hasattr(j.child,  "name") else str(j.child)

            # Child link pose in world; if missing, skip
            if child_name not in fk_by_name:
                print(f"⚠️ Missing child in FK (by name): {child_name}")
                continue

            child_T = fk_by_name[child_name]
            child_pos = child_T[:3, 3].tolist()

            joints.append({
                "name": j.name,
                "parent": parent_name,
                "child": child_name,
                "position": child_pos
            })

            # Edge parent → child (use world = origin when parent doesn't exist in FK)
            if parent_name in fk_by_name:
                parent_pos = fk_by_name[parent_name][:3, 3].tolist()
                edges.append({"from": parent_name, "to": child_name, "p1": parent_pos, "p2": child_pos})
            elif parent_name == "world":
                edges.append({"from": "world", "to": child_name, "p1": [0.0, 0.0, 0.0], "p2": child_pos})
            else:
                # Parent not in FK and not 'world' -> skip edge but keep joint node
                print(f"ℹ️ Parent not in FK and not 'world': {parent_name}")

        print(f"✅ Joints found: {len(joints)}")
        print(f"✅ Edges found:  {len(edges)}")
        return {"joints": joints, "edges": edges}


# ---------------- Flask setup ----------------
app = Flask(__name__, template_folder="templates")

URDF_PATH = "/home/seokjoonkim/Desktop/programming/ur10e_bundle/ur10e.urdf"
visualizer = URDFVisualizer(URDF_PATH)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/joints")
def api_joints():
    # Optional: support /api/joints?shoulder_pan_joint=1.0&elbow_joint=0.5
    cfg = {}
    for j in visualizer.robot.joints:
        if j.name in request.args:
            try:
                cfg[j.name] = float(request.args[j.name])
            except ValueError:
                pass
    cfg = cfg or None
    return jsonify(visualizer.get_joint_positions(joint_cfg=cfg))

if __name__ == "__main__":
    app.run(debug=True)
