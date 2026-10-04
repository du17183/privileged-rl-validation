"""Author a graspable bar handle on Isaac's articulated Sektion right door.

The referenced source articulation (including its hinge and joint limits) is
unchanged. The additional collider and visible cube belong to its knob body.
"""

from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app

from pathlib import Path
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

root = Path(__file__).resolve().parents[1]
path = root / "assets" / "panda_door_cabinet.usd"
source = f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd"
stage = Usd.Stage.CreateNew(str(path))
cabinet = UsdGeom.Xform.Define(stage, Sdf.Path("/Cabinet"))
stage.SetDefaultPrim(cabinet.GetPrim())
cabinet.GetPrim().GetReferences().AddReference(source)
bar = UsdGeom.Cube.Define(stage, Sdf.Path("/Cabinet/door_right_nob_link/grasp_bar"))
bar.CreateSizeAttr(1.0)
bar.AddTranslateOp().Set(Gf.Vec3d(0.13, -0.35, 0.185))
bar.AddScaleOp().Set(Gf.Vec3f(0.12, 0.065, 0.025))
bar.CreateDisplayColorAttr([Gf.Vec3f(0.15, 0.35, 0.85)])
UsdPhysics.CollisionAPI.Apply(bar.GetPrim()).CreateCollisionEnabledAttr(True)
stage.GetRootLayer().Save()
print("ASSET",path,"SOURCE",source,flush=True)
print("CHILDREN",[p.GetName() for p in cabinet.GetPrim().GetChildren()],flush=True)
app.close()
