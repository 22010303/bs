import os
import glob
from dm_control import mjcf


class PIHArena(object):
    """Arena for single-arm peg-in-hole task with table, embedded hole, and cameras."""

    def __init__(self) -> None:
        self._mjcf_model = mjcf.RootElement()

        self._mjcf_model.option.timestep = 0.002
        self._mjcf_model.option.flag.warmstart = "enable"

        self._mjcf_model.compiler.angle = 'radian'

        # Texture and meshes directories
        assets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../assets/'))
        self._mjcf_model.compiler.texturedir = assets_dir
        pih_meshes_dir = os.path.join(assets_dir, 'peg-in-hole/')

        # Visual quality
        self._mjcf_model.visual.quality.shadowsize = 8192
        self._mjcf_model.visual.__getattr__('global').offwidth = 1280
        self._mjcf_model.visual.__getattr__('global').offheight = 960

        # Default classes for hole collision/visual geoms
        visual_default = self._mjcf_model.default.add('default', dclass='visual')
        visual_default.geom.set_attributes(
            group=3, type='mesh', contype=0, conaffinity=0
        )
        collision_default = self._mjcf_model.default.add('default', dclass='collision')
        collision_default.geom.set_attributes(
            group=2, type='mesh',
            solimp=[0.9, 0.95, 0.001, 0.5, 2],
            solref=[0.02, 1],
            condim=6
        )

        # Floor texture (reuse light-wood.png)
        texture = self._mjcf_model.asset.add(
            "texture",
            type="2d",
            file="arenas/light-wood.png",
            width=300,
            height=300,
        )
        grid = self._mjcf_model.asset.add(
            "material",
            name="grid",
            texture=texture,
            texrepeat=[5, 5],
        )

        # Wood material for table
        self._mjcf_model.asset.add(
            "material", name="Wood", rgba=[0.6, 0.4, 0.2, 1.0]
        )

        # Material for hole
        self._mjcf_model.asset.add(
            'material', name='hole_mat',
            specular=0.8, shininess=0.5, rgba=[0.8, 0.8, 0.8, 1]
        )

        # Floor
        self._mjcf_model.worldbody.add(
            "geom", type="plane", size=[2, 2, 0.1], material=grid
        )

        # Table (from peg_in_hole_stl.xml)
        table_body = self._mjcf_model.worldbody.add('body', name='table', pos=[0, 0, 0])
        table_body.add('geom', type='box', size=[1.2, 1.2, 0.05], pos=[0, 0, 1], material='Wood')
        table_body.add('geom', type='cylinder', size=[0.05, 0.5], pos=[-1.0, -1.0, 0.5], material='Wood')
        table_body.add('geom', type='cylinder', size=[0.05, 0.5], pos=[1.0, -1.0, 0.5], material='Wood')
        table_body.add('geom', type='cylinder', size=[0.05, 0.5], pos=[-1.0, 1.0, 0.5], material='Wood')
        table_body.add('geom', type='cylinder', size=[0.05, 0.5], pos=[1.0, 1.0, 0.5], material='Wood')

        # Hole embedded in arena (fixed body on table, no freejoint)
        # Use hole_27 meshes from peg_hole_stl/ (matching peg_in_hole_stl.xml)
        # These meshes are in mm units, so scale=0.001 to convert to meters
        hole_27_dir = os.path.join(pih_meshes_dir, 'peg_hole_stl', 'hole_27')
        num_hole_collision = len(glob.glob(os.path.join(hole_27_dir, 'hole_27_collision_*.obj')))

        for i in range(num_hole_collision):
            self._mjcf_model.asset.add(
                'mesh', name=f'hole_collision_{i}',
                file=os.path.join(hole_27_dir, f'hole_27_collision_{i}.obj'),
                scale=[0.001, 0.001, 0.001]
            )
        self._mjcf_model.asset.add(
            'mesh', name='hole_visual',
            file=os.path.join(hole_27_dir, 'hole_27.obj'),
            scale=[0.001, 0.001, 0.001]
        )

        hole_body = self._mjcf_model.worldbody.add(
            'body', name='hole',
            euler=[1.5707, 0, 0],
            pos=[0, 0.2, 1.03]
        )
        hole_body.add(
            'geom', material='hole_mat', mesh='hole_visual',
            dclass='visual', mass=0.05
        )
        for i in range(num_hole_collision):
            hole_body.add(
                'geom', mesh=f'hole_collision_{i}', dclass='collision'
            )

        # Camera target body (centered between arm base and hole)
        self._mjcf_model.worldbody.add(
            'body', name='camera_center',
            euler=[0, 0, 0], pos=[-0.1, 0.1, 1.1]
        )

        # Overhead camera (camera_id=0)
        cam_body = self._mjcf_model.worldbody.add(
            'body', name='fixed_camera_body',
            euler=[0, 0, 0], pos=[0.3, 0.0, 2.3]
        )
        cam_body.add(
            'camera', name='fixed_camera',
            mode='targetbody', target='camera_center', fovy=42.5
        )

        # Side camera for close-up insertion view (camera_id=1)
        side_cam_body = self._mjcf_model.worldbody.add(
            'body', name='side_camera_body',
            pos=[0.35, -0.15, 1.3]
        )
        side_cam_body.add(
            'camera', name='side_camera',
            mode='targetbody', target='hole',
            fovy=30
        )

        # Lighting
        for x in [-1, 1]:
            self._mjcf_model.worldbody.add(
                "light", pos=[2 * x, -1, 4], dir=[-2 * x, 1, -2]
            )

    def attach(self, child, pos=None, quat=None) -> mjcf.Element:
        if pos is None:
            pos = [0, 0, 0]
        if quat is None:
            quat = [1, 0, 0, 0]
        frame = self._mjcf_model.attach(child)
        frame.pos = pos
        frame.quat = quat
        return frame

    def attach_free(self, child, pos=None, quat=None) -> mjcf.Element:
        if pos is None:
            pos = [0, 0, 0]
        if quat is None:
            quat = [1, 0, 0, 0]
        frame = self.attach(child, pos, quat)
        frame.add('freejoint')
        return frame

    @property
    def mjcf_model(self) -> mjcf.RootElement:
        return self._mjcf_model
