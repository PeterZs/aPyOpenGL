from __future__ import annotations
from OpenGL.GL import *
import glm
import copy


from .motion import Skeleton, Pose
from .core   import MeshGL

class Mesh:
    def __init__(
        self,
        mesh_gl: MeshGL,
        materials    = None,
        skeleton: Skeleton = None,
        joint_map: dict[str, str] = None,
    ):
        self.mesh_gl      = mesh_gl
        self.materials    = materials

        # skinning
        self.skeleton     = skeleton
        self.joint_map    = joint_map
        self.joint_map_idx = None
        self.use_skinning = (skeleton is not None)
        self.buffer       = [glm.mat4(1.0)] * len(self.mesh_gl.joint_names)

        if self.skeleton is None and self.joint_map is not None:
            raise ValueError("Joint map requires a skeleton")
        
    def __deepcopy__(self, memo):
        res = Mesh(self.mesh_gl, copy.deepcopy(self.materials), self.skeleton)
        res.buffer = copy.deepcopy(self.buffer)
        memo[id(self)] = res
        return res

    def set_materials(self, materials):
        self.materials = materials

    def update_mesh(self, pose: Pose):
        if self.skeleton is None:
            return
        
        self.buffer = [glm.mat4(1.0) for _ in range(len(self.mesh_gl.joint_names))]
        if self.joint_map is None:
            self._update_without_joint_map(pose)
        else:
            self._update_with_joint_map(pose)

    
    def _update_with_joint_map(self, pose: Pose):
        global_xforms = pose.global_xforms
        if self.joint_map_idx is None:
            self.joint_map_idx = [None for _ in range(len(pose.skeleton.joints))]
            for i in range(len(pose.skeleton.joints)):
                src_jname = pose.skeleton.joints[i].name
                tgt_jname = self.joint_map.get(src_jname, None)
                if tgt_jname is None:
                    continue
                self.joint_map_idx[i] = self.mesh_gl.name_to_idx.get(tgt_jname, None)

        buffer_updated = [False for _ in range(len(self.mesh_gl.joint_names))]
        for i in range(len(pose.skeleton.joints)):
            tgt_idx = self.joint_map_idx[i]
            if tgt_idx is None:
                continue

            # map global xform
            global_xform = glm.mat4(*global_xforms[i].T.ravel())
            bind_xform_inv = self.mesh_gl.bind_xform_inv[tgt_idx]
            self.buffer[tgt_idx] = global_xform * bind_xform_inv
            buffer_updated[tgt_idx] = True

        for i, updated in enumerate(buffer_updated):
            if not updated:
                jname = self.mesh_gl.joint_names[i]
                parent_idx = self.skeleton.parent_idx[self.skeleton.idx_by_name[jname]]
                pjoint_name = self.skeleton.joints[parent_idx].name
                while not buffer_updated[self.mesh_gl.name_to_idx[pjoint_name]]:
                    parent_idx = self.skeleton.parent_idx[parent_idx]
                    pjoint_name = self.skeleton.joints[parent_idx].name
                    if parent_idx == -1:
                        raise ValueError(f"Parent not found for joint {jname}")
                self.buffer[i] = self.buffer[self.mesh_gl.name_to_idx[pjoint_name]]
                buffer_updated[i] = True

    def _update_without_joint_map(self, pose: Pose):
        global_xforms = pose.global_xforms
        for i in range(len(self.mesh_gl.joint_names)):
            jidx = self.skeleton.idx_by_name[self.mesh_gl.joint_names[i]]
            global_xform = glm.mat4(*global_xforms[jidx].T.ravel())
            bind_xform_inv = self.mesh_gl.bind_xform_inv[i]
            self.buffer[i] = global_xform * bind_xform_inv
