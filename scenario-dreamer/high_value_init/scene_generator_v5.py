"""RIFT-facing strict ONE-SHOT V5 entry. No NuPlan decoder or generated-state edit."""

from high_value_init.scene_generator import _snapshot_to_record
from high_value_init.v5_candidates import make_candidates
from high_value_init.v5_model import CARLAJointRawGenerator


class ScenarioDreamerHighValueSceneGeneratorV5:
    def __init__(self, checkpoint_path: str, device='cuda', config=None):
        import torch
        config=config or {}
        self.fov=float(config.get('hv_sd_fov',64.0))
        self.actor_radius=float(config.get('hv_actor_radius',80.0))
        self.mode=str(config.get('hv_generation_mode','full_fov_inpainting'))
        self.map_mode=str(config.get('hv_snapshot_map_coordinate_mode','legacy'))
        if int(config.get('hv_num_generate_agents',2))!=2:
            raise ValueError('V5 requires exactly TWO native CARLA Tesla Model 3 vehicles')
        if self.fov!=64.0 or self.map_mode!='right_handed':
            raise ValueError('V5 requires 64m FOV and CARLA-correct right_handed HD Map')
        if self.mode!='full_fov_inpainting':
            raise ValueError('V5 trains with full-FOV observed actors; require full_fov_inpainting')
        if config.get('hv_generation_clamp_generated_states',False):
            raise ValueError('V5 strictly forbids output clamp')
        # torch 2.0 / torch 2.6 compatible loader, with no implicit fallback.
        try: ck=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        except TypeError: ck=torch.load(checkpoint_path,map_location='cpu')
        if not isinstance(ck,dict) or ck.get('schema_version')!='hv_v5_raw_checkpoint_v1':
            raise ValueError('Refusing checkpoint with unknown V5 provenance')
        if ck.get('config',{}).get('map_mode')!='right_handed':
            raise ValueError('V5 checkpoint map-mode mismatch')
        self.device=torch.device(device)
        self.model=CARLAJointRawGenerator(hidden=int(ck['config']['hidden']))
        self.model.load_state_dict(ck['model'],strict=True)
        self.model.to(self.device).eval()

    def generate(self,snapshot):
        import torch
        if (snapshot.get('diagnostics') or {}).get('map_coordinate_mode')!='right_handed':
            raise ValueError('Snapshot map-coordinate-mode not right_handed')
        record=_snapshot_to_record(snapshot,mode=self.mode,num_generate_agents=2,
                                   actor_radius=self.actor_radius,fov=self.fov)
        cond=record['condition']
        candidates=make_candidates(cond)
        if not candidates:
            raise ValueError('No HD-Map/ego-route interaction candidates; no fabricated fallback states')
        with torch.no_grad():
            result=self.model.sample_once(cond,candidates,stochastic=True)
        result.update({
            'scene_id':snapshot.get('scene_id'),
            'coordinate_frame':'ego_local_rh_x_forward',
            'requested_generation_mode':self.mode,
            'model_generation_mode':'joint_map_candidate_raw_v5',
            'num_online_map_candidates':len(candidates),
            'rift_labels_used_as_input':False,
            'source':'carla_v5_joint_raw_map_intent_single_sample',
        })
        return result
