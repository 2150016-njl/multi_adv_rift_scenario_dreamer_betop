"""V5 joint two-actor CARLA raw-state generator.

Unlike V2-V4C, this contains NO frozen NuPlan autoencoder/latent diffusion.
One forward pass chooses a JOINT ordered pair of online map-only approach
candidates, and one Gaussian draw samples all eight continuous coordinates.
The differentiable lane chart is *inside the model's output head*, not an
external snap/clamp/rejection of the resulting [x,y,yaw,speed] states.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

from high_value_init.v5_candidates import MAX_CANDIDATES, MAX_LANE_POINTS

FEAT_DIM = 12
LAT_LIMIT = 1.35              # intrinsic learned chart support, metres
YAW_LIMIT = 0.40              # intrinsic learned chart support, radians
SPEED_MAX = 12.0              # intrinsic learned chart support, m/s
SIZE_FIXED = (4.624, 1.909)   # documentation only; CARLA blueprint is authoritative


def _angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


def _logit(x):
    x = x.clamp(1.e-4, 1.-1.e-4)
    return torch.log(x/(1.-x))


def candidate_tensors(candidates, device='cpu'):
    """Convert map-only candidates into immutable condition tensors; no targets."""
    n = len(candidates)
    if not 1 <= n <= MAX_CANDIDATES:
        raise ValueError('V5 requires 1..%s map candidates, got %s' % (MAX_CANDIDATES, n))
    features = torch.zeros((n, FEAT_DIM), dtype=torch.float32, device=device)
    polylines = torch.zeros((n, MAX_LANE_POINTS, 2), dtype=torch.float32, device=device)
    counts = torch.zeros((n,), dtype=torch.long, device=device)
    anchor = torch.zeros((n,), dtype=torch.float32, device=device)
    ego_s = torch.zeros((n,), dtype=torch.float32, device=device)
    for i,c in enumerate(candidates):
        p = torch.tensor(c['polyline'], dtype=torch.float32, device=device)
        if p.ndim != 2 or len(p) < 2 or len(p) > MAX_LANE_POINTS:
            raise ValueError('Invalid candidate polyline')
        counts[i] = len(p)
        polylines[i, :len(p)] = p
        anchor[i] = float(c['anchor_s'])
        ego_s[i] = float(c['ego_route_s'])
        dx = math.cos(float(c['anchor_heading']))
        dy = math.sin(float(c['anchor_heading']))
        features[i] = torch.tensor([
            c['anchor_xy'][0] / 32., c['anchor_xy'][1] / 32.,
            dx, dy, c['ego_route_s']/32., c['anchor_s']/64.,
            c['lane_length']/64., c['distance_to_ego_route']/3.,
            c['polyline'][0][0]/32., c['polyline'][0][1]/32.,
            c['polyline'][-1][0]/32., c['polyline'][-1][1]/32.,
        ], dtype=torch.float32, device=device)
    return {'features':features, 'polylines':polylines, 'counts':counts,
            'anchor_s':anchor, 'ego_s':ego_s}


def context_features(condition, device='cpu'):
    """Scene features: ego motion, route outline, and observed context only."""
    ego = condition.get('ego_state') or [0.,0.,0.,0.]
    route = (condition.get('ego_route') or {}).get('polyline') or []
    agents = condition.get('observed_agent_states') or []
    # 14 dims; do not use target labels, RIFT score, target route or actor IDs.
    if len(route)>1:
        a, b = route[0], route[-1]
        route_descriptor = [float(a[0])/32.,float(a[1])/32.,float(b[0])/32.,float(b[1])/32.]
    else:
        route_descriptor=[0.]*4
    if agents:
        pos = torch.tensor([[float(t[0]),float(t[1])] for t in agents], dtype=torch.float32)
        mean = (pos/32.).mean(dim=0).tolist()
        nearest = float(torch.linalg.norm(pos,dim=1).min()) / 32.
        speed = sum(float(a[3]) for a in agents)/len(agents)/12.
    else:
        mean=[0.,0.]; nearest=1.; speed=0.
    vec=[float(ego[3])/12., len(agents)/20., min(nearest,2.), speed]+mean+route_descriptor
    vec += [float(ego[4])/6. if len(ego)>4 else 0.,
            float(ego[5])/3. if len(ego)>5 else 0.,
            math.sin(float(ego[2])) if len(ego)>2 else 0.,
            math.cos(float(ego[2])) if len(ego)>2 else 1.]
    return torch.tensor(vec,dtype=torch.float32,device=device)


def _interpolate(poly, count, s):
    """Differentiable interpolation at learned *arclength*, not a posthoc snap."""
    poly = poly[:int(count)]
    delta = poly[1:]-poly[:-1]
    lengths = torch.linalg.norm(delta,dim=-1).clamp_min(1.e-6)
    prefix = torch.cat([torch.zeros_like(lengths[:1]), lengths.cumsum(dim=0)])
    # Model already constrains s within support [0, anchor_s].
    idx = (prefix[1:].detach() <= s.detach()).sum().clamp(max=len(lengths)-1)
    fraction = (s-prefix[idx])/lengths[idx]
    position = poly[idx]+fraction*delta[idx]
    tangent = torch.atan2(delta[idx,1],delta[idx,0])
    return position,tangent


class CARLAJointRawGenerator(nn.Module):
    """Joint categorical approach + eight continuous chart coordinates."""
    version = 'carla_joint_raw_v5_1'
    def __init__(self, hidden=128):
        super().__init__()
        self.hidden=int(hidden)
        self.scene_net=nn.Sequential(nn.Linear(14,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU())
        self.candidate_net=nn.Sequential(nn.Linear(FEAT_DIM,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU())
        self.pair_net=nn.Sequential(nn.Linear(hidden*4,hidden),nn.SiLU(),nn.Linear(hidden,1))
        self.state_net=nn.Sequential(nn.Linear(hidden*4,hidden*2),nn.SiLU(),
                                     nn.Linear(hidden*2,hidden),nn.SiLU(),nn.Linear(hidden,16))

    def distribution(self, candidates, condition):
        device=next(self.parameters()).device
        t=candidate_tensors(candidates,device)
        sc=self.scene_net(context_features(condition,device))
        h=self.candidate_net(t['features'])
        n=h.shape[0]
        h0=h[:,None,:].expand(n,n,-1)
        h1=h[None,:,:].expand(n,n,-1)
        sc2=sc.view(1,1,-1).expand(n,n,-1)
        pair_features=torch.cat([h0,h1,h0*h1,sc2],dim=-1)
        logits=self.pair_net(pair_features).squeeze(-1)
        return logits,h,sc,t

    def normal_params(self,h,sc,indices):
        i,j=int(indices[0]),int(indices[1])
        feats=torch.cat([h[i],h[j],h[i]*h[j],sc])
        p=self.state_net(feats)
        means=p[:8].reshape(2,4)
        # Restrict standard deviation in the *learned Gaussian*; not a state clamp.
        log_stds=-3.5+2.5*torch.sigmoid(p[8:].reshape(2,4))
        return means,log_stds

    def decode(self,z,indices,t):
        """MODEL HEAD: intrinsic lane coordinate chart -> native ego-local x/y/yaw/v."""
        rows=[]
        for slot,index in enumerate(indices):
            k=int(index)
            # Progress always upstream of route-contact anchor and within lane.
            s=(t['anchor_s'][k]-0.30).clamp_min(0.3)*torch.sigmoid(z[slot,0])
            position,tangent=_interpolate(t['polylines'][k],t['counts'][k],s)
            lateral=LAT_LIMIT*torch.tanh(z[slot,1])
            heading=tangent+YAW_LIMIT*torch.tanh(z[slot,2])
            speed=SPEED_MAX*torch.sigmoid(z[slot,3])
            normal=torch.stack([-torch.sin(tangent),torch.cos(tangent)])
            xy=position+lateral*normal
            rows.append(torch.stack([xy[0],xy[1],_angle(heading),speed]))
        return torch.stack(rows)

    def target_latents(self,candidates,targets,indices):
        """OFFLINE teacher-forcing encoding for two observed RIFT positives."""
        from high_value_init.v5_candidates import closest
        latents=[]
        for slot,index in enumerate(indices):
            c=candidates[int(index)]
            target=targets[slot]
            d,s,tangent,projected=closest(torch.tensor(c['polyline']).numpy(),target[:2])
            perp=(-(math.sin(tangent)),math.cos(tangent))
            lat=(float(target[0])-projected[0])*perp[0]+(float(target[1])-projected[1])*perp[1]
            yaw=math.atan2(math.sin(float(target[2])-tangent),math.cos(float(target[2])-tangent))
            frac=s/max(float(c['anchor_s'])-0.30,0.3)
            if not (0.001 < frac < .999 and abs(lat) < LAT_LIMIT*.98 and
                    abs(yaw) < YAW_LIMIT*.98 and 0.0 <= float(target[3]) < SPEED_MAX*.999):
                raise ValueError('Target outside intrinsic V5 chart support')
            latents.append([math.log(frac/(1-frac)),math.atanh(lat/LAT_LIMIT),
                            math.atanh(yaw/YAW_LIMIT),float(_logit(torch.tensor(max(target[3]/SPEED_MAX,1.e-4))))])
        return torch.tensor(latents,dtype=torch.float32,device=next(self.parameters()).device)

    def training_losses(self,row):
        candidates=row['candidates']; condition=row['condition']; targets=row['targets']
        indices=row['assigned_indices']
        logits,h,sc,t=self.distribution(candidates,condition)
        n=logits.shape[0]
        ce=F.cross_entropy(logits.reshape(1,-1),torch.tensor([indices[0]*n+indices[1]],device=logits.device))
        mu,log_std=self.normal_params(h,sc,indices)
        truth=self.target_latents(candidates,targets,indices)
        # The decoder is differentiable in its 4D latent coordinates. No GT leakage at inference.
        nll=(.5*((truth-mu)/log_std.exp()).square()+log_std).mean()
        pred=self.decode(mu,indices,t)
        goal=torch.tensor([x[:4] for x in targets],dtype=torch.float32,device=logits.device)
        xy_err=(pred[:,:2]-goal[:,:2]).abs().mean()
        yaw_err=_angle(pred[:,2]-goal[:,2]).abs().mean()
        speed_err=(pred[:,3]-goal[:,3]).abs().mean()
        route_gap=[]
        for slot,idx in enumerate(indices):
            c=candidates[int(idx)]
            # Explicit static *surrogate*; NOT RIFT's CARLA trace_route oracle.
            remaining=(t['anchor_s'][idx]-0.30)*(1.-torch.sigmoid(mu[slot,0]))
            ego_progress=t['ego_s'][idx]
            route_gap.append((remaining-ego_progress).abs())
        # Keep route proxy weak: mis-approximating CARLA trace_route must not destroy feasibility.
        route_proxy=torch.stack(route_gap).mean()
        others=[[0.0,0.0]]+[o[:2] for o in condition.get('observed_agent_states',[])]
        others_t=torch.tensor(others,device=logits.device,dtype=torch.float32)
        dists=torch.cdist(pred[:,:2],others_t)
        clear_others=F.relu(4.0-dists).mean()
        clear_pair=F.relu(5.0-torch.linalg.norm(pred[0,:2]-pred[1,:2]))
        clearance=clear_pair+clear_others
        total=ce+0.25*nll+0.25*xy_err+0.35*yaw_err+0.07*speed_err+0.01*route_proxy+0.15*clearance
        return {'loss':total,'pair_ce':ce,'latent_nll':nll,'xy_l1_m':xy_err,
                'yaw_abs_rad':yaw_err,'speed_l1_mps':speed_err,
                'static_route_gap_m':route_proxy,'clearance_proxy_m':clearance}

    @torch.no_grad()
    def sample_once(self,condition,candidates,stochastic=True):
        """Exactly one joint pair sample and one 8D Gaussian draw, no retries."""
        self.eval()
        logits,h,sc,t=self.distribution(candidates,condition)
        n=logits.shape[0]
        if stochastic:
            # ONE categorical draw over all ordered two-actor candidate pairs.
            pair_index=int(torch.multinomial(F.softmax(logits.flatten(),dim=0),1).item())
        else:
            pair_index=int(torch.argmax(logits.flatten()).item())
        indices=[pair_index//n,pair_index%n]
        mu,ls=self.normal_params(h,sc,indices)
        z=mu+ls.exp()*torch.randn_like(mu) if stochastic else mu
        states=self.decode(z,indices,t)
        if not bool(torch.isfinite(states).all()):
            raise RuntimeError('Non-finite raw model output: no fallback or resampling')
        raw=states.cpu().tolist()
        for row in raw:
            row.extend([SIZE_FIXED[0],SIZE_FIXED[1],1.0])
        return {'generated_agent_states':raw,'candidate_indices':indices,
                'sampler':'one_joint_categorical_one_8d_gaussian',
                'raw_state_postprocessing':'none'}
