"""Analytic spherical Jacobians and CUDA sparse damped least squares.

No upstream engines are modified. The CPU path uses the same analytic
Jacobian. CUDA solves scaled sparse normal equations with preconditioned CG;
it never materializes a dense camera/point Hessian.
"""
import time
from types import SimpleNamespace
import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation


def skew(v):
    v=np.asarray(v);out=np.zeros(v.shape[:-1]+(3,3))
    out[...,0,1]=-v[...,2];out[...,0,2]=v[...,1]
    out[...,1,0]=v[...,2];out[...,1,2]=-v[...,0]
    out[...,2,0]=-v[...,1];out[...,2,1]=v[...,0]
    return out


def left_jacobian(v):
    theta=np.linalg.norm(v,axis=1);t2=theta**2
    a=np.empty_like(theta);b=np.empty_like(theta);small=theta<1e-4
    a[small]=.5-t2[small]/24+t2[small]**2/720
    b[small]=1/6-t2[small]/120+t2[small]**2/5040
    a[~small]=(1-np.cos(theta[~small]))/t2[~small]
    b[~small]=(theta[~small]-np.sin(theta[~small]))/theta[~small]**3
    k=skew(v);return np.eye(3)+a[:,None,None]*k+b[:,None,None]*(k@k)


class BearingProblem:
    def __init__(self,cameras,free_names,point_ids,measurements,gcps,distances,scale_name,baseline,scale_anchor,anchor_camera=None):
        self.names=list(cameras);self.free=list(free_names);self.ids=list(point_ids)
        self.nc=len(self.free);self.offset=self.nc*6;self.nparams=self.offset+3*len(self.ids)
        self.cmap={n:i for i,n in enumerate(self.names)};self.pmap={n:i for i,n in enumerate(self.ids)}
        self.free_rows=np.array([self.cmap[n] for n in self.free],dtype=int)
        self.free_map={n:i for i,n in enumerate(self.free)}
        self.r0=np.array([cameras[n][0] for n in self.names]);self.t0=np.array([cameras[n][1] for n in self.names])
        self.ci=np.array([self.cmap[m[1]] for m in measurements]);self.pi=np.array([self.pmap[m[0]] for m in measurements])
        self.fi=np.array([self.free_map.get(m[1],-1) for m in measurements]);self.mask=self.fi>=0
        self.b=np.array([m[2] for m in measurements])
        self.weight=np.array([np.eye(3)/m[3] if np.ndim(m[3])==0 else m[3] for m in measurements])
        self.m=len(measurements);self.gcps=gcps;self.distances=distances
        self.anchor=scale_name if scale_anchor else None;self.baseline=baseline
        first=self.cmap[anchor_camera or sorted(cameras)[0]]
        self.fixed_center=-self.r0[first].T@self.t0[first]
        self.rows=3*self.m+3*len(gcps)+len(distances)+int(scale_anchor)
        base=np.arange(self.m)*3
        self.pr=(base[:,None,None]+np.arange(3)[None,:,None]+np.zeros((1,1,3),int)).ravel()
        self.pc=(self.offset+self.pi[:,None,None]*3+np.arange(3)[None,None,:]+np.zeros((1,3,1),int)).ravel()
        self.cr=(base[self.mask,None,None]+np.arange(3)[None,:,None]+np.zeros((1,1,6),int)).ravel()
        self.cc=(self.fi[self.mask,None,None]*6+np.arange(6)[None,None,:]+np.zeros((1,3,1),int)).ravel()
        self.cache=None
    def unpack(self,x):
        r=self.r0.copy();t=self.t0.copy()
        if self.nc:
            poses=x[:self.offset].reshape(-1,6);r[self.free_rows]=Rotation.from_rotvec(poses[:,:3]).as_matrix();t[self.free_rows]=poses[:,3:]
        return r,t,x[self.offset:].reshape(-1,3)
    def evaluate(self,x,jac=False,weighted=True):
        r,t,p=self.unpack(x);rotated=np.einsum('nij,nj->ni',r[self.ci],p[self.pi]);q=rotated+t[self.ci]
        norm=np.maximum(np.linalg.norm(q,axis=1),1e-12);u=q/norm[:,None]
        residual=np.einsum('nij,nj->ni',self.weight,u-self.b) if weighted else u-self.b
        if not weighted:return residual.ravel()
        extra=[];er=[];ec=[];ev=[];row=3*self.m
        if jac:
            d=self.weight@((np.eye(3)-u[:,:,None]*u[:,None,:])/norm[:,None,None])
            jp=d@r[self.ci];jr=-(d[self.mask]@skew(rotated[self.mask])@left_jacobian(x[:self.offset].reshape(-1,6)[self.fi[self.mask],:3]))
            jc=np.concatenate([jr,d[self.mask]],axis=2)
        def block(row,col,values):
            rr,cc=np.indices(values.shape);er.extend((row+rr).ravel());ec.extend((col+cc).ravel());ev.extend(values.ravel())
        for name,g in self.gcps.items():
            index=self.pmap[name];sigma=float(g.get('sigma_m',.01))
            extra.extend((p[index]-np.asarray(g['position']))/sigma)
            if jac:block(row,self.offset+index*3,np.eye(3)/sigma)
            row+=3
        for distance in self.distances:
            a,b=self.pmap[str(distance['a'])],self.pmap[str(distance['b'])];delta=p[a]-p[b]
            length=max(np.linalg.norm(delta),1e-12);sigma=float(distance.get('sigma_m',.01))
            extra.append((length-float(distance['meters']))/sigma)
            if jac:
                block(row,self.offset+a*3,(delta/(length*sigma))[None]);block(row,self.offset+b*3,(-delta/(length*sigma))[None])
            row+=1
        if self.anchor:
            index=self.cmap[self.anchor];center=-r[index].T@t[index];delta=center-self.fixed_center
            length=max(np.linalg.norm(delta),1e-12);sigma=max(self.baseline*.001,1e-8)
            extra.append((length-self.baseline)/sigma)
            if jac:
                fi=self.free_map[self.anchor];jl=left_jacobian(x[fi*6:fi*6+3][None])[0]
                derivative=np.concatenate([-r[index].T@skew(t[index])@jl,-r[index].T],axis=1)
                block(row,fi*6,(delta/(length*sigma))[None]@derivative)
        result=np.concatenate([residual.ravel(),np.asarray(extra)])
        if not jac:return result
        matrix=coo_matrix((np.concatenate([jp.ravel(),jc.ravel(),ev]),
                          (np.concatenate([self.pr,self.cr,er]).astype(int),np.concatenate([self.pc,self.cc,ec]).astype(int))),
                         shape=(self.rows,self.nparams)).tocsr()
        return result,matrix


def cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:return False


def retriangulate_batch(names,observations,cameras):
    """Triangulate a bounded point batch with the scalar solver's safeguards."""
    cmap={n:i for i,n in enumerate(cameras)}
    camera_names=list(cameras);rotation=np.asarray([cameras[n][0] for n in camera_names])
    translation=np.asarray([cameras[n][1] for n in camera_names])
    rows=[(i,cmap[shot],pixel) for i,name in enumerate(names) for shot,pixel,_ in observations[name]]
    pi=np.asarray([row[0] for row in rows]);ci=np.asarray([row[1] for row in rows]);pixel=np.asarray([row[2] for row in rows])
    from manual_camera_geometry import bearings_batch
    local=np.empty((len(rows),3))
    for index in np.unique(ci):
        mask=ci==index;local[mask]=bearings_batch(pixel[mask],cameras[camera_names[index]])
    rays=np.einsum('nji,nj->ni',rotation[ci],local);centers=-np.einsum('nji,nj->ni',rotation[ci],translation[ci])
    projectors=np.eye(3)-rays[:,:,None]*rays[:,None,:]
    normal=np.zeros((len(names),3,3));rhs=np.zeros((len(names),3))
    np.add.at(normal,pi,projectors);np.add.at(rhs,pi,np.einsum('nij,nj->ni',projectors,centers))
    eigenvalues=np.linalg.eigvalsh(normal);valid=eigenvalues[:,0]>eigenvalues[:,2]/1e6
    xyz=np.zeros((len(names),3));xyz[valid]=np.linalg.solve(normal[valid],rhs[valid,:,None])[:,:,0]
    behind=np.sum((xyz[pi]-centers)*rays,axis=1)<=0;valid[np.unique(pi[behind])]=False
    return xyz,valid


def robust_values(residual,loss='soft_l1',scale=2.):
    z=(residual/scale)**2
    if loss=='linear':rho=z;derivative=np.ones_like(z)
    elif loss=='soft_l1':rho=2*(np.sqrt(1+z)-1);derivative=1/np.sqrt(1+z)
    elif loss=='huber':rho=np.where(z<=1,z,2*np.sqrt(z)-1);derivative=1/np.sqrt(np.maximum(z,1))
    else:raise ValueError('Choose soft_l1, huber or linear robust loss.')
    return float(.5*scale**2*np.sum(rho)),np.sqrt(derivative)


def solve_cuda(problem,initial,iterations,progress=None,loss='soft_l1',loss_scale=2.,ftol=1e-8,xtol=1e-8,gtol=1e-7):
    import torch
    device='cuda';dtype=torch.float64;started=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    baseline_memory=torch.cuda.memory_allocated()
    x=initial.copy();mu=1e-3;accepted=0;success=False;termination='Iteration budget reached'
    def cost(residual):return robust_values(residual,loss,loss_scale)[0]
    def tensor(v):return torch.as_tensor(v,device=device,dtype=dtype)
    with torch.no_grad():
        for iteration in range(iterations):
            residual,jac=problem.evaluate(x,jac=True);objective=cost(residual)
            weight=robust_values(residual,loss,loss_scale)[1]
            jac=jac.multiply(weight[:,None]).tocsr();rw=residual*weight
            diagonal=np.maximum(np.asarray(jac.power(2).sum(axis=0)).ravel(),1e-12);scale=1/np.sqrt(diagonal)
            jac=jac.multiply(scale[None,:]).tocsr();trans=jac.T.tocsr()
            def sparse(a):
                return torch.sparse_csr_tensor(torch.as_tensor(a.indptr.astype(np.int64),device=device),torch.as_tensor(a.indices.astype(np.int64),device=device),tensor(a.data),size=a.shape)
            j=sparse(jac);jt=sparse(trans);rhs=-torch.mv(jt,tensor(rw));rhs_norm=torch.linalg.vector_norm(rhs)
            if float(rhs_norm)<gtol:success=True;termination='Scaled gradient tolerance reached';break
            pcg_used=0;committed=False
            for trial in range(8):
                def matvec(v):return torch.mv(jt,torch.mv(j,v))+mu*v
                step=torch.zeros_like(rhs);rr=rhs.clone();direction=rr.clone();dot=torch.dot(rr,rr);target=rhs_norm*1e-3
                # Column scaling makes the normal diagonal equal to one;
                # CG therefore works in Jacobi-preconditioned coordinates.
                for cg in range(160):
                    hd=matvec(direction);den=torch.dot(direction,hd).clamp_min(1e-30);alpha=dot/den
                    step+=alpha*direction;rr-=alpha*hd;newdot=torch.dot(rr,rr);pcg_used+=1
                    if (cg+1)%8==0 and float(torch.sqrt(newdot))<=float(target):break
                    direction=rr+(newdot/dot.clamp_min(1e-30))*direction;dot=newdot
                delta=step.cpu().numpy()*scale
                candidate=x+delta;new_residual=problem.evaluate(candidate);newcost=cost(new_residual)
                if np.isfinite(newcost) and newcost<objective:
                    x=candidate;accepted+=1;committed=True;mu=max(mu/3,1e-9)
                    if objective-newcost<=ftol*max(1.,objective):success=True;termination='Robust objective tolerance reached'
                    if np.linalg.norm(delta)<=xtol*(xtol+np.linalg.norm(x)):success=True;termination='Step tolerance reached'
                    break
                mu=min(mu*10,1e12)
            if progress:progress({'phase':'optimize','iteration':iteration+1,'iterations':iterations,'cost':newcost if committed else objective,'accepted':accepted,'pcg_steps':pcg_used,'elapsed':time.perf_counter()-started,'device':torch.cuda.get_device_name()})
            if success:break
            if not committed:termination='No improving damped step; inspect constraints';break
        torch.cuda.synchronize()
        peak=torch.cuda.max_memory_allocated()-baseline_memory
    return SimpleNamespace(x=x,success=success,message=termination,nfev=iteration+1,
                           device=torch.cuda.get_device_name(),peak_vram_mib=peak/1024**2,accepted=accepted)
