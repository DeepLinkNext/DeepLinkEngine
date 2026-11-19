"""Standalone component candidate: preserve the restore BF16 boundary before native gated RMS arithmetic."""
from vllm.triton_utils import tl,triton


@triton.jit(do_not_specialize=['eps'])
def restore_gated_rms(P,V,MAP,Z,W,Y,Rstd,T:tl.constexpr,eps,BLOCK_M:tl.constexpr):
    token=tl.program_id(0)*BLOCK_M+tl.arange(0,BLOCK_M)
    head=tl.program_id(1)
    cols=tl.arange(0,128)
    valid_token=tl.where(token<T,token,0)
    source=tl.load(MAP+valid_token,token<T,other=0)
    spec=source>=T
    # Original address-safety contract: both branch addresses remain in storage,
    # even if Ascend lowers masked gathers to post-load selection.
    pre_source=tl.where(spec,0,source)
    ver_source=tl.where(spec,source-T,0)
    p=tl.load(P+(head*T+pre_source[:,None])*128+cols[None,:],
        (token[:,None]<T)&~spec[:,None],other=0)
    v=tl.load(V+ver_source[:,None]*2048+head*128+cols[None,:],
        (token[:,None]<T)&spec[:,None],other=0)
    # The old restore materializes x.dtype before the RMS reads it. Preserve
    # that rounding even if either producer is FP32, avoiding an accidental
    # precision change disguised as layout fusion.
    x=tl.where(spec[:,None],v,p).to(Y.dtype.element_ty).to(tl.float32)
    row_mask=token<T
    col_mask=cols<128
    w=tl.load(W+cols,mask=col_mask).to(tl.float32)
    z=tl.load(Z+valid_token[:,None]*2048+head*128+cols[None,:],
        mask=row_mask[:,None]&col_mask[None,:]).to(tl.float32)
    xbar=tl.where(col_mask[None,:],x,0.0)
    var=tl.sum(xbar*xbar,axis=1)/128
    rstd=1.0/tl.sqrt(var+eps)
    tl.store(Rstd+valid_token*16+head,rstd,mask=row_mask)
    x_hat=x*rstd[:,None]
    y=x_hat*w[None,:]
    y*=z*tl.sigmoid(z)
    tl.store(Y+valid_token[:,None]*2048+head*128+cols[None,:],y,
        mask=row_mask[:,None]&col_mask[None,:])
