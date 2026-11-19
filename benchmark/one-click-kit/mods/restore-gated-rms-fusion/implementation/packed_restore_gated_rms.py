"""Gather native activation layout into contiguous token/head RMS rows, retaining BF16 rounding."""
from vllm.triton_utils import tl,triton


@triton.jit(do_not_specialize=['eps'])
def restore_gated_rms(P,V,MAP,Z,W,Y,Rstd,T:tl.constexpr,eps,BLOCK_M:tl.constexpr,PREFILL_ONLY:tl.constexpr):
    rows=tl.program_id(0)*BLOCK_M+tl.arange(0,BLOCK_M)
    row_mask=rows<T*16
    token=rows//16
    head=rows%16
    cols=tl.arange(0,128)
    valid_token=tl.where(row_mask,token,0)
    source=tl.load(MAP+valid_token,row_mask,other=0)
    spec=source>=T
    # Original address-safety contract: both branch addresses remain in storage,
    # even if Ascend lowers masked gathers to post-load selection.
    pre_source=tl.where(spec,0,source)
    p=tl.load(P+(head[:,None]*T+pre_source[:,None])*128+cols[None,:],
        row_mask[:,None]&~spec[:,None],other=0)
    if PREFILL_ONLY:
        producer=p
    else:
        ver_source=tl.where(spec,source-T,0)
        v=tl.load(V+ver_source[:,None]*2048+head[:,None]*128+cols[None,:],
            row_mask[:,None]&spec[:,None],other=0)
        producer=tl.where(spec[:,None],v,p)
    # The old restore materializes x.dtype before the RMS reads it. Preserve
    # that rounding even if either producer is FP32, avoiding an accidental
    # precision change disguised as layout fusion.
    x=producer.to(Y.dtype.element_ty).to(tl.float32)
    col_mask=cols<128
    w=tl.load(W+cols,mask=col_mask).to(tl.float32)
    z=tl.load(Z+(valid_token[:,None]*16+head[:,None])*128+cols[None,:],
        mask=row_mask[:,None]&col_mask[None,:]).to(tl.float32)
    xbar=tl.where(col_mask[None,:],x,0.0)
    var=tl.sum(xbar*xbar,axis=1)/128
    rstd=1.0/tl.sqrt(var+eps)
    valid_row=valid_token*16+head
    tl.store(Rstd+valid_row,rstd,mask=row_mask)
    x_hat=x*rstd[:,None]
    y=x_hat*w[None,:]
    y*=z*tl.sigmoid(z)
    tl.store(Y+valid_row[:,None]*128+cols[None,:],y,
        mask=row_mask[:,None]&col_mask[None,:])
