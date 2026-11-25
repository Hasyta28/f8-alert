"""Sổ lệnh ẢO (paper trading) cho F8 4h và 1h squeeze. Không đặt lệnh thật.
Không lưu trạng thái: mỗi lần chạy, mô phỏng lại toàn bộ từ ngày bắt đầu (PAPER_START) bằng đúng luật backtest,
nên kết quả luôn nhất quán và không cần cơ sở dữ liệu."""
import numpy as np, pandas as pd, json, urllib.request, datetime as dt
FEE,SLIP=0.0005,0.0002
ema=lambda x,n: pd.Series(x).ewm(span=n,adjust=False).mean().values
def rsi(c,n=14):
    s=pd.Series(c); d=s.diff(); u=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean(); dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean(); return (100-100/(1+u/dn)).values
def atr(h,l,c,n=14):
    pc=pd.Series(c).shift(); tr=np.maximum(h-l,np.maximum(abs(h-pc),abs(l-pc))); return pd.Series(tr).ewm(alpha=1/n,adjust=False).mean().values
MS={"1h":3600000,"4h":14400000}
def hist(sym,iv,start_ms):
    out=[]; t=int(start_ms)
    now=dt.datetime.now(dt.timezone.utc).timestamp()*1000
    while True:
        u=f"https://data-api.binance.vision/api/v3/klines?symbol={sym}&interval={iv}&startTime={t}&limit=1000"
        k=json.loads(urllib.request.urlopen(u,timeout=30).read())
        if not k: break
        out+=k; t=k[-1][0]+MS[iv]
        if len(k)<1000: break
    d=pd.DataFrame([x[:6]+[x[6]] for x in out],columns=["t","o","h","l","c","v","tc"]).astype(float).drop_duplicates("t")
    return d[d.tc<now].reset_index(drop=True)

# ---------------- F8 4h ----------------
def f8_ind(A,B,extra=None):
    I=dict(o=A.o.values,h=A.h.values,l=A.l.values,c=A.c.values,t=A.t.values,tc=A.tc.values)
    c=I["c"]; I["bc"]=B.set_index("t").reindex(A.t.values).ffill().c.values
    for k in (20,50,100,200,300): I[f"e{k}"]=ema(c,k)
    I["a"]=atr(I["h"],I["l"],c); I["R"]=rsi(c); bc=I["bc"]
    I["b20"],I["b50"],I["b200"]=ema(bc,20),ema(bc,50),ema(bc,200); I["rng"]=np.maximum(I["h"]-I["l"],1e-12)
    sg=np.sign(I["e50"]-I["e100"]); age=np.zeros(len(c),int)
    for k in range(1,len(c)): age[k]=age[k-1]+1 if sg[k]==sg[k-1] else 0
    I["age"]=age; I["ratio"]=c/bc; I["n"]=len(c); I["extra"]=extra   # extra(i,s)-> số điểm thêm (L/S, funding) nếu có dữ liệu
    return I
def f8_sig(I,i,s):
    o,h,l,c,a,e20,e50,e100,rng=(I[k] for k in ("o","h","l","c","a","e20","e50","e100","rng"))
    ok=(e50[i]>e100[i] and e100[i]>e100[i-10]) if s==1 else (e50[i]<e100[i] and e100[i]<e100[i-10])
    if not ok: return None
    if not ((s==1 and c[i]>e20[i] and c[i]>o[i]) or (s==-1 and c[i]<e20[i] and c[i]<o[i])): return None
    tj=[j for j in range(i-10,i) if (l[j]<=e50[j]+0.3*a[j] if s==1 else h[j]>=e50[j]-0.3*a[j])]
    rej=[j for j in tj if (((c[j]-l[j])/rng[j]>=0.5 and c[j]>e50[j]-0.5*a[j]) if s==1 else ((h[j]-c[j])/rng[j]>=0.5 and c[j]<e50[j]+0.5*a[j]))]
    if not (len(rej)>=2 and tj and i-tj[0]>=3): return None
    r=I["ratio"]
    if not s*(r[i]/r[i-42]-1)>0: return None
    if not ((l[i-10:i+1].min()>l[i-40:i-10].min()) if s==1 else (h[i-10:i+1].max()<h[i-40:i-10].max())): return None
    bc,b20,b50,b200,R,age,e300=(I[k] for k in ("bc","b20","b50","b200","R","age","e300"))
    btc_st=(b20[i]>b50[i] and bc[i]>b200[i]) if s==1 else (b20[i]<b50[i] and bc[i]<b200[i])
    rr=R[i] if s==1 else 100-R[i]
    sc=int(sum([age[i]<80, s*(bc[i]/bc[i-18]-1)>0, s*(c[i]-e50[i])/a[i]>1, rr>55, btc_st]))+(I["extra"](i,s) if I["extra"] else 0)
    late=int(sum([age[i]>=100, s*(e50[i]-e100[i])/a[i]>0.9, s*(c[i]-e300[i])/a[i]>5, s*(c[i]-e20[i])/a[i]<0.6]))
    if sc<=2 or late>=3: return None
    rk,addr=(0.04,0) if sc<=4 else (0.12,5)
    if late==2: rk,addr=rk/2,0
    elif late==0: rk*=1.5
    e200=I["e200"]; ep=o[i+1] if i+1<len(o) else c[i]; sp=(min(e200[i],l[i])-a[i]) if s==1 else (max(e200[i],h[i])+a[i])
    d=min(abs(ep-sp)/ep,0.12)
    return dict(sc=sc,late=late,rk=rk,addr=addr,d=d)
def engulf(o,c,j,s): return (s==1 and c[j]>o[j] and c[j]>=o[j-1] and o[j]<=c[j-1] and c[j-1]<o[j-1]) or (s==-1 and c[j]<o[j] and c[j]<=o[j-1] and o[j]>=c[j-1] and c[j-1]>o[j-1])
def f8_trade(I,i,s,addr):
    o,h,l,c,a,e20,e100,e200,bc,b20,R=(I[k] for k in ("o","h","l","c","a","e20","e100","e200","bc","b20","R")); n=I["n"]
    legs=[(o[i+1]*(1+SLIP*s),1.0)]; sp=(min(e200[i],l[i])-a[i]) if s==1 else (max(e200[i],h[i])+a[i])
    sp=max(sp,o[i+1]*(1-0.12)) if s==1 else min(sp,o[i+1]*(1+0.12)); sp0=sp
    hp=o[i+1]-s*1.5*abs(o[i+1]-sp); added=addr==0; trailing=False; best=None; xp=None; why=None; j=i+1; ev=[("entry",i+1,legs[0][0])]
    while j<n:
        if not added:
            lvl=e100[j-1]
            if (s==1 and l[j]<=lvl<legs[0][0] and lvl>sp) or (s==-1 and h[j]>=lvl>legs[0][0] and lvl<sp):
                px=(min(o[j],lvl) if s==1 else max(o[j],lvl))*(1+SLIP*s); legs.append((px,addr)); added=True; ev.append(("add",j,px))
        q=sum(x for _,x in legs); ap=sum(p*x for p,x in legs)/q; risk=s*(ap-sp); tpp=ap+s*risk
        if (s==1 and o[j]<=sp) or (s==-1 and o[j]>=sp): xp=o[j]; why="SL" if not trailing else "trail"
        elif trailing and ((s==1 and l[j]<=sp) or (s==-1 and h[j]>=sp)): xp=sp; why="trail"
        elif (s==1 and l[j]<=hp) or (s==-1 and h[j]>=hp): xp=hp; why="hard"
        elif (not trailing) and s*(c[j]-sp)<0: xp=c[j]; why="SL"
        elif not trailing and ((s==1 and h[j]>=tpp) or (s==-1 and l[j]<=tpp)):
            if s*(c[j]-e20[j])>0 and s*(bc[j]-b20[j])>0 and ((R[j]>60) if s==1 else (R[j]<40)):
                trailing=True; sp=ap+s*0.5*risk; best=h[j] if s==1 else l[j]; ev.append(("trail",j,sp))
            else: xp=tpp; why="TP"
        if xp is None and not trailing and s*(c[j]-ap)>=0.3*risk and (engulf(o,c,j,-s) or s*(c[j]-e20[j])<0): xp=c[j]; why="early"
        if xp is not None: break
        if trailing:
            best=max(best,h[j]) if s==1 else min(best,l[j]); ns=best-s*2.5*a[j]; sp=max(sp,ns) if s==1 else min(sp,ns)
        j+=1
    q=sum(x for _,x in legs)
    if xp is None: return dict(open=True,j=n-1,legs=legs,sp=sp,sp0=sp0,hp=hp,trailing=trailing,ev=ev,
                               ret=sum(x*s*(c[-1]-p)/p for p,x in legs)-q*2*FEE)
    xpp=xp*(1-SLIP*s); ret=sum(x*s*(xpp-p)/p for p,x in legs)-q*2*FEE; ev.append(("exit",j,xp))
    return dict(open=False,j=j,legs=legs,why=why,ret=ret,xp=xp,sp0=sp0,hp=hp,ev=ev)
def f8_book(I,start_i,eq0=1000.0):
    eq=eq0; T=[]; last=-1; i=max(start_i,300); n=I["n"]
    while i<n:
        for s in (1,-1):
            g=f8_sig(I,i,s)
            if g: break
        if not g or i<=last: i+=1; continue
        tot=min(5,g["rk"]/g["d"]); bf=tot/(1+g["addr"])
        if i==n-1:   # tín hiệu ở nến vừa đóng: lệnh chờ vào ở giá mở nến kế tiếp
            c=I["c"][i]; sp=(min(I["e200"][i],I["l"][i])-I["a"][i]) if s==1 else (max(I["e200"][i],I["h"][i])+I["a"][i])
            sp=max(sp,c*0.88) if s==1 else min(sp,c*1.12)
            T.append(dict(i=i,s=s,sig=g,bf=bf,tot=tot,eq0=eq,pnl=0.0,open=True,pending=True,j=i,legs=[(c,1.0)],sp=sp,sp0=sp,ret=0.0,ev=[],trailing=False)); break
        r=f8_trade(I,i,s,g["addr"])
        eqb=eq; pnl=bf*r["ret"]*eq
        T.append(dict(i=i,s=s,sig=g,bf=bf,tot=tot,eq0=eqb,pnl=pnl,**r))
        if r["open"]: break
        eq+=pnl; last=r["j"]; i=r["j"]+1
    return T,eq

# ---------------- 1h squeeze ----------------
def h1_book(H,A4,start_i,vol=200.0,hl=False):
    o,h,l,c=(H[x].values for x in "ohlc"); a=atr(h,l,c); n=len(c)
    e20,e50,e100,e200=(ema(c,k) for k in (20,50,100,200)); EM=np.vstack([e20,e50,e100,e200])
    hi,lo=EM.max(0),EM.min(0); w=(hi-lo)/a; nar=pd.Series(w<1.5).rolling(30).max().values
    up=(e20>e50)&(e50>e100)&(e100>e200); dn=(e20<e50)&(e50<e100)&(e100<e200)
    c4=A4.c.values; f50,f100=ema(c4,50),ema(c4,100); sl10=pd.Series(f100).diff(10).values
    tr4=np.where((f50>f100)&(sl10>0),1,np.where((f50<f100)&(sl10<0),-1,0))
    sg=np.sign(f50-f100); age=np.zeros(len(c4),int)
    for k in range(1,len(c4)): age[k]=age[k-1]+1 if sg[k]==sg[k-1] else 0
    idx=np.searchsorted(A4.tc.values,H.tc.values,side="right")-1
    htr=np.where(idx>=0,tr4[np.maximum(idx,0)],0); hage=np.where(idx>=0,age[np.maximum(idx,0)],999)
    T=[]; i=max(start_i,300); tot=0.0
    while i<n:
        d=1 if (up[i] and not up[i-1]) else -1 if (dn[i] and not dn[i-1]) else 0
        if d==0 or nar[i]!=1 or htr[i]!=d or hage[i]>=100: i+=1; continue
        if hl and not ((l[i-15:i+1].min()>l[i-30:i-15].min()) if d==1 else (h[i-15:i+1].max()>h[i-30:i-15].max())): i+=1; continue
        sp=(lo[i]-a[i]) if d==1 else (hi[i]+a[i])
        if i==n-1:   # tín hiệu ở nến vừa đóng: vào ở giá mở nến kế tiếp (≈ giá đóng hiện tại)
            if d*(c[i]-sp)>0: T.append(dict(i=i,s=d,ep=c[i],sp=sp,open=True,pending=True,j=i,usd=0.0))
            break
        ep=o[i+1]*(1+SLIP*d)
        if d*(ep-sp)<=0: i+=1; continue
        j=i+1; xp=None; why=None
        while j<n:
            if (d==1 and l[j]<=sp) or (d==-1 and h[j]>=sp): xp=o[j] if d*(o[j]-sp)<0 else sp; why="SL"; break
            if (e20[j]<e50[j]) if d==1 else (e20[j]>e50[j]): xp=c[j]; why="cross"; break
            j+=1
        if xp is None:
            T.append(dict(i=i,s=d,ep=ep,sp=sp,open=True,j=n-1,usd=vol*(d*(c[-1]-ep)/ep-2*FEE))); break
        usd=vol*(d*(xp-ep)/ep-2*FEE-SLIP); tot+=usd
        T.append(dict(i=i,s=d,ep=ep,sp=sp,xp=xp,why=why,open=False,j=j,usd=usd)); i=j
    return T,tot
