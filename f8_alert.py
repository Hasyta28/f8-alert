#!/usr/bin/env python3
"""F8 – AAVE 4h: kiểm tra kèo mỗi khi nến 4h đóng và báo qua Telegram.
Chỉ đọc giá công khai (data-api.binance.vision, giá spot AAVEUSDT/BTCUSDT). Không API key, không đặt lệnh.
Biến môi trường: TELEGRAM_TOKEN, TELEGRAM_CHAT_ID (mặc định), CHAT_4H / CHAT_1H / CHAT_PAPER (nhóm riêng, tùy chọn), VON (vốn USD, mặc định 1000), STATUS=1 để luôn gửi trạng thái."""
import os, json, urllib.request, urllib.parse, datetime as dt
import numpy as np, pandas as pd
ema=lambda x,n: pd.Series(x).ewm(span=n,adjust=False).mean().values
def rsi(c,n=14):
    s=pd.Series(c); d=s.diff(); u=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean(); dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean(); return (100-100/(1+u/dn)).values
def atr(h,l,c,n=14):
    pc=pd.Series(c).shift(); tr=np.maximum(h-l,np.maximum(abs(h-pc),abs(l-pc))); return pd.Series(tr).ewm(alpha=1/n,adjust=False).mean().values
def klines(sym,iv="4h"):
    u=f"https://data-api.binance.vision/api/v3/klines?symbol={sym}&interval={iv}&limit=1000"
    k=json.loads(urllib.request.urlopen(u,timeout=30).read())
    d=pd.DataFrame([x[:6]+[x[6]] for x in k],columns=["t","o","h","l","c","v","tc"]).astype(float)
    now=dt.datetime.now(dt.timezone.utc).timestamp()*1000
    return d[d.tc<now].reset_index(drop=True)   # bỏ nến chưa đóng
# Thống kê lịch sử F8 2022-11/2026 theo nhóm (điểm, số dấu hiệu muộn): (số lệnh, % thắng, lãi TB %vốn, lệnh tệ nhất %vốn)
HIST={("3-4",0):(17,88,6.7,-8.2),("3-4",1):(24,71,0.5,-6.1),("3-4",2):(22,77,0.4,-3.1),
      ("5-7",0):(26,96,2.8,-14.5),("5-7",1):(10,90,2.0,-8.3),("5-7",2):(18,78,2.1,-6.8)}
def evaluate(A,B,von=1000.0):
    o,h,l,c=(A[x].values for x in "ohlc"); i=len(c)-1
    bc=B.set_index("t").reindex(A.t.values).ffill().c.values
    e20,e50,e100,e200,e300=(ema(c,k) for k in (20,50,100,200,300)); a=atr(h,l,c); R=rsi(c)
    b20,b50,b200=ema(bc,20),ema(bc,50),ema(bc,200); rng=np.maximum(h-l,1e-12)
    sg=np.sign(e50-e100); age=np.zeros(len(c),int)
    for k in range(1,len(c)): age[k]=age[k-1]+1 if sg[k]==sg[k-1] else 0
    ratio=c/bc; out=[]
    for s in (1,-1):
        info=dict(side="LONG" if s==1 else "SHORT")
        trend=(e50[i]>e100[i] and e100[i]>e100[i-10]) if s==1 else (e50[i]<e100[i] and e100[i]<e100[i-10])
        candle=(s==1 and c[i]>e20[i] and c[i]>o[i]) or (s==-1 and c[i]<e20[i] and c[i]<o[i])
        tj=[j for j in range(i-10,i) if (l[j]<=e50[j]+0.3*a[j] if s==1 else h[j]>=e50[j]-0.3*a[j])]
        rej=[j for j in tj if (((c[j]-l[j])/rng[j]>=0.5 and c[j]>e50[j]-0.5*a[j]) if s==1 else ((h[j]-c[j])/rng[j]>=0.5 and c[j]<e50[j]+0.5*a[j]))]
        sig=trend and candle and len(rej)>=2 and bool(tj) and i-tj[0]>=3
        f1=s*(ratio[i]/ratio[i-42]-1)>0
        f2=(l[i-10:i+1].min()>l[i-40:i-10].min()) if s==1 else (h[i-10:i+1].max()<h[i-40:i-10].max())
        r=R[i] if s==1 else 100-R[i]
        btc_st=(b20[i]>b50[i] and bc[i]>b200[i]) if s==1 else (b20[i]<b50[i] and bc[i]<b200[i])
        auto=[age[i]<80, s*(bc[i]/bc[i-18]-1)>0, s*(c[i]-e50[i])/a[i]>1, r>55, btc_st]
        late=[age[i]>=100, s*(e50[i]-e100[i])/a[i]>0.9, s*(c[i]-e300[i])/a[i]>5, s*(c[i]-e20[i])/a[i]<0.6]
        sc=int(sum(auto)); nl=int(sum(late))
        ep=c[i]; sp=(min(e200[i],l[i])-a[i]) if s==1 else (max(e200[i],h[i])+a[i]); d=min(abs(ep-sp)/ep,0.12); sp=ep*(1-s*d)
        info.update(trend=trend,sig=sig,f1=f1,f2=f2,auto=sc,late=nl,age=int(age[i]),ep=ep,sp=sp,hard=ep*(1-s*1.5*d),tp=ep*(1+s*d),d=d,e100=e100[i],
                    e50=e50[i],e20=e20[i],atr=a[i])
        plans=[]
        for scx in (sc,sc+1,sc+2):     # +1/+2 = nếu 1 hoặc 2 ý tự kiểm tra (Long/Short ratio <1.5, funding <0.04%) đạt
            if scx<=2 or nl>=3: plans.append((scx,None)); continue
            rk,add=(0.04,0) if scx<=4 else (0.12,5)
            if nl==2: rk,add=rk/2,0
            elif nl==0: rk*=1.5
            tot=min(5,rk/d)*von; plans.append((scx,dict(rk=rk,add=add,tot=tot,leg1=tot/(1+add),leg2=tot*add/(1+add),hist=HIST.get(("3-4" if scx<=4 else "5-7",nl)))))
        info["plans"]=plans; info["ok"]=sig and f1 and f2 and nl<3; out.append(info)
    return out, A.tc.iloc[-1]

# ===================== TRÌNH BÀY TIN NHẮN =====================
import io, html
VN=pd.Timedelta(hours=7)
def fmt(x):
    if abs(x)>=100 or x==0: return f"{x:,.2f}"
    import math; return f"{x:,.{max(2,4-int(math.floor(math.log10(abs(x)))))}f}"
WIN=[0,0]   # (w0,w1]: chỉ báo các nến ĐÓNG trong khoảng này (từ lần chạy thành công trước đến giờ) -> không gửi trùng
def inw(close_ms): return WIN[0]<close_ms<=WIN[1]
def set_window(now):
    w0=now-15*60000
    try:
        rep=os.environ.get("GITHUB_REPOSITORY"); tok=os.environ.get("GITHUB_TOKEN")
        if rep and tok:
            u=f"https://api.github.com/repos/{rep}/actions/workflows/f8.yml/runs?status=success&per_page=1"
            r=json.loads(urllib.request.urlopen(urllib.request.Request(u,headers={"Authorization":"Bearer "+tok,"Accept":"application/vnd.github+json"}),timeout=20).read())
            if r.get("workflow_runs"):
                st=r["workflow_runs"][0].get("run_started_at") or r["workflow_runs"][0]["created_at"]
                w0=pd.Timestamp(st).value//10**6
    except Exception as e: print("không đọc được lần chạy trước:",e)
    WIN[0]=max(w0,now-6*3600000); WIN[1]=now
def tvn(ms): return (pd.to_datetime(ms,unit="ms").ceil("min")+VN).strftime("%d/%m %H:%M")
SEP="━━━━━━━━━━━━━━━━━━"
def chart(df,title,levels,mark=None,n=90,zone=None):
    """Ảnh nến + EMA + các mức giá. levels: list (giá, nhãn, màu, kiểu)."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    d=df.iloc[-n:].reset_index(drop=True); c=df.c.values
    E={k:ema(c,k)[-n:] for k in (20,50,100,200)}
    bg,fg,grid="#0f1419","#d0d4da","#232a33"
    fig,ax=plt.subplots(figsize=(10,5.6),dpi=110); fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    x=np.arange(len(d)); up=d.c>=d.o
    ax.vlines(x,d.l,d.h,color=np.where(up,"#26a69a","#ef5350"),lw=1)
    ax.bar(x,(d.c-d.o).abs().clip(lower=(d.h-d.l).max()*0.004),bottom=np.minimum(d.o,d.c),color=np.where(up,"#26a69a","#ef5350"),width=0.7)
    for k,col in ((20,"#f5c542"),(50,"#42a5f5"),(100,"#ab47bc"),(200,"#9e9e9e")): ax.plot(x,E[k],color=col,lw=1.3,label=f"EMA{k}")
    N=len(d); x0=N+1; x1=N+16
    if zone: ax.axhspan(zone[0],zone[1],color="#42a5f5",alpha=0.12); ax.text(N-1,zone[1],(zone[2] if len(zone)>2 else "")+"  ",color="#90caf9",fontsize=9,va="bottom",ha="right")
    lv={lab:p for p,lab,*_ in levels}
    if "Entry" in lv and "SL" in lv:   # hộp rủi ro / mục tiêu kiểu TradingView
        ax.fill_between([x0,x1],lv["Entry"],lv["SL"],color="#ef5350",alpha=0.18,lw=0)
        tgt=lv.get("1R")
        if tgt is not None: ax.fill_between([x0,x1],lv["Entry"],tgt,color="#26a69a",alpha=0.18,lw=0)
    for p,lab,col,ls in levels:
        ax.hlines(p,N-20,x1,colors=col,linestyles=ls,lw=1.4)
        ax.text(x1+0.6,p,f"{lab}  {fmt(p)}",color=col,fontsize=9.5,va="center",fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25",fc=bg,ec=col,lw=0.8))
    if mark:
        side_up=lv.get("SL",d.c.iloc[-1])<d.c.iloc[-1]
        yy=d.l.iloc[-1]-(d.h.max()-d.l.min())*0.03 if side_up else d.h.iloc[-1]+(d.h.max()-d.l.min())*0.03
        ax.scatter([N-1],[yy],marker="^" if side_up else "v",s=110,color="#ffffff",zorder=5)
        ax.text(N-1,yy,("\n\n" if side_up else "")+"tín hiệu"+("" if side_up else "\n\n"),color="#ffffff",fontsize=8.5,ha="center",va="top" if side_up else "bottom")
    ax.set_xlim(-1,x1+17)
    lo=min([d.l.min()]+[p for p,*_ in levels]); hi=max([d.h.max()]+[p for p,*_ in levels])
    ax.set_ylim(lo-(hi-lo)*0.06,hi+(hi-lo)*0.08)
    ticks=list(range(0,N,max(1,N//6))); ax.set_xticks(ticks); ax.set_xticklabels([(pd.to_datetime(d.t.iloc[k],unit="ms")+VN).strftime("%d/%m %Hh") for k in ticks],color=fg,fontsize=8)
    ax.tick_params(colors=fg,labelsize=8); ax.ticklabel_format(axis="y",style="plain",useOffset=False)
    for sp_ in ax.spines.values(): sp_.set_color(grid)
    ax.grid(color=grid,lw=0.6); ax.legend(loc="upper left",fontsize=8,facecolor=bg,edgecolor=grid,labelcolor=fg,ncol=4)
    fig.text(0.99,0.015,"F8 · tín hiệu tham khảo · không phải lời khuyên đầu tư",color="#5c6670",fontsize=8,ha="right")
    ax.set_title(title,color=fg,fontsize=12,loc="left",fontweight="bold")
    fig.tight_layout(); b=io.BytesIO(); fig.savefig(b,format="png",facecolor=bg); plt.close(fig); return b.getvalue()

def f8_posts(res,A4,von,status):
    tc=A4.tc.iloc[-1]; t=tvn(tc); posts=[]
    for r in res:
        if not r["ok"]: continue
        L=r["side"]=="LONG"; icon="🟢" if L else "🔴"; dirw="LONG" if L else "SHORT"
        cap=(f"{icon} <b>KÈO {dirw} · AAVE/USDT · 4H</b>  <i>(F8)</i>\n🕒 Nến đóng {t} (giờ VN)\n{SEP}\n"
             f"📍 <b>Entry</b>      <code>{fmt(r['ep'])}</code>  <i>(giá mở nến kế tiếp)</i>\n"
             f"🛑 <b>SL</b>           <code>{fmt(r['sp'])}</code>  <i>(cách {r['d']*100:.1f}%, cắt khi nến 4H đóng qua)</i>\n"
             f"⛔ <b>SL cứng</b>   <code>{fmt(r['hard'])}</code>  <i>(đặt sẵn trên sàn)</i>\n"
             f"🎯 <b>Mốc 1R</b>   <code>{fmt(r['tp'])}</code>\n{SEP}\n"
             f"📊 Điểm tự động <b>{r['auto']}/5</b> · Cuối sóng <b>{r['late']}/4</b> · Tuổi xu hướng {r['age']} nến")
        lv=[(r['tp'],"1R","#66bb6a","-"),(r['ep'],"Entry","#ffffff","-"),(r['sp'],"SL","#ef5350","--"),(r['hard'],"SL cứng","#b71c1c",":")]
        if any(p and p["add"] for _,p in r["plans"]): lv.append((r['e100'],"Nhồi (EMA100)","#ffa726","-."))
        img=chart(A4,f"AAVE/USDT 4H · {dirw} · {t}",lv,mark=True)
        rows=[]; seen={}; seen_s={}
        for scx,p in r["plans"]:
            key=None if p is None else (p["rk"],p["add"])
            if key in seen: rows[seen[key]]=rows[seen[key]].replace(f"Tổng điểm {seen_s[key]}",f"Tổng điểm {seen_s[key]}–{scx}",1); continue
            seen[key]=len(rows); seen_s[key]=scx
            if p is None: rows.append(f"• Tổng điểm {scx}: <b>BỎ QUA</b>"); continue
            hs=p["hist"]; add=f"\n   ↳ chân 1 <code>{p['leg1']:,.0f}$</code> + treo nhồi <code>{p['leg2']:,.0f}$</code> tại EMA100 <code>{fmt(r['e100'])}</code>" if p["add"] else ""
            hstr=f"\n   ↳ lịch sử: {hs[0]} lệnh · thắng {hs[1]}% · TB {hs[2]:+.1f}% vốn · tệ nhất {hs[3]}%" if hs else ""
            rows.append(f"• Tổng điểm {scx}: rủi ro <b>{p['rk']*100:.0f}%</b> vốn → vol <b>{p['tot']:,.0f}$</b>{add}{hstr}")
        txt=(f"💼 <b>PHƯƠNG ÁN VOL</b> (vốn {von:,.0f}$)\n"+"\n".join(rows)+
             f"\n<i>Tự kiểm 2 ý trên Binance: Long/Short ratio &lt; 1,5 và funding TB thị trường &lt; 0,04% — đạt mỗi ý +1 điểm.</i>\n{SEP}\n"
             "🧭 <b>QUẢN LÝ LỆNH</b>\n• Tại 1R: nếu giá đóng cùng phía EMA20, BTC cùng phía EMA20 BTC và RSI &gt; 60 (short &lt; 40) → dời SL lên +0,5R, gồng 2,5 ATR. Không đủ → chốt hết.\n"
             "• Chốt sớm: lãi ≥ 0,3R mà xuất hiện nến nhấn chìm ngược hoặc nến đóng ngược qua EMA20.\n\n<i>Tín hiệu tham khảo, bạn tự quyết định.</i>")
        posts.append(dict(photo=img,caption=cap,text=txt))
    if not posts and status:
        r=[x for x in res if x["trend"]]
        if r:
            r=r[0]; L=r["side"]=="LONG"; z=r["e50"]; a_=r["atr"]
            cap=(f"📋 <b>TRẠNG THÁI AAVE · 4H</b>\n🕒 {t} (giờ VN)\n{SEP}\n"
                 f"Xu hướng: <b>{'TĂNG' if L else 'GIẢM'}</b> · tuổi {r['age']} nến\nGiá <code>{fmt(r['ep'])}</code> · EMA20 <code>{fmt(r['e20'])}</code> · EMA50 <code>{fmt(z)}</code>\n"
                 f"Cuối sóng: <b>{r['late']}/4</b> {'→ có tín hiệu cũng BỎ QUA' if r['late']>=3 else ''}\n"
                 f"Vùng chờ pullback EMA50: <code>{fmt(z-0.3*a_)} – {fmt(z+0.3*a_)}</code>\n⏳ Chưa có kèo 4H.")
            img=chart(A4,f"AAVE/USDT 4H · trạng thái · {t}",[(z,"EMA50","#42a5f5","--")],zone=(z-0.3*a_,z+0.3*a_,"vùng chờ"))
            posts.append(dict(photo=img,caption=cap,text=None))
        else: posts.append(dict(photo=None,caption=None,text=f"📋 <b>TRẠNG THÁI AAVE · 4H</b> · {t}\nKhông có xu hướng rõ (EMA50/EMA100 chưa đồng thuận). ⏳ Chưa có kèo 4H."))
    return posts
# ===== Phương pháp 1h (squeeze EMA, lọc G3) =====
# 4 EMA 20/50/100/200 khung 1h co cụm (< 1,5 ATR) trong 30 nến gần nhất, rồi vừa xếp thẳng hàng cùng chiều xu hướng 4h
# (EMA50>EMA100 và EMA100 dốc lên 10 nến, ngược lại cho short), xu hướng 4h dưới 100 nến. Vào ở giá mở nến 1h kế tiếp.
# SL = mép dải EMA ∓ 1 ATR(1h). Thoát khi EMA20 cắt ngược EMA50 (nến 1h đóng). Vol cố định (mặc định 200$).
H1_HIST=dict(n=101,win=42,IS=(63,35),OOS=(38,53),avg_win=16.6,avg_loss=-6.7,worst=-17.9,total=301)
def h1_eval(H,A4):
    o,h,l,c=(H[x].values for x in "ohlc"); a=atr(h,l,c); n=len(c)
    e20,e50,e100,e200=(ema(c,k) for k in (20,50,100,200)); EM=np.vstack([e20,e50,e100,e200])
    hi,lo=EM.max(0),EM.min(0); w=(hi-lo)/a; nar=pd.Series(w<1.5).rolling(30).max().values
    up=(e20>e50)&(e50>e100)&(e100>e200); dn=(e20<e50)&(e50<e100)&(e100<e200)
    c4=A4.c.values; f50,f100=ema(c4,50),ema(c4,100); sl10=pd.Series(f100).diff(10).values
    tr4=np.where((f50>f100)&(sl10>0),1,np.where((f50<f100)&(sl10<0),-1,0))
    sg=np.sign(f50-f100); age=np.zeros(len(c4),int)
    for k in range(1,len(c4)): age[k]=age[k-1]+1 if sg[k]==sg[k-1] else 0
    idx=np.searchsorted(A4.tc.values,H.tc.values,side="right")-1   # nến 4h đã đóng gần nhất tại lúc nến 1h đóng
    htr=np.where(idx>=0,tr4[np.maximum(idx,0)],0); hage=np.where(idx>=0,age[np.maximum(idx,0)],999)
    def entry(i):
        d=1 if (up[i] and not up[i-1]) else -1 if (dn[i] and not dn[i-1]) else 0
        if d==0 or nar[i]!=1 or htr[i]!=d or hage[i]>=100: return 0
        sp=(lo[i]-a[i]) if d==1 else (hi[i]+a[i])
        return d if d*(c[i]-sp)>0 else 0
    return dict(entry=entry,n=n,o=o,h=h,l=l,c=c,a=a,lo=lo,hi=hi,e20=e20,e50=e50,hage=hage,w=w,tc=H.tc.values)

def h1_posts(H,A4,vol,status,lab="1H",hs=None,tlab="4H",warn=""):
    R=h1_eval(H,A4); i=R["n"]-1; posts=[]; hs=hs or H1_HIST; t=tvn(R["tc"][i])
    d=R["entry"](i)
    if d:
        L=d==1; sp=(R["lo"][i]-R["a"][i]) if L else (R["hi"][i]+R["a"][i]); ep=R["c"][i]; rp=abs(ep-sp)/ep
        cap=(f"{'🟢' if L else '🔴'} <b>KÈO {'LONG' if L else 'SHORT'} · AAVE/USDT · {lab}</b>  <i>(squeeze EMA)</i>\n🕒 Nến đóng {t} (giờ VN)\n{SEP}\n"
             f"📍 <b>Entry</b>   <code>{fmt(ep)}</code>  <i>(giá mở nến kế tiếp)</i>\n🛑 <b>SL</b>        <code>{fmt(sp)}</code>  <i>(cách {rp*100:.1f}%)</i>\n"
             f"🚪 <b>Thoát</b>   khi EMA20 cắt ngược EMA50 (tool sẽ nhắn)\n💼 <b>Vol</b>       <code>{vol:,.0f}$</code> → lỗ tối đa ~<code>{vol*rp:,.1f}$</code>\n{SEP}\n"
             f"📈 Lịch sử: {hs['n']} lệnh · thắng {hs['win']}% (2025-26: {hs['OOS'][1]}%) · thắng TB +{hs['avg_win']}$ · thua TB {hs['avg_loss']}$\n"
             f"<i>Thắng ít nhưng ăn dày: nhiều lệnh thua nhỏ là bình thường. Bạn tự quyết định.</i>"+warn)
        img=chart(H,f"AAVE/USDT {lab} · {'LONG' if L else 'SHORT'} · {t}",[(ep+(ep-sp),"1R","#66bb6a",":"),(ep,"Entry","#ffffff","-"),(sp,"SL","#ef5350","--")],mark=True,n=120)
        posts.append(dict(photo=img,caption=cap,text=None))
    k=i-1; start=None
    while k>max(300,i-600):
        if R["entry"](k): start=k; break
        k-=1
    if start is not None and not d:
        dd=R["entry"](start); sp=(R["lo"][start]-R["a"][start]) if dd==1 else (R["hi"][start]+R["a"][start]); ep=R["o"][start+1]
        for j in range(start+1,i+1):
            hit=(dd==1 and R["l"][j]<=sp) or (dd==-1 and R["h"][j]>=sp)
            cross=(R["e20"][j]<R["e50"][j]) if dd==1 else (R["e20"][j]>R["e50"][j])
            if hit or cross:
                if j==i:
                    xp=sp if hit else R["c"][j]; pct=dd*(xp-ep)/ep; pnl=pct*vol
                    img=chart(H,f"AAVE/USDT {lab} · thoát lệnh · {t}",[(ep,"Entry","#ffffff","-"),(xp,"Thoát","#66bb6a" if pnl>0 else "#ef5350","--")],n=120)
                    posts.append(dict(photo=img,text=None,caption=(f"🚪 <b>THOÁT LỆNH {'LONG' if dd==1 else 'SHORT'} · AAVE/USDT · {lab}</b>\n🕒 Nến đóng {t} (giờ VN)\n{SEP}\n"
                        f"Lý do: <b>{'chạm SL' if hit else 'EMA20 cắt ngược EMA50'}</b>\nVào <code>{fmt(ep)}</code> ({tvn(R['tc'][start])}) → thoát ~<code>{fmt(xp)}</code>\n"
                        f"Kết quả: <b>{pct*100:+.2f}%</b> ≈ <b>{pnl:+,.1f}$</b> với vol {vol:,.0f}$ <i>(chưa trừ phí)</i>")))
                break
    if status and not posts:
        hi_,lo_=R["hi"][i],R["lo"][i]
        img=chart(H,f"AAVE/USDT {lab} · trạng thái · {t}",[(hi_,"Đỉnh dải EMA","#ffb74d","--"),(lo_,"Đáy dải EMA","#ffb74d","--")],n=120,zone=(lo_,hi_))
        posts.append(dict(photo=img,caption=f"📋 <b>TRẠNG THÁI AAVE · {lab}</b> · {t}\nDải 4 EMA rộng <b>{R['w'][i]:.1f} ATR</b> (cần &lt; 1,5 trong 30 nến gần nhất) · xu hướng {tlab} tuổi {int(R['hage'][i])} nến.\nKèo {lab} xuất hiện khi dải EMA (vùng tô) bó hẹp lại rồi 4 EMA xếp tầng cùng chiều xu hướng {tlab}.\n⏳ Chưa có kèo {lab}.",text=None))
    return posts


M15_HIST=dict(n=383,win=27,IS=(0,0),OOS=(0,27),avg_win=8.0,avg_loss=-3.5,worst=0,total=-178)
M15_WARN="\n⚠️ <b>Lưu ý:</b> kiểu này ở khung m15 backtest 2022–10/2026 <b>lỗ −178$</b> (383 lệnh, vol 200$) do phí ăn mòn. Chỉ để tham khảo/quan sát."
# ===================== SỔ LỆNH ẢO (paper trading) =====================
def event_posts(head,lines,G,maxphoto=8):
    """Mỗi sự kiện có dữ liệu biểu đồ -> 1 tin ảnh; quá nhiều thì phần còn lại gộp thành 1 tin chữ."""
    posts=[]; rest=[]
    for k,ln in enumerate(lines):
        if k in G and len(posts)<maxphoto:
            try:
                df,title,lv=G[k]; posts.append(dict(photo=chart(df,title,lv,n=120),caption=head+"\n"+ln,text=None)); continue
            except Exception as e: print("chart lỗi:",e)
        rest.append(ln)
    if rest: posts.append(dict(photo=None,caption=None,text=head+"\n"+"\n".join(rest)))
    return posts
def paper_posts(status):
    import paper as P
    start=pd.Timestamp(os.environ.get("PAPER_START","2026-11-23"),tz="UTC").value//10**6
    eq0=float(os.environ.get("VON","1000")); vol=float(os.environ.get("VOL_1H","200"))
    A=P.hist("AAVEUSDT","4h",start-1100*P.MS["4h"]); B=P.hist("BTCUSDT","4h",start-1100*P.MS["4h"])
    H=P.hist("AAVEUSDT","1h",start-1100*P.MS["1h"])
    I=P.f8_ind(A,B); s4=int(np.searchsorted(A.t.values,start)); T4,eq=P.f8_book(I,s4,eq0)
    s1=int(np.searchsorted(H.t.values,start)); T1,tot1=P.h1_book(H,A,s1,vol)
    lines=[]; G={}
    side=lambda s:"LONG" if s==1 else "SHORT"
    c4=lambda k: inw(A.tc.values[k]+1); c1=lambda k: inw(H.tc.values[k]+1)
    if True:
        for x in T4:
            if c4(x["i"]):
                G[len(lines)]=(A,f"Lệnh ảo AAVE 4H · {side(x['s'])}",[(x['legs'][0][0],"Entry","#ffffff","-"),(x['sp0'],"SL","#ef5350","--")])
                lines.append(f"📥 <b>4H · vào lệnh ảo {side(x['s'])}</b> tại giá mở nến kế tiếp · vol chân 1 <code>{x['bf']*x['eq0']:,.0f}$</code>"
                             +(f" · treo nhồi <code>{x['bf']*x['sig']['addr']*x['eq0']:,.0f}$</code> ở EMA100" if x['sig']['addr'] else "")+f" · SL <code>{fmt(x['sp0'])}</code>")
            for kind,j,px in x["ev"]:
                if not c4(j) or kind=="entry": continue
                G[len(lines)]=(A,f"Lệnh ảo AAVE 4H · {side(x['s'])}",[(x['legs'][0][0],"Entry","#ffffff","-"),(px,{"add":"Nhồi","trail":"SL mới","exit":"Thoát"}[kind],"#42a5f5" if kind=="add" else "#66bb6a" if kind=="trail" else ("#66bb6a" if x.get("pnl",0)>0 else "#ef5350"),"--")])
                if kind=="add": lines.append(f"➕ <b>4H · khớp lệnh nhồi</b> {side(x['s'])} tại <code>{fmt(px)}</code>")
                if kind=="trail": lines.append(f"🛡 <b>4H · đạt 1R, lệnh khỏe → gồng</b>: dời SL lên <code>{fmt(px)}</code> (+0,5R), kéo theo 2,5 ATR")
                if kind=="exit":
                    why={"trail":"chạm SL kéo theo","early":"chốt sớm (nến đảo chiều/đóng qua EMA20)","TP":"chốt 1R","SL":"cắt lỗ (nến đóng qua SL)","hard":"chạm SL cứng"}[x["why"]]
                    lines.append(f"📤 <b>4H · đóng lệnh ảo {side(x['s'])}</b> tại <code>{fmt(px)}</code> · {why} · <b>{x['pnl']:+,.2f}$</b> ({x['pnl']/x['eq0']*100:+.2f}% vốn)")
    for x in T1:
        if c1(x["i"]): G[len(lines)]=(H,f"Lệnh ảo AAVE 1H · {side(x['s'])}",[(x['ep'],"Entry","#ffffff","-"),(x['sp'],"SL","#ef5350","--")]); lines.append(f"📥 <b>1H · vào lệnh ảo {side(x['s'])}</b> vol <code>{vol:,.0f}$</code> · SL <code>{fmt(x['sp'])}</code>")
        if not x["open"] and c1(x["j"]): G[len(lines)]=(H,f"Lệnh ảo AAVE 1H · đóng {side(x['s'])}",[(x['ep'],"Entry","#ffffff","-"),(x['xp'],"Thoát","#66bb6a" if x['usd']>0 else "#ef5350","--")]); lines.append(f"📤 <b>1H · đóng lệnh ảo {side(x['s'])}</b> tại <code>{fmt(x['xp'])}</code> · {'chạm SL' if x['why']=='SL' else 'EMA20 cắt ngược EMA50'} · <b>{x['usd']:+,.2f}$</b>")
    posts=event_posts("📒 <b>SỔ LỆNH ẢO</b>",lines,G)
    if status:
        c4=[x for x in T4 if not x["open"]]; c1=[x for x in T1 if not x["open"]]
        o4=[x for x in T4 if x["open"]]; o1=[x for x in T1 if x["open"]]
        eq_open=eq+sum(x["bf"]*x["ret"]*eq for x in o4)
        txt=(f"📒 <b>SỔ LỆNH ẢO</b> · từ {(pd.to_datetime(start,unit='ms')+VN).strftime('%d/%m/%Y')}\n{SEP}\n"
             f"<b>4H F8</b> · vốn {eq0:,.0f}$ → <b>{eq:,.2f}$</b> ({(eq/eq0-1)*100:+.2f}%)\n"
             f"Đã đóng {len(c4)} lệnh · thắng {sum(x['pnl']>0 for x in c4)}"
             +(f"\nĐang mở: {side(o4[0]['s'])} từ <code>{fmt(o4[0]['legs'][0][0])}</code>, SL <code>{fmt(o4[0]['sp'])}</code>, tạm tính {eq_open-eq:+,.2f}$" if o4 else "\nKhông có lệnh mở")+"\n"
             f"<b>1H squeeze</b> · vol {vol:,.0f}$/lệnh · tổng <b>{tot1:+,.2f}$</b>\nĐã đóng {len(c1)} lệnh · thắng {sum(x['usd']>0 for x in c1)}"
             +(f"\nĐang mở: {side(o1[0]['s'])} từ <code>{fmt(o1[0]['ep'])}</code>, SL <code>{fmt(o1[0]['sp'])}</code>, tạm tính {o1[0]['usd']:+,.2f}$" if o1 else "\nKhông có lệnh mở")
             +"\n<i>Lệnh ảo theo đúng luật backtest, giá spot, có phí 0,05%/chiều, chưa tính funding. Điểm 4H chỉ dùng 5 tiêu chí tự động (thận trọng hơn).</i>")
        posts.append(dict(photo=None,caption=None,text=txt))
    return posts

def tg(method,data,files=None):
    tok=os.environ.get("TELEGRAM_TOKEN"); url=f"https://api.telegram.org/bot{tok}/{method}"
    if files:
        bd="----f8bound"; body=b""
        for k,v in data.items(): body+=f"--{bd}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
        for k,(fn,b) in files.items(): body+=f"--{bd}\r\nContent-Disposition: form-data; name=\"{k}\"; filename=\"{fn}\"\r\nContent-Type: image/png\r\n\r\n".encode()+b+b"\r\n"
        body+=f"--{bd}--\r\n".encode(); req=urllib.request.Request(url,data=body,headers={"Content-Type":f"multipart/form-data; boundary={bd}"})
    else: req=urllib.request.Request(url,data=urllib.parse.urlencode(data).encode())
    return json.loads(urllib.request.urlopen(req,timeout=60).read())
CH_ENV={"4h":"CHAT_4H","1h":"CHAT_1H","15m":"CHAT_15M","paper":"CHAT_PAPER","multi":"CHAT_PAPER"}
def chat_for(ch):   # mỗi loại tin một nhóm riêng; chưa đặt thì gửi về chat mặc định (m15 chưa có nhóm thì vào nhóm 1h)
    v=(os.environ.get(CH_ENV.get(ch,""),"") or "").strip()
    if not v and ch=="15m": v=(os.environ.get("CHAT_1H","") or "").strip()
    return v or os.environ.get("TELEGRAM_CHAT_ID")
def send_post(p):
    tok,cid=os.environ.get("TELEGRAM_TOKEN"),chat_for(p.get("ch","4h"))
    if not (tok and cid):
        print(p.get("caption") or "", "\n", p.get("text") or "")
        if p.get("photo") and os.environ.get("SAVE_PNG"): open(os.environ["SAVE_PNG"]+f"_{abs(hash(p['caption']))%1000}.png","wb").write(p["photo"])
        return
    if p.get("photo"): tg("sendPhoto",{"chat_id":cid,"caption":p["caption"],"parse_mode":"HTML"},{"photo":("chart.png",p["photo"])})
    if p.get("text"): tg("sendMessage",{"chat_id":cid,"text":p["text"],"parse_mode":"HTML","disable_web_page_preview":"true"})
def send(text): send_post(dict(text=html.escape(text)))

if __name__=="__main__":
    import sys
    von=float(os.environ.get("VON","1000")); vol1=float(os.environ.get("VOL_1H","200")); status=os.environ.get("STATUS")=="1"
    A4=klines("AAVEUSDT"); B4=klines("BTCUSDT"); H1=klines("AAVEUSDT","1h")
    if "--chats" in sys.argv:   # liệt kê các nhóm bot đã được thêm vào (để lấy chat id)
        me=tg("getMe",{}).get("result",{}); print("BOT:",me.get("first_name"),"@"+str(me.get("username")))
        seen={}
        for u in tg("getUpdates",{"allowed_updates":'["message","my_chat_member","channel_post"]'}).get("result",[]):
            for k in ("message","my_chat_member","channel_post"):
                c=(u.get(k) or {}).get("chat")
                if c: seen[c["id"]]=c.get("title") or c.get("username") or c.get("first_name")
        for i,t in seen.items(): print(i,t)
        sys.exit()
    if "--test" in sys.argv:   # mô phỏng tin kèo trên dữ liệu thật (ép bật tín hiệu) để xem mẫu
        res,_=evaluate(A4,B4,von); r=[x for x in res if x["trend"]] or res; r[0]["ok"]=True
        for p in f8_posts([r[0]],A4,von,False): p["caption"]="🧪 <b>TIN MẪU – KHÔNG PHẢI KÈO THẬT</b>\n"+p["caption"]; send_post(p)
        sys.exit()
    now=dt.datetime.now(dt.timezone.utc).timestamp()*1000; set_window(now)
    status=status or inw(now//86400000*86400000)   # bản tin trạng thái mỗi ngày lúc 7h sáng VN (0h UTC)
    print("cửa sổ:",tvn(WIN[0]),"→",tvn(WIN[1]),"| trạng thái:",status)
    res,_=evaluate(A4,B4,von); fresh4=inw(A4.tc.iloc[-1]+1)
    posts=[dict(p,ch="4h") for p in (f8_posts(res,A4,von,status) if (fresh4 or status) else [])]
    def walk(L,T,ch,**kw):   # xét lần lượt từng nến đóng trong cửa sổ để không sót / không trùng kèo
        out=[]; idx=[int(x) for x in np.where([inw(v+1) for v in L.tc.values])[0]]
        if status and (len(L)-1) not in idx: idx.append(len(L)-1)
        for e in sorted(idx):
            sub=L.iloc[:e+1]; st=status and e==len(L)-1
            ps=h1_posts(sub,T[T.tc<=sub.tc.iloc[-1]],vol1,st,**kw)
            if not inw(sub.tc.iloc[-1]+1): ps=[p for p in ps if p.get("caption","") and "TRẠNG THÁI" in (p.get("caption") or "")]
            out+=[dict(p,ch=ch) for p in ps]
        return out
    posts+=walk(H1,A4,"1h")
    try:
        M15=klines("AAVEUSDT","15m"); posts+=walk(M15,H1,"15m",lab="M15",hs=M15_HIST,tlab="1H",warn=M15_WARN)
    except Exception as ex: print("m15 lỗi:",ex)
    try: posts+=[dict(p,ch="paper") for p in paper_posts(status)]
    except Exception as e: print("paper lỗi:",e)
    try:
        import multi; multi.WIN=WIN; multi.EVP=event_posts; posts+=[dict(p,ch="multi") for p in multi.multi_posts(status)]
    except Exception as e: print("multi lỗi:",e)
    for p in posts: send_post(p)
    if not posts: print("Chưa có kèo.")
