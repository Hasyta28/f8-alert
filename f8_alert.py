#!/usr/bin/env python3
"""F8 – AAVE 4h: kiểm tra kèo mỗi khi nến 4h đóng và báo qua Telegram.
Chỉ đọc giá công khai (data-api.binance.vision, giá spot AAVEUSDT/BTCUSDT). Không API key, không đặt lệnh.
Biến môi trường: TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, VON (vốn USD, mặc định 1000), STATUS=1 để luôn gửi trạng thái."""
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
def fmt(x): return f"{x:,.2f}"
def message(res,tc,von):
    t=(pd.to_datetime(tc,unit="ms")+pd.Timedelta(hours=7)).strftime("%d/%m %H:%M")
    lines=[]
    for r in res:
        if not r["ok"]: continue
        lines.append(f"🔔 KÈO F8 {r['side']} AAVE – nến 4h đóng {t} (giờ VN)")
        lines.append(f"Entry: ~{fmt(r['ep'])} (giá mở nến kế tiếp)\nSL (cắt khi NẾN 4H ĐÓNG qua): {fmt(r['sp'])} ({r['d']*100:.1f}%)\nSL cứng đặt trên sàn: {fmt(r['hard'])}\nMốc 1R: {fmt(r['tp'])}")
        lines.append(f"Điểm tự động {r['auto']}/5 + tự kiểm tra trên Binance: Long/Short ratio <1.5? funding TB thị trường <0.04%?\nDấu hiệu cuối sóng: {r['late']}/4 | tuổi xu hướng {r['age']} nến")
        for scx,p in r["plans"]:
            if p is None: lines.append(f"• Nếu tổng điểm {scx}: BỎ QUA"); continue
            hs=p["hist"]; hstr=f" | lịch sử: {hs[0]} lệnh, thắng {hs[1]}%, TB {hs[2]:+.1f}% vốn, tệ nhất {hs[3]}%" if hs else ""
            add=f", chân 1 {p['leg1']:,.0f}$ + treo nhồi {p['leg2']:,.0f}$ ở EMA100 {fmt(r['e100'])}" if p["add"] else ""
            lines.append(f"• Nếu tổng điểm {scx}: rủi ro {p['rk']*100:.0f}% vốn → tổng vol {p['tot']:,.0f}${add}{hstr}")
        lines.append("Tại 1R: giá đóng cùng phía EMA20 + BTC cùng phía EMA20 BTC + RSI>60 (short <40) → dời SL lên +0.5R và gồng 2.5 ATR; không thì chốt hết.\nChốt sớm: lãi ≥0.3R mà có nến nhấn chìm ngược hoặc đóng ngược qua EMA20.\nBạn tự quyết định. Đây không phải lệnh tự động.")
    if not lines and os.environ.get("STATUS")=="1":
        for r in res:
            if r["trend"]: lines.append(f"AAVE {t}: xu hướng {r['side']}, tuổi {r['age']} nến, {r['late']}/4 dấu hiệu cuối sóng, giá {fmt(r['ep'])}, EMA20 {fmt(r['e20'])}, EMA50 {fmt(r['e50'])}. Chưa có kèo.")
        if not lines: lines.append(f"AAVE {t}: không có xu hướng rõ, chưa có kèo.")
    return "\n".join(lines)

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
def h1_messages(H,A4,vol):
    R=h1_eval(H,A4); i=R["n"]-1; msgs=[]; t=lambda k:(pd.to_datetime(R["tc"][k],unit="ms")+pd.Timedelta(hours=7)).strftime("%d/%m %H:%M")
    hs=H1_HIST
    d=R["entry"](i)
    if d:
        sp=(R["lo"][i]-R["a"][i]) if d==1 else (R["hi"][i]+R["a"][i]); ep=R["c"][i]; rp=abs(ep-sp)/ep
        msgs.append(f"🔔 KÈO 1H {'LONG' if d==1 else 'SHORT'} AAVE (squeeze EMA) – nến 1h đóng {t(i)}\n"
            f"Entry: ~{fmt(ep)} (giá mở nến kế tiếp)\nSL: {fmt(sp)} ({rp*100:.1f}%) – lỗ tối đa ~{vol*rp:,.0f}$ với vol {vol:,.0f}$\n"
            f"Thoát: khi EMA20 cắt ngược EMA50 trên nến 1h đóng (tool sẽ nhắn).\nXu hướng 4h tuổi {int(R['hage'][i])} nến.\n"
            f"Lịch sử 2022-11/2026: {hs['n']} lệnh, thắng {hs['win']}% (2025-26: {hs['OOS'][1]}%), thắng TB +{hs['avg_win']}$, thua TB {hs['avg_loss']}$ với vol 200$. Ăn nhờ vài lệnh chạy dài, thua nhiều lệnh nhỏ.\nBạn tự quyết định.")
    # nhắn thoát: tìm lệnh 1h gần nhất còn mở và xem nến vừa đóng có phải điểm thoát
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
                    xp=sp if hit else R["c"][j]; pnl=dd*(xp-ep)/ep*vol
                    msgs.append(f"🚪 THOÁT LỆNH 1H {'LONG' if dd==1 else 'SHORT'} AAVE – nến 1h đóng {t(i)}\n"
                        f"{'Chạm SL' if hit else 'EMA20 cắt ngược EMA50'}. Vào ~{fmt(ep)} lúc {t(start)}, thoát ~{fmt(xp)} → khoảng {pnl:+,.1f}$ với vol {vol:,.0f}$ (chưa trừ phí).")
                break
    return msgs, R
def send(text):
    tok,cid=os.environ.get("TELEGRAM_TOKEN"),os.environ.get("TELEGRAM_CHAT_ID")
    if not (tok and cid): print(text); return
    data=urllib.parse.urlencode({"chat_id":cid,"text":text}).encode()
    urllib.request.urlopen(f"https://api.telegram.org/bot{tok}/sendMessage",data=data,timeout=30)
if __name__=="__main__":
    import sys
    von=float(os.environ.get("VON","1000"))
    if "--test" in sys.argv:   # gửi 1 tin mẫu (kèo F8 short 04/11/2026 đã xảy ra) để thử Telegram
        send("🧪 TIN THỬ – tool F8 đã kết nối Telegram.\nVí dụ kèo thật đã xảy ra: SHORT AAVE nến 4h đóng 04/11 07:00, entry ~127.1, SL đóng nến 142.4 (12%), SL cứng 150.0, mốc 1R 111.9, 2/4 dấu hiệu cuối sóng → rủi ro giảm nửa. Kết quả: gồng tới 15/11, thoát 105.8, +8.3% vốn.")
    A4=klines("AAVEUSDT"); B4=klines("BTCUSDT"); H1=klines("AAVEUSDT","1h")
    now=dt.datetime.now(dt.timezone.utc).timestamp()*1000
    out=[]
    res,tc=evaluate(A4,B4,von)
    fresh4=now-A4.tc.iloc[-1] < 3600*1000          # chỉ báo kèo 4h ở lượt chạy ngay sau khi nến 4h đóng
    msg=message(res,tc,von)
    if msg and (fresh4 or os.environ.get("STATUS")=="1"): out.append(msg)
    vol1=float(os.environ.get("VOL_1H","200"))
    m1,R=h1_messages(H1,A4,vol1); out+=m1
    if os.environ.get("STATUS")=="1":
        i=R["n"]-1; out.append(f"1h: dải EMA rộng {R['w'][i]:.1f} ATR (cần <1,5 trong 30 nến gần nhất), xu hướng 4h tuổi {int(R['hage'][i])} nến. Chưa có kèo 1h mới.")
    if out: send("\n\n".join(out))
    else: print("Chưa có kèo.")
