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
def klines(sym):
    u=f"https://data-api.binance.vision/api/v3/klines?symbol={sym}&interval=4h&limit=1000"
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
    res,tc=evaluate(klines("AAVEUSDT"),klines("BTCUSDT"),von)
    msg=message(res,tc,von)
    if msg: send(msg)
    else: print("Chưa có kèo.")
