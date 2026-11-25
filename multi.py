"""Sổ lệnh ẢO đa coin: kiểu squeeze (4 EMA bó hẹp rồi xếp tầng) ở khung 1h (xu hướng theo 4h của chính coin đó)
và khung m15 (xu hướng theo 1h). Không đặt lệnh thật. Không lưu trạng thái: mỗi lần chạy mô phỏng lại từ PAPER_START.
Backtest 35 coin 2022–10/2026 (vol cố định 200$/lệnh): 1h: 2.584 lệnh, +2.882$ (giai đoạn kiểm định 2025–26 vẫn dương),
m15: 11.300+ lệnh, lỗ sau phí -> m15 chỉ để quan sát."""
import os, numpy as np, pandas as pd, datetime as dt, concurrent.futures as cf
import paper as P
P.MS.setdefault("15m",900000)
COINS="BTC ETH SOL BNB XRP DOGE ADA AVAX LINK DOT LTC BCH TRX ATOM NEAR FIL APT ARB OP INJ SUI UNI AAVE ETC XLM HBAR TIA SEI WLD PEPE RENDER TON ICP STX FET".split()
VN=pd.Timedelta(hours=7)
WIN=[0,0]; EVP=None
def px(x):
    from f8_alert import fmt; return fmt(x)
def books(start,vol):
    def one(coin):
        s=coin+"USDT"
        try:
            H4=P.hist(s,"4h",start-1100*P.MS["4h"]); H1=P.hist(s,"1h",start-1100*P.MS["1h"]); M=P.hist(s,"15m",start-1100*P.MS["15m"])
            out={}
            for k,(L,Tr) in {"1h":(H1,H4),"15m":(M,H1)}.items():
                if len(L)<400 or len(Tr)<300: continue
                T,_=P.h1_book(L,Tr,int(np.searchsorted(L.t.values,start)),vol,hl=(k=="15m"))
                for x in T:
                    x["coin"]=coin; x["t_sig"]=L.t.values[x["i"]]+P.MS[k]; x["t_x"]=L.t.values[x["j"]]+P.MS[k]; x["last"]=L.c.values[-1]
                out[k]=T; out[k+"_df"]=L
            return out
        except Exception as e: print("multi lỗi",coin,e); return {}
    with cf.ThreadPoolExecutor(8) as ex: R=list(ex.map(one,COINS))
    B={"1h":[],"15m":[],"df":{}}
    for r in R:
        for k in ("1h","15m"):
            if k in r:
                B[k]+=r[k]
                if r[k]: B["df"][(r[k][0]["coin"],k)]=r[k+"_df"]
    return B
def multi_posts(status):
    start=pd.Timestamp(os.environ.get("PAPER_START","2026-11-23"),tz="UTC").value//10**6
    vol=float(os.environ.get("VOL_MULTI","100"))
    w0,w1=WIN   # cửa sổ do f8_alert đặt: các nến đóng từ lần chạy trước đến giờ
    B=books(start,vol); side=lambda s:"LONG" if s==1 else "SHORT"; lines=[]; G={}
    for k in ("1h","15m"):
        for x in sorted(B[k],key=lambda x:x["t_sig"]):
            if w0<x["t_sig"]<=w1:
                G[len(lines)]=(B["df"][(x["coin"],k)],f"Lệnh ảo {x['coin']}/USDT {k.upper()} · {side(x['s'])}",[(x['ep'],"Entry","#ffffff","-"),(x['sp'],"SL","#ef5350","--")])
                lines.append(f"📥 {k.upper()} <b>{x['coin']}</b> {side(x['s'])} ~<code>{px(x['ep'])}</code> · SL <code>{px(x['sp'])}</code>")
            if not x["open"] and w0<x["t_x"]<=w1:
                G[len(lines)]=(B["df"][(x["coin"],k)],f"Lệnh ảo {x['coin']}/USDT {k.upper()} · đóng {side(x['s'])}",[(x['ep'],"Entry","#ffffff","-"),(x['xp'],"Thoát","#66bb6a" if x['usd']>0 else "#ef5350","--")])
                lines.append(f"📤 {k.upper()} <b>{x['coin']}</b> {side(x['s'])} đóng <code>{px(x['xp'])}</code> · {'SL' if x['why']=='SL' else 'EMA cắt ngược'} · <b>{x['usd']:+,.2f}$</b>")
    head=f"🗂 <b>LỆNH ẢO ĐA COIN</b> · vol {vol:,.0f}$/lệnh"
    posts=EVP(head,lines,G) if EVP else ([dict(photo=None,caption=None,text=head+"\n"+"\n".join(lines))] if lines else [])
    if status:
        txt=f"🗂 <b>SỔ ẢO ĐA COIN</b> · {len(COINS)} coin · từ {(pd.to_datetime(start,unit='ms')+VN).strftime('%d/%m/%Y')} · vol {vol:,.0f}$/lệnh\n━━━━━━━━━━━━━━━━━━"
        for k,lab in (("1h","1H squeeze (backtest có lãi)"),("15m","M15 squeeze + lọc cấu trúc (backtest lãi mỏng)")):
            T=B[k]; c=[x for x in T if not x["open"]]; o=[x for x in T if x["open"] and not x.get("pending")]
            tot=sum(x["usd"] for x in c); fl=sum(x["usd"] for x in o)
            best=pd.Series({}, dtype=float) if not c else pd.DataFrame(c).groupby("coin").usd.sum().sort_values()
            txt+=(f"\n<b>{lab}</b>\nĐã đóng {len(c)} lệnh · thắng {sum(x['usd']>0 for x in c)} · lãi/lỗ <b>{tot:+,.2f}$</b>"
                  f"\nĐang mở {len(o)} lệnh · tạm tính {fl:+,.2f}$")
            if len(best): txt+=f"\nTốt nhất: {best.index[-1]} {best.iloc[-1]:+,.1f}$ · Tệ nhất: {best.index[0]} {best.iloc[0]:+,.1f}$"
        txt+="\n<i>Giá spot, có phí 0,05%/chiều, chưa tính funding. Lệnh ảo, không đặt lệnh thật.</i>"
        posts.append(dict(photo=None,caption=None,text=txt))
    return posts
