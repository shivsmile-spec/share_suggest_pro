import json, math, time, subprocess, sys
from datetime import datetime, date
import requests

subprocess.check_call([sys.executable,"-m","pip","install","-q","-U","yfinance","curl_cffi","pandas","numpy"])
import pandas as pd
import numpy as np
import yfinance as yf

SYMBOL_URL="https://raw.githubusercontent.com/ganeshbiyer/Nse_Historical_Data/main/nifty500_symbols.csv"
DIVIDEND_URL="https://raw.githubusercontent.com/shivsmile-spec/dividend_shares3/main/data/dividends.json"

symbols_df=pd.read_csv(SYMBOL_URL)
col=next(c for c in symbols_df.columns if "symbol" in c.lower())
symbols=sorted(set(str(x).strip().upper() for x in symbols_df[col].dropna() if str(x).strip()))
tickers=[s+".NS" for s in symbols]
print("Nifty 500 symbols:",len(symbols))

try:
    div_data=requests.get(DIVIDEND_URL,timeout=30).json()
    div_map={str(x.get("symbol","")).upper():x for x in div_data.get("events",[])}
except Exception as e:
    print("Dividend feed unavailable:",e); div_map={}

hist=yf.download(tickers=tickers,period="1y",interval="1d",group_by="ticker",auto_adjust=False,threads=True,progress=False)

def rsi(c,n=14):
    d=c.diff();g=d.clip(lower=0);l=-d.clip(upper=0)
    ag=g.rolling(n).mean();al=l.rolling(n).mean()
    if len(c)<n+1:return None
    a=float(ag.iloc[-1]);b=float(al.iloc[-1])
    return 100.0 if b==0 else float(100-100/(1+a/b))

def atr(h,n=14):
    prev=h["Close"].shift(1)
    tr=pd.concat([(h["High"]-h["Low"]),(h["High"]-prev).abs(),(h["Low"]-prev).abs()],axis=1).max(axis=1)
    v=tr.rolling(n).mean().iloc[-1]
    return float(v) if pd.notna(v) else None

def adx(h,n=14):
    high,low,close=h["High"],h["Low"],h["Close"]
    up=high.diff(); down=-low.diff()
    plus=np.where((up>down)&(up>0),up,0.0); minus=np.where((down>up)&(down>0),down,0.0)
    prev=close.shift(1)
    tr=pd.concat([(high-low),(high-prev).abs(),(low-prev).abs()],axis=1).max(axis=1)
    atrv=pd.Series(tr,index=h.index).rolling(n).mean()
    pdi=100*pd.Series(plus,index=h.index).rolling(n).mean()/atrv
    mdi=100*pd.Series(minus,index=h.index).rolling(n).mean()/atrv
    dx=100*(pdi-mdi).abs()/(pdi+mdi)
    v=dx.rolling(n).mean().iloc[-1]
    return float(v) if pd.notna(v) else None

def calc_one(sym):
    try:
        h=hist[sym+".NS"].dropna()
        if len(h)<60:return None
        close=h["Close"].astype(float); high=h["High"].astype(float); low=h["Low"].astype(float); vol=h["Volume"].astype(float)
        p=float(close.iloc[-1]); prev=float(close.iloc[-2])
        r=rsi(close); a=atr(h); ad=adx(h)
        ma20=float(close.tail(20).mean()); ma50=float(close.tail(50).mean()); ma200=float(close.tail(200).mean()) if len(close)>=200 else None
        e12=close.ewm(span=12,adjust=False).mean(); e26=close.ewm(span=26,adjust=False).mean(); macd_line=e12-e26; signal=macd_line.ewm(span=9,adjust=False).mean()
        macd=float(macd_line.iloc[-1]); macd_sig=float(signal.iloc[-1])
        mom1=float((p/close.iloc[-2]-1)*100); mom5=float((p/close.iloc[-6]-1)*100) if len(close)>=6 else None
        mom20=float((p/close.iloc[-21]-1)*100) if len(close)>=21 else None
        mom6=float((p/close.iloc[-127]-1)*100) if len(close)>=127 else None
        ret=close.pct_change().dropna().tail(20); volatility=float(ret.std()*math.sqrt(252)*100) if len(ret)>5 else None
        hi52=float(high.tail(252).max()); lo52=float(low.tail(252).min()); range_pos=float((p-lo52)/(hi52-lo52)*100) if hi52>lo52 else 50
        avgvol=float(vol.tail(20).mean()); vr=float(vol.iloc[-1]/avgvol) if avgvol else None
        support=float(low.tail(20).min()); resistance=float(high.tail(20).max())
        bbmid=close.rolling(20).mean(); bbstd=close.rolling(20).std(); upper=bbmid+2*bbstd; lower=bbmid-2*bbstd
        bb=float((p-lower.iloc[-1])/(upper.iloc[-1]-lower.iloc[-1])*100) if pd.notna(upper.iloc[-1]) and upper.iloc[-1]!=lower.iloc[-1] else None
        ll=low.tail(14).min(); hh=high.tail(14).max(); stoch=float((p-ll)/(hh-ll)*100) if hh>ll else None

        # Short-term score: momentum + trend + volume + RSI + MACD + ADX + proximity to support/resistance + volatility.
        st=50.0; sw=50.0; checks=[]
        def add(name,reading,signal,impact,score_st=0,score_sw=0):
            nonlocal st,sw
            st+=score_st; sw+=score_sw
            checks.append({"factor":name,"reading":reading,"signal":signal,"impact":impact,"score":score_st+score_sw})

        add("Price vs 20 DMA", "Above" if p>ma20 else "Below", "Bullish" if p>ma20 else "Bearish", "ST", 8 if p>ma20 else -8, 5 if p>ma20 else -5)
        if ma50: add("Price vs 50 DMA","Above" if p>ma50 else "Below","Bullish" if p>ma50 else "Bearish","ST/SW",5 if p>ma50 else -5,7 if p>ma50 else -7)
        if ma200: add("Price vs 200 DMA","Above" if p>ma200 else "Below","Bullish" if p>ma200 else "Bearish","SW/LT",2 if p>ma200 else -2,8 if p>ma200 else -8)
        if r is not None:
            sig="Oversold" if r<30 else "Overbought" if r>70 else "Healthy" if 45<=r<=65 else "Weak"
            delta=5 if 45<=r<=65 else 2 if r<35 else -5 if r>72 else -2
            add("RSI 14",f"{r:.1f}",sig,"ST",delta,delta/2)
        if macd_sig is not None:
            add("MACD crossover","Above signal" if macd>macd_sig else "Below signal","Bullish" if macd>macd_sig else "Bearish","ST/SW",5 if macd>macd_sig else -5,5 if macd>macd_sig else -5)
        if mom5 is not None:add("5-day momentum",f"{mom5:.1f}%","Positive" if mom5>0 else "Negative","ST",6 if mom5>0 else -6,2 if mom5>0 else -2)
        if mom20 is not None:add("20-day momentum",f"{mom20:.1f}%","Positive" if mom20>0 else "Negative","SW",3 if mom20>0 else -3,6 if mom20>0 else -6)
        if ad is not None:add("ADX 14",f"{ad:.1f}","Strong trend" if ad>=25 else "Weak trend","ST/SW",4 if ad>=25 else 0,4 if ad>=25 else 0)
        if vr is not None:add("Volume vs 20-day avg",f"{vr:.2f}x","Above average" if vr>1.2 else "Normal/low","ST",5 if vr>1.2 else 0,2 if vr>1.2 else 0)
        if a is not None:
            if volatility is not None and volatility>45:add("Volatility",f"{volatility:.1f}%","High risk","ST",-5,-3)
            elif volatility is not None:add("Volatility",f"{volatility:.1f}%","Manageable","ST",2,2)
        dist_res=(resistance-p)/p*100 if p else None
        dist_sup=(p-support)/p*100 if p else None
        if dist_res is not None and dist_sup is not None:
            if dist_res<2 and dist_sup>4:add("Range location",f"{dist_res:.1f}% below resistance","Near resistance","ST",-4,0)
            elif dist_sup<2:add("Range location",f"{dist_sup:.1f}% above support","Near support","ST",4,1)
            else:add("Range location",f"{dist_res:.1f}% to resistance","Mid-range","ST",1,1)

        st=max(0,min(100,st)); sw=max(0,min(100,sw))

        # Dividend catalyst score, using the user's dividend scanner feed when an upcoming event exists.
        d=div_map.get(sym,{})
        div_amt=d.get("dividend"); ex=d.get("exDate")
        days=None
        try: days=(date.fromisoformat(ex)-date.today()).days if ex else None
        except: pass
        dy=float(div_amt/p*100) if div_amt and p else None
        dv=20.0
        if div_amt: dv+=30
        if dy and dy>=4: dv+=20
        elif dy and dy>=2: dv+=12
        if days is not None:
            if 0<=days<=7: dv+=20
            elif 8<=days<=21: dv+=12
            elif days<0: dv-=10
        dv=max(0,min(100,dv))
        dvv="Upcoming dividend catalyst" if dv>=70 else "Dividend worth monitoring" if dv>=50 else "No strong dividend catalyst"

        # Long-term score remains conservative because the free dataset does not reliably contain fundamentals.
        lt=50.0
        if ma200: lt += 8 if p>ma200 else -8
        if mom6 is not None: lt += 7 if mom6>10 else 3 if mom6>0 else -7
        if volatility is not None: lt += 2 if volatility<30 else -3 if volatility>45 else 0
        lt=max(0,min(100,lt))
        ltv="Positive / watch" if lt>=60 else "Neutral / mixed" if lt>=45 else "Weak technical backdrop"

        swingv="Bullish swing setup" if sw>=70 else "Positive / watch" if sw>=55 else "Neutral / mixed" if sw>=45 else "Weak swing setup"
        stv="Trade setup favorable" if st>=70 else "Watch for confirmation" if st>=55 else "Avoid chasing / wait" if st>=40 else "Weak short-term setup"

        # Entry/target/stop reference. Conservative: only propose an entry near support when reward/risk is acceptable.
        entry_low=support
        entry_high=min(ma20,resistance) if ma20 else resistance
        target=resistance
        stop=max(0,support-(a or p*0.02))
        risk=max(0,entry_high-stop); reward=max(0,target-entry_high)
        rr=(reward/risk) if risk>0 else None

        if st>=70 and (rr is None or rr>=1.5):
            main="SHORT-TERM: TRADE SETUP LOOKS FAVORABLE"
        elif st>=55:
            main="SHORT-TERM: WATCH — WAIT FOR CONFIRMATION"
        else:
            main="SHORT-TERM: NO CLEAR BUY SIGNAL YET"
        why=f"Short-term score {st:.0f}/100. RSI {r:.1f} and recent momentum/volume/trend are weighted separately; frequent price swings alone are not treated as bullish."

        return {
          "symbol":sym,"company":sym,"price":p,"change":((p-prev)/prev*100 if prev else None),
          "rsi":r,"ma50":ma50,"ma200":ma200,"macd":macd,"momentum1d":mom1,"momentum5d":mom5,"momentum20d":mom20,"momentum6m":mom6,
          "volatility":volatility,"volume":float(vol.iloc[-1]),"volumeVsAvg":f"{vr:.2f}x of 20-day average" if vr else "Unavailable","rangePosition":range_pos,
          "atr":a,"adx":ad,"stoch":stoch,"bbPosition":bb,"support":support,"resistance":resistance,
          "entryLow":entry_low,"entryHigh":entry_high,"target":target,"stopLoss":stop,"riskReward":rr,
          "pe":None,"forwardPE":None,"roe":None,"roa":None,"debtEquity":None,"margin":None,"revenueGrowth":None,
          "dividendAmount":div_amt,"exDate":ex,"daysToEx":days,"dividendYield":dy,
          "shortTermScore":st,"shortTermVerdict":stv,"swingScore":sw,"swingVerdict":swingv,
          "dividendScore":dv,"dividendVerdict":dvv,"longTermScore":lt,"longTermVerdict":ltv,
          "mainDecision":main,"mainDecisionWhy":why,"score":st,
          "verdict":stv,"reason":why,"checks":checks
        }
    except Exception as e:
        print("skip",sym,e); return None

out=[]
for i,s in enumerate(symbols,1):
    x=calc_one(s)
    if x:out.append(x)
    if i%50==0:print(i,"/",len(symbols))

data={"updated":datetime.now().strftime("%d %b %Y, %I:%M %p IST"),"count":len(out),"universe":"NIFTY 500","stocks":out}
with open("data/stocks.json","w",encoding="utf-8") as f:json.dump(data,f,ensure_ascii=False)
print("Saved",len(out))
