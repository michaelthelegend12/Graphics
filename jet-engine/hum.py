import numpy as np, wave
SR=44100; D=20.0; n=int(SR*D); t=np.arange(n)/SR
rng=np.random.default_rng(3)
def bandpass(x,lo,hi):
    X=np.fft.rfft(x); f=np.fft.rfftfreq(len(x),1/SR)
    m=np.exp(-((np.clip(lo-f,0,None)/ (lo*0.5+1))**2))*np.exp(-((np.clip(f-hi,0,None)/(hi*0.5+1))**2))
    return np.fft.irfft(X*m,len(x))
def sm(x): x=np.clip(x,0,1); return x*x*(3-2*x)
ramp=sm(t/2.0)                      # spool-up in the intro
load=0.55+0.45*sm((t-11)/6)         # roar builds through BANG / BLOW
chans=[]
for c in range(2):
    rumble=bandpass(rng.standard_normal(n),30,220); rumble/=np.abs(rumble).max()
    roar=bandpass(rng.standard_normal(n),500,3500); roar/=np.abs(roar).max()
    f0=62+36*sm(t/20)+1.2*np.sin(2*np.pi*0.4*t+c)          # shaft hum, slowly rising
    ph=2*np.pi*np.cumsum(f0)/SR
    hum=sum(np.sin(k*ph+0.7*k+c*0.3)/k**1.15 for k in range(1,9))
    wf=520+380*sm(t/20)                                     # compressor whine
    wph=2*np.pi*np.cumsum(wf)/SR
    whine=np.sin(wph)*(0.6+0.4*np.sin(ph)) + 0.4*np.sin(2*wph)
    sig=(0.9*rumble*(0.6+0.4*load)+0.30*hum*(0.8+0.2*np.sin(2*np.pi*0.23*t))
         +0.07*whine*ramp*(0.6+0.4*load)+0.16*roar*load**2)
    sig*=0.25+0.75*ramp
    chans.append(sig)
y=np.stack(chans,1)
env=sm(t/0.5)*(1-sm((t-19.2)/0.8))
y*=env[:,None]
y*=0.5/np.abs(y).max()
pcm=(y*32767).astype('<i2')
with wave.open('hum.wav','wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
print('rms dB', 20*np.log10(np.sqrt((y**2).mean())), 'peak', np.abs(y).max())
