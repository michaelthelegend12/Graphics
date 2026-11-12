import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve
from scipy.io import wavfile
SR=44100; DUR=20.0; N=int(SR*DUR)
rng=np.random.default_rng(7)
tt=np.arange(N)/SR
BEAT=0.625; BAR=2.5; EIGHTH=BEAT/2
mtof=lambda m:440.0*2**((m-69)/12)
def pan(sig,p):  # p in -1..1
    l=np.cos((p+1)*np.pi/4); r=np.sin((p+1)*np.pi/4); return np.stack([sig*l,sig*r])
def place(buf,sig,t0,gain=1.0,p=0.0):
    i=int(t0*SR); n=min(len(sig),N-i)
    if n<=0: return
    buf[:,i:i+n]+=pan(sig[:n],p)*gain
def tone(f,dur,harm=(1,.3,.12),decay=.3,att=.003,t=None):
    n=int(dur*SR); x=np.arange(n)/SR
    s=sum(a*np.sin(2*np.pi*f*(k+1)*x) for k,a in enumerate(harm))
    env=np.minimum(1,x/att)*np.exp(-x/decay)
    return s*env
dry=np.zeros((2,N)); low=np.zeros((2,N))

# ---- chords (midi) per time span
CH=[(0,5,[48,55,59,62,64],36),(5,10,[53,57,60,64,67],41),(10,15,[57,60,64,67,71],45),(15,17.5,[55,59,62,64,69],43),(17.5,20.4,[48,55,59,62,64,67],36)]
# ---- pad
pad=np.zeros((2,N))
for (a,b,notes,root) in CH:
    a0=max(0,a-.5); b0=min(DUR+.5,b+.9)
    n=int((b0-a0)*SR); x=np.arange(n)/SR
    env=np.minimum(1,x/(a-a0+.9))*np.minimum(1,(b0-a0-x)/1.0)
    env=np.clip(env,0,1)**1.5
    for j,m in enumerate(notes):
        for det,pp in ((-5,-.5),(0,0),(5,.5)):
            f=mtof(m)*2**(det/1200)
            s=(np.sin(2*np.pi*f*x)+.45*np.sin(2*np.pi*2*f*x+j)+.18*np.sin(2*np.pi*3*f*x)+.07*np.sin(2*np.pi*4*f*x))
            s*=1+.08*np.sin(2*np.pi*(.25+.07*j)*x)
            place(pad,s*env,a0,.05,pp*0.8)
lvl=np.interp(tt,[0,.4,2.5,5,17.5,19.4,20],[0,.55,.8,1.0,1.1,1.0,.9])
dry+=pad*lvl
# ---- arpeggio
PAT=[0,2,4,2,1,3,4,3]
step=0
t=2.5
while t<17.5-1e-6:
    for (a,b,notes,root) in CH:
        if a<=t<b: ns=notes; break
    idx=PAT[step%8]%len(ns)
    m=ns[idx]+12
    g=np.interp(t,[2.5,5,10,17.5],[.45,.8,1.0,.8])
    sig=tone(mtof(m),1.2,(1,.32,.12),decay=.26)
    place(dry,sig,t,.11*g,.45*np.sin(step*1.3))
    step+=1; t+=EIGHTH
# soft high shimmer on final bar
for k,m in enumerate([72,76,79,83,86]):
    place(dry,tone(mtof(m),2.2,(1,.0,.0),decay=.8),18.05+k*.12,.07,(-.5+k*.25)); place(dry,tone(mtof(m)*2.76,2.2,(1,),decay=.3),18.05+k*.12,.025,(-.5+k*.25))
# ---- bass + kick
def kick(g=1.0):
    n=int(.35*SR); x=np.arange(n)/SR
    f=48+90*np.exp(-x/.03); ph=2*np.pi*np.cumsum(f)/SR
    return np.sin(ph)*np.exp(-x/.11)*np.minimum(1,x/.002)
for (a,b,notes,root) in CH:
    if a<5: continue
    bt=max(a,5.0)
    while bt<b-.01 and bt<19.0:
        if abs(((bt-a)%BAR))<1e-6 or abs(((bt-a)%BAR)-2*BEAT)<1e-6:
            d=1.1 if abs(((bt-a)%BAR))<1e-6 else .55
            x=np.arange(int(d*SR))/SR; s=np.sin(2*np.pi*mtof(root)*x)+.25*np.sin(2*np.pi*2*mtof(root)*x)
            s*=np.minimum(1,x/.01)*np.exp(-x/(d*.55))
            place(low,s,bt,.20,0)
        bt+=BEAT
t=5.0
while t<17.5-1e-6:
    place(low,kick(),t,.30 if abs(((t-5)%BAR))<1e-6 else .20,0); t+=BEAT
# ---- hats / claps
sos_hp=butter(4,6500,'hp',fs=SR,output='sos'); sos_bp=butter(2,[1200,5200],'bp',fs=SR,output='sos')
def burst(dur,sos,decay):
    n=int(dur*SR); x=np.arange(n)/SR; s=sosfilt(sos,rng.standard_normal(n)); return s*np.exp(-x/decay)*np.minimum(1,x/.001)
t=7.5+EIGHTH
while t<17.5:
    place(dry,burst(.08,sos_hp,.018),t,.030,.3); t+=BEAT
t=10.0+BEAT
while t<17.5:
    place(dry,burst(.2,sos_bp,.05),t,.075,0); t+=2*BEAT
# ---- sfx
def whoosh(t0,dur,lo,hi,g,p=0):
    n=int(dur*SR); x=np.arange(n)/SR; w=rng.standard_normal(n)
    out=np.zeros(n); seg=8
    for k in range(seg):
        a=int(n*k/seg); b=int(n*(k+1)/seg)
        fc=lo*(hi/lo)**(np.sin(np.pi*(k+.5)/seg)**1)
        sos=butter(2,[fc*.7,fc*1.4],'bp',fs=SR,output='sos'); out[a:b]=sosfilt(sos,w)[a:b]
    env=np.sin(np.pi*x/dur)**2
    out=out*env; out/= (np.abs(out).max()+1e-9)
    place(dry,out,t0,g,p)
for s in (6.25,10.0,13.75,17.5): whoosh(s-.12,.9,500,3200,.16,0)
whoosh(8.0,.7,900,5500,.10,.3)
# counter shimmer 3.4-5.95 (rising soft sine)
n=int(2.6*SR); x=np.arange(n)/SR
f=mtof(72)*2**(x/2.6); ph=2*np.pi*np.cumsum(f)/SR
sh=np.sin(ph)*np.sin(np.pi*x/2.6)**2*(1+.5*np.sin(2*np.pi*9*x)); place(dry,sh,3.4,.035,.2)
n=int(1.0*SR); x=np.arange(n)/SR
f=mtof(76)*2**(x/1.0); ph=2*np.pi*np.cumsum(f)/SR
place(dry,np.sin(ph)*np.sin(np.pi*x/1.0)**2,10.9,.03,-.2)
# bell helper
def bell(m,dur=1.6,decay=.5):
    f=mtof(m); n=int(dur*SR); x=np.arange(n)/SR
    return (np.sin(2*np.pi*f*x)+.4*np.sin(2*np.pi*f*2.76*x)+.2*np.sin(2*np.pi*f*5.4*x))*np.exp(-x/decay)*np.minimum(1,x/.002)
# whoosh tick for slider/tooltip pop at 11.9, button press at 15.95
place(dry,bell(91,.5,.08),11.9,.05,.3)
place(dry,burst(.05,sos_hp,.01),15.95,.05,0); place(low,kick(),15.95,.12,0)
# success chime 16.45
for k,m in enumerate([86,91,95]): place(dry,bell(m,1.8,.55),16.45+k*.07,.10,(-.3+k*.3))
# riser 16.0 -> 17.5, impact at 17.5
n=int(1.5*SR); x=np.arange(n)/SR
r=rng.standard_normal(n); sosr=butter(2,[1500,9000],'bp',fs=SR,output='sos'); r=sosfilt(sosr,r)*(x/1.5)**2.5
place(dry,r,16.0,.20,0)
n=int(2.2*SR); x=np.arange(n)/SR
f=34+60*np.exp(-x/.12); ph=2*np.pi*np.cumsum(f)/SR
place(low,np.sin(ph)*np.exp(-x/.7)*np.minimum(1,x/.003),17.5,.55,0)
place(dry,bell(84,3.0,1.0),17.5,.09,-.2); place(dry,bell(79,3.0,1.0),17.5,.07,.2)
# ---- reverb
ir_n=int(2.4*SR); ix=np.arange(ir_n)/SR
irs=[sosfilt(butter(1,5000,'lp',fs=SR,output='sos'),rng.standard_normal(ir_n))*np.exp(-ix/.55) for _ in range(2)]
irs=[ir/np.sqrt((ir**2).sum()) for ir in irs]
wet=np.stack([fftconvolve(dry[c],irs[c])[:N] for c in range(2)])
# ping-pong echo on arp region (dotted 8th)
dl=int(.46875*SR); echo=np.zeros_like(dry)
echo[0,dl:]+=dry[1,:-dl]*.28; echo[1,dl:]+=dry[0,:-dl]*.28
echo[0,2*dl:]+=dry[0,:-2*dl]*.12; echo[1,2*dl:]+=dry[1,:-2*dl]*.12
mix=dry*.85+wet*.55+echo*.5+low
# ---- master
mix=np.tanh(mix*1.25)/np.tanh(1.25)
mix*=np.interp(tt,[0,.08,19.2,20],[0,1,1,0])**1.0
pk=np.abs(mix).max(); mix*=0.89/pk
print('peak',pk,'rms dB',20*np.log10(np.sqrt((mix**2).mean())))
wavfile.write('music.wav',SR,(mix.T*32767).astype(np.int16))
