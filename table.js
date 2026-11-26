/* Liar's Dice: first-person 3D table. Loaded on demand by index.html, after three.js r128.
   Sections: rules + AI policies, sound, scene, characters, frame loop, game flow, HUD. */
(()=>{"use strict";
const $=id=>document.getElementById(id);
const esc=s=>String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const WINS=window.LD_WIN||{};
const NAMES={cfr:"Game Theorist",dqn:"Exploiter",ppo:"Gut Player",nfsp:"Hybrid",ismcts:"Thinker",mc:"Old-Schooler",math:"Math Bot",random:"Random Bot"};
const COL={cfr:"#ff5a4e",ismcts:"#ffb020",ppo:"#13b38a",nfsp:"#9b5cff",mc:"#2ba8ff",dqn:"#ff4fb0",math:"#8a93a8",random:"#c9cede"};
const STYLE={cfr:"Balanced and hard to read",ismcts:"Simulates the round before every move",ppo:"Mixes in bluffs on instinct",nfsp:"Half hunter, half averager",mc:"Learned only from who won",dqn:"Always plays its top-rated move",math:"Plays the raw odds",random:"Pure chaos"};

const FW=["","ones","twos","threes","fours","fives","sixes"],F1=["","one","two","three","four","five","six"];
const fw=(q,f)=>q+" "+(q===1?F1[f]:FW[f]);
const pct=p=>Math.round(p*100)+"%";
const store={get(k,d){try{const v=localStorage.getItem("ld4-"+k);return v===null?d:JSON.parse(v)}catch(e){return d}},set(k,v){try{localStorage.setItem("ld4-"+k,JSON.stringify(v))}catch(e){}}};
let DICE=store.get("dice",5);if(![2,3,5].includes(DICE))DICE=5;
let SPD=store.get("spd",1)===2?2:1,COACH=store.get("coach",true)!==false,MUTE=store.get("mute",false)===true;
const reduce=matchMedia("(prefers-reduced-motion: reduce)").matches;
const sleep=ms=>new Promise(r=>setTimeout(r,ms/SPD));
const sum=a=>a.reduce((x,y)=>x+y,0);

//<logic> rules and AI policies (mirrors mp.py for the 2-dice agents and gen.py for the any-dice agents; checked against both in tests)
const CALL=120,bq=k=>Math.floor(k/6)+1,bf=k=>k%6+1;
const legal=(l,tot)=>{const a=[];for(let k=l+1;k<tot*6;k++)a.push(k);if(l>=0)a.push(CALL);return a};
const comb=(n,k)=>{let r=1;for(let i=1;i<=k;i++)r=r*(n-k+i)/i;return r};
// PT[o][need] = chance that at least `need` of `o` unseen dice show one given face
const PT=[];for(let o=0;o<=15;o++){const r=[];for(let nd=0;nd<=21;nd++){let v=0;if(nd<=0)v=1;else if(nd<=o)for(let k=nd;k<=o;k++)v+=comb(o,k)*Math.pow(1/6,k)*Math.pow(5/6,o-k);r.push(v)}PT.push(r)}
const tail=(o,nd)=>nd<=0?1:nd>o?0:PT[o][nd];
const cnt6=h=>{const c=[0,0,0,0,0,0];for(const d of h)c[d-1]++;return c};
const ptrue=(q,f,h,ot)=>{let m=0;for(const d of h)if(d===f)m++;return tail(ot,q-m)};
const mathBot=(o,acts)=>{const h=o[0],ot=o[2],l=o[4];let best=null,bp=-1;for(const b of acts){if(b===CALL)continue;const p=ptrue(bq(b),bf(b),h,ot);if(p>bp+1e-9){best=b;bp=p}}
 if(l>=0){const pl=ptrue(bq(l),bf(l),h,ot);if(best===null||pl<1-bp)return CALL}return best};
const mathC=(c,ot,l,tot)=>{let best=-1,bp=-1;for(let b=l+1;b<tot*6;b++){const p=tail(ot,((b/6)|0)+1-c[b%6]);if(p>bp+1e-9){best=b;bp=p}}
 if(l>=0&&(best<0||tail(ot,((l/6)|0)+1-c[l%6])<1-bp))return CALL;return best};
const randomBot=(o,acts)=>{if(acts.includes(CALL)&&(acts.length===1||Math.random()<1/3))return CALL;const b=acts.filter(a=>a!==CALL);return b[Math.floor(Math.random()*b.length)]};
function decodeNet(layers){return layers.map(L=>{const bin=atob(L.d),u8=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u8[i]=bin.charCodeAt(i);
 const q=new Int16Array(u8.buffer),f=new Float32Array(q.length);for(let i=0;i<q.length;i++)f[i]=q[i]*L.s;return{i:L.i,o:L.o,f}})}
function fwd(net,x){let h=x;for(let li=0;li<net.length;li++){const L=net[li],o=new Float32Array(L.o),f=L.f,bo=L.i*L.o;for(let k=0;k<L.o;k++)o[k]=f[bo+k];
 for(let j=0;j<L.i;j++){const v=h[j];if(v===0)continue;const r=j*L.o;for(let k=0;k<L.o;k++)o[k]+=v*f[r+k]}
 if(li<net.length-1)for(let k=0;k<L.o;k++)if(o[k]<0)o[k]=0;h=o}return h}
const sample=pairs=>{let r=Math.random()*pairs.reduce((s,x)=>s+x[1],0);for(const p of pairs){r-=p[1];if(r<=0)return p[0]}return pairs[pairs.length-1][0]};
// ----- 2-dice agents: learned on the raw game (8 dice, call = 48 in their own numbering)
const tkey=o=>o[0].join("")+"|"+o[1]+"|"+o[2]+"|"+o[4];
const encode=o=>{const x=new Float32Array(115);for(const d of o[0])x[d-1]+=.5;x[6+o[1]-1]=1;x[8+o[2]-1]=1;x[14+o[3]-1]=1;x[17+o[4]+1]=1;x[66+o[5]+1]=1;return x};
const AG={},AG5={};
function policy2(n){
 if(n==="math")return mathBot;if(n==="random")return randomBot;if(n==="ismcts")return ismcts;
 const ag=AG[n];if(!ag)return mathBot;const to1=a=>a===CALL?48:a,from1=a=>a===48?CALL:a;
 if(ag.kind==="det")return(o,acts)=>{const a=from1(ag.policy[tkey(o)]);return a!==undefined&&acts.includes(a)?a:mathBot(o,acts)};
 if(ag.kind==="table")return(o,acts)=>{const p=(ag.policy[tkey(o)]||[]).map(x=>[from1(x[0]),x[1]]).filter(x=>acts.includes(x[0]));return p.length?sample(p):mathBot(o,acts)};
 return(o,acts)=>{const out=fwd(ag.net,encode(o)),v=a=>out[to1(a)];if(ag.kind==="greedy"){let b=acts[0];for(const a of acts)if(v(a)>v(b))b=a;return b}
  let m=-1e9;for(const a of acts)if(v(a)>m)m=v(a);return sample(acts.map(a=>[a,Math.exp(v(a)-m)]))}}
const H2={1:[[1],[2],[3],[4],[5],[6]],2:[]},HP={1:[1/6,1/6,1/6,1/6,1/6,1/6],2:[]};for(let a=1;a<=6;a++)for(let b=a;b<=6;b++){H2[2].push([a,b]);HP[2].push(a===b?1/36:2/36)}
function ismcts(o,acts,ctx,sims=700,c=1){const h=o[0],last=o[4];if(acts.length===1)return acts[0];
 const order=ctx.order,k=order.length,total=ctx.total;
 const cand=order.slice(1).map(s=>{const bids=ctx.hist.filter(x=>x[0]===s).map(x=>x[1]),n=ctx.counts[s];const ws=H2[n].map((x,i)=>HP[n][i]*bids.reduce((m,b)=>m*(1+1.5*x.filter(d=>d===bf(b)).length),1));return[H2[n],ws,ws.reduce((a,b)=>a+b,0)]});
 const pickH=cd=>{let r=Math.random()*cd[2];for(let i=0;i<cd[0].length;i++){r-=cd[1][i];if(r<=0)return cd[0][i]}return cd[0][0]};
 const N=new Map(),W=new Map();
 for(let it=0;it<sims;it++){const hands=[h].concat(cand.map(pickH));let node="",l=last,mover=0,bidder=last>=0?k-1:-1,path=[],a;
  for(;;){const la=legal(l,total),un=la.filter(x=>!N.has(node+","+x));
   if(un.length){a=un[Math.floor(Math.random()*un.length)];node+=","+a;N.set(node,0);W.set(node,new Float64Array(k));path.push(node);break}
   let t=0;for(const x of la)t+=N.get(node+","+x);let bv=-1e9;for(const x of la){const kk=node+","+x,n=N.get(kk),v=W.get(kk)[mover]/n+c*Math.sqrt(Math.log(t)/n);if(v>bv){bv=v;a=x}}
   node+=","+a;path.push(node);if(a===CALL)break;l=a;bidder=mover;mover=(mover+1)%k}
  while(a!==CALL){l=a;bidder=mover;mover=(mover+1)%k;const hm=hands[mover];a=mathBot([hm,hm.length,total-hm.length,k-1,l,-1],legal(l,total))}
  const f=bf(l),cnt=hands.reduce((s,x)=>s+x.filter(d=>d===f).length,0),loser=cnt>=bq(l)?mover:bidder;
  for(const kk of path){N.set(kk,N.get(kk)+1);const w=W.get(kk);for(let s=0;s<k;s++)w[s]+=s===loser?-1:1/(k-1)}}
 let best=acts[0],bn=-1;for(const x of acts){const n=N.get(","+x)||0;if(n>bn){bn=n;best=x}}return best}
// ----- any-dice agents (3 to 5 dice each): a shortlist of sensible moves, each described by a few odds-based numbers
const N50=PT.map((r,o)=>{let m=0;for(let nd=0;nd<=o;nd++)if(r[nd]>=.5)m=nd;return m}),N25=PT.map((r,o)=>{let m=0;for(let nd=0;nd<=o;nd++)if(r[nd]>=.25)m=nd;return m});
function cands5(c,n,others,last){const total=n+others,out=[],lq=last>=0?bq(last):0,lf=last>=0?bf(last):0;
 for(let f=1;f<=6;f++){const q0=last>=0&&f>lf?lq:lq+1;if(q0>total)continue;const qc=c[f-1]+N50[others],qa=c[f-1]+N25[others];out.push((q0-1)*6+f-1);
  if(q0<qc&&qc<=total)out.push((qc-1)*6+f-1);if(qa>qc&&q0<qa&&qa<=total)out.push((qa-1)*6+f-1)}
 out.sort((a,b)=>a-b);if(last>=0)out.push(CALL);return out}
function feats5(c,n,others,nopp,last,prev,acts){const total=n+others,sd=Math.sqrt(others*5/36)||1,mu=others/6,lq=last>=0?bq(last):0,lf=last>=0?bf(last):0,F=new Float32Array(acts.length*15),cl=v=>Math.max(-3,Math.min(3,v))/3;
 let pl=0,zl=0,jl=0;if(last>=0){const nl=lq-c[lf-1];pl=tail(others,nl);zl=cl((nl-mu)/sd);jl=prev>=0?Math.max(0,Math.min(3,lq-bq(prev)))/3:0}
 acts.forEach((a,i)=>{const r=i*15;F[r]=n/5;F[r+1]=others/15;F[r+2]=nopp/3;F[r+3]=last>=0?1:0;F[r+4]=pl;F[r+5]=zl;F[r+6]=jl;if(a===CALL){F[r+7]=1;return}
  const q=bq(a),f=bf(a),nd=q-c[f-1];F[r+8]=tail(others,nd);F[r+9]=cl((nd-mu)/sd);F[r+10]=Math.max(0,Math.min(4,q-lq))/4;F[r+11]=last>=0&&f===lf?1:0;F[r+12]=c[f-1]/n;F[r+13]=f/6;F[r+14]=q/total});return F}
const skey5=(c,n,others,nopp,last)=>{let b=0;if(last>=0){const pl=tail(others,bq(last)-c[bf(last)-1]);b=1+[.05,.2,.4,.6,.8].filter(t=>pl>=t).length}return b+"|"+(others<=2?0:others<=5?1:others<=9?2:3)+"|"+n+"|"+nopp};
function bmap5(c,others,last,acts){const lq=last>=0?bq(last):0,lf=last>=0?bf(last):0,out=new Map(),best={};
 for(const a of acts){if(a===CALL){out.set(0,a);continue}const q=bq(a),f=bf(a),pa=tail(others,q-c[f-1]),dq=q-lq,b=1+[.15,.35,.55,.8].filter(t=>pa>=t).length*6+(dq<=1?0:dq===2?1:2)*2+(last>=0&&f===lf?1:0);
  if(!out.has(b)||pa>best[b]+1e-9){out.set(b,a);best[b]=pa}}return out}
function policy5(n){
 if(n==="math")return mathBot;if(n==="random")return randomBot;if(n==="ismcts")return ismcts5;
 const ag=AG5[n];if(!ag)return mathBot;
 if(ag.net)return(o)=>{const c=cnt6(o[0]),A=cands5(c,o[1],o[2],o[4]);if(A.length===1)return A[0];const F=feats5(c,o[1],o[2],o[3],o[4],o[5],A),z=A.map((_,i)=>fwd(ag.net,F.subarray(i*15,i*15+15))[0]);
  if(ag.kind==="q"){let b=0;for(let i=1;i<z.length;i++)if(z[i]>z[b])b=i;return A[b]}const m=Math.max.apply(null,z);return sample(A.map((a,i)=>[a,Math.exp(z[i]-m)]))};
 return(o,acts)=>{const c=cnt6(o[0]),bm=bmap5(c,o[2],o[4],cands5(c,o[1],o[2],o[4])),row=ag.t[skey5(c,o[1],o[2],o[3],o[4])];if(!row)return mathBot(o,acts);const bs=[...bm.keys()].sort((a,b)=>a-b);
  if(ag.kind==="mcq"){let b=bs[0];for(const x of bs)if(row[x]>row[b])b=x;return bm.get(b)}const w=bs.map(b=>[bm.get(b),Math.max(row[b],0)]);return w.some(x=>x[1]>0)?sample(w):mathBot(o,acts)}}
const H5={};function hands5(n){if(H5[n])return H5[n];const out=[],fact=[1,1,2,6,24,120],rec=(f,left,c)=>{if(f===5){c[5]=left;let p=fact[n];for(const x of c)p/=fact[x];out.push({c:c.slice(),p:p/Math.pow(6,n)});return}for(let x=0;x<=left;x++){c[f]=x;rec(f+1,left-x,c)}};rec(0,n,[0,0,0,0,0,0]);return H5[n]=out}
function ismcts5(o,acts,ctx,sims=500,cx=1){const myc=cnt6(o[0]),last=o[4],order=ctx.order,k=order.length,total=ctx.total,ns=order.map(s=>ctx.counts[s]);
 const A0=cands5(myc,ns[0],total-ns[0],last);if(A0.length===1)return A0[0];
 const cand=order.slice(1).map(s=>{const fs=ctx.hist.filter(x=>x[0]===s).map(x=>bf(x[1])-1),L=hands5(ctx.counts[s]),ws=L.map(h=>{let w=h.p;for(const f of fs)w*=1+1.5*h.c[f];return w});let t=0;for(const w of ws)t+=w;return[L,ws,t]});
 const pickH=cd=>{let r=Math.random()*cd[2];for(let i=0;i<cd[0].length;i++){r-=cd[1][i];if(r<=0)return cd[0][i].c}return cd[0][0].c};
 const N=new Map(),W=new Map();
 for(let it=0;it<sims;it++){const C=[myc].concat(cand.map(pickH));let node="",l=last,mover=0,bidder=last>=0?k-1:-1,path=[],a;
  for(;;){const la=cands5(C[mover],ns[mover],total-ns[mover],l),un=la.filter(x=>!N.has(node+","+x));
   if(un.length){a=un[Math.floor(Math.random()*un.length)];node+=","+a;N.set(node,0);W.set(node,new Float64Array(k));path.push(node);break}
   let t=0;for(const x of la)t+=N.get(node+","+x);let bv=-1e9;for(const x of la){const kk=node+","+x,n=N.get(kk),v=W.get(kk)[mover]/n+cx*Math.sqrt(Math.log(t)/n);if(v>bv){bv=v;a=x}}
   node+=","+a;path.push(node);if(a===CALL)break;l=a;bidder=mover;mover=(mover+1)%k}
  while(a!==CALL){l=a;bidder=mover;mover=(mover+1)%k;a=mathC(C[mover],total-ns[mover],l,total)}
  const f=bf(l)-1;let cnt=0;for(const c of C)cnt+=c[f];const loser=cnt>=bq(l)?mover:bidder;
  for(const kk of path){N.set(kk,N.get(kk)+1);const w=W.get(kk);for(let s=0;s<k;s++)w[s]+=s===loser?-1:1/(k-1)}}
 let best=A0[0],bn=-1;for(const x of A0){const n=N.get(","+x)||0;if(n>bn){bn=n;best=x}}return best}
const policyOf=(n,start)=>start<=2?policy2(n):policy5(n);
//</logic>
async function loadAgent(n,start){const five=start>2,S=five?AG5:AG;if(n==="math"||n==="random"||n==="ismcts"||S[n])return true;
 try{const r=await fetch((five?"ai5/":"ai/")+n+".json");if(!r.ok)throw 0;const d=await r.json();S[n]=d.layers?{kind:d.kind,net:decodeNet(d.layers)}:d;return true}catch(e){return false}}

// ---------- sound: tiny synthesised effects, no audio files ----------
let AC=null,OUT=null,NB=null;
function audioOn(){if(MUTE)return;try{if(!AC){AC=new(window.AudioContext||window.webkitAudioContext)();OUT=AC.createGain();OUT.gain.value=.2;OUT.connect(AC.destination)}if(AC.state==="suspended")AC.resume()}catch(e){AC=null}}
function tone(f,d,type,v,at,f2){if(!AC||MUTE)return;try{const t=AC.currentTime+(at||0),o=AC.createOscillator(),g=AC.createGain();o.type=type||"sine";o.frequency.setValueAtTime(f,t);if(f2)o.frequency.exponentialRampToValueAtTime(f2,t+d);
 g.gain.setValueAtTime(v||.4,t);g.gain.exponentialRampToValueAtTime(.001,t+d);o.connect(g);g.connect(OUT);o.start(t);o.stop(t+d+.03)}catch(e){}}
function noise(d,v,at,fr){if(!AC||MUTE)return;try{if(!NB){NB=AC.createBuffer(1,AC.sampleRate*.3,AC.sampleRate);const a=NB.getChannelData(0);for(let i=0;i<a.length;i++)a[i]=Math.random()*2-1}
 const t=AC.currentTime+(at||0),s=AC.createBufferSource(),f=AC.createBiquadFilter(),g=AC.createGain();s.buffer=NB;f.type="bandpass";f.frequency.value=fr||2500;g.gain.setValueAtTime(v||.4,t);g.gain.exponentialRampToValueAtTime(.001,t+d);
 s.connect(f);f.connect(g);g.connect(OUT);s.start(t,Math.random()*.2,d+.03)}catch(e){}}
const SFX={rattle(){for(let i=0;i<13;i++)noise(.035,.5,i*.05+Math.random()*.03,1800+Math.random()*2600)},
 bid(){tone(430,.09,"triangle",.5,0,300);noise(.03,.25,0,3000)},turn(){tone(660,.1,"sine",.32);tone(880,.15,"sine",.32,.1)},
 liar(){tone(170,.32,"sawtooth",.5,0,80);noise(.14,.6,0,400)},tick(n){tone(520*Math.pow(1.122,n),.1,"sine",.45)},cup(){noise(.05,.3,0,900)},
 lose(){tone(380,.34,"triangle",.45,0,170)},good(){tone(587,.1,"triangle",.35);tone(784,.16,"triangle",.35,.09)},
 win(){[523,659,784,1047].forEach((f,i)=>tone(f,.2,"triangle",.4,i*.11))},out(){tone(200,.5,"sawtooth",.28,0,70)}};

// ---------- scene ----------
const TY=.78,SEAT_R=2.12,CUP_R=1.3,SA=[Math.PI/2,Math.PI+.34,Math.PI*1.5,-.34];
let R3,scene,cam,clock,running=false,anims=[],seats=[],G=null,zoom=0,zoomT=0,W=1,H=1,spotL,lowQ=false,fpsN=0,fpsT=0;
let DM,DMG,DMD,rackG,rackParts,youX=-.7,youZ=.82,lift=0,liftT=0,fit=1,fitT=1,MAT={},GEO={},AV={},dicePool=[],youG,youCup,turnArc,arcMat,bidG,bidDie,bidGlow,qc,qtx,qtySp,dust,confP,conf={life:0},bulbs=[],tgt,tgtT,pv;
const look={x:0,y:0,tx:0,ty:0};
const seatPos=(i,r,y)=>new THREE.Vector3(Math.cos(SA[i])*r,y||0,Math.sin(SA[i])*r);
const lin=h=>new THREE.Color(h).convertSRGBToLinear();
const std=(h,r,m,x)=>new THREE.MeshStandardMaterial(Object.assign({color:lin(h),roughness:r===undefined?.6:r,metalness:m||0},x));
const lam=(h,x)=>new THREE.MeshLambertMaterial(Object.assign({color:lin(h)},x));
const bas=(h,x)=>new THREE.MeshBasicMaterial(Object.assign({color:lin(h),toneMapped:false},x));
const shade=(h,k)=>"#"+new THREE.Color(h).multiplyScalar(k).getHexString();
const part=(g,m,s,p,r)=>{const o=new THREE.Mesh(g,m);if(s)o.scale.set(s[0],s[1],s[2]);if(p)o.position.set(p[0],p[1],p[2]);if(r)o.rotation.set(r[0],r[1],r[2]);return o};
const anim=(dur,f,e)=>new Promise(res=>anims.push({t0:performance.now(),dur:Math.max(1,dur/SPD),f,e:e||(k=>1-Math.pow(1-k,3)),res}));
function tex(w,h,draw,rep){const c=document.createElement("canvas");c.width=w;c.height=h;draw(c.getContext("2d"),w,h);const t=new THREE.CanvasTexture(c);t.encoding=THREE.sRGBEncoding;t.anisotropy=4;
 if(rep){t.wrapS=t.wrapT=THREE.RepeatWrapping;t.repeat.set(rep[0],rep[1])}return t}
const rr=(g,x,y,w,h,r)=>{g.beginPath();g.moveTo(x+r,y);g.arcTo(x+w,y,x+w,y+h,r);g.arcTo(x+w,y+h,x,y+h,r);g.arcTo(x,y+h,x,y,r);g.arcTo(x,y,x+w,y,r);g.closePath()};
const PIPS={1:[[.5,.5]],2:[[.28,.28],[.72,.72]],3:[[.27,.27],[.5,.5],[.73,.73]],4:[[.28,.28],[.72,.28],[.28,.72],[.72,.72]],5:[[.27,.27],[.73,.27],[.5,.5],[.27,.73],[.73,.73]],6:[[.28,.25],[.72,.25],[.28,.5],[.72,.5],[.28,.75],[.72,.75]]};
const pip=v=>'<svg viewBox="0 0 100 100" aria-label="'+v+'"><rect x="5" y="5" width="90" height="90" rx="20" fill="#fbfaf7"/>'+PIPS[v].map(p=>'<circle cx="'+p[0]*100+'" cy="'+p[1]*100+'" r="9" fill="#0b1440"/>').join("")+"</svg>";
const FT={};
function faceMat(v,mode){const t=FT[v]||(FT[v]=tex(128,128,(g,w)=>{g.fillStyle="#cfc7b6";g.fillRect(0,0,w,w);const gr=g.createLinearGradient(0,0,w,w);gr.addColorStop(0,"#fffdf8");gr.addColorStop(1,"#ece4d3");g.fillStyle=gr;rr(g,5,5,118,118,24);g.fill();
  for(const p of PIPS[v]){g.fillStyle="#0b1440";g.beginPath();g.arc(p[0]*w,p[1]*w,v===1?15:12,0,7);g.fill();g.fillStyle="rgba(255,255,255,.28)";g.beginPath();g.arc(p[0]*w-3,p[1]*w-3,4,0,7);g.fill()}}));
 const m=new THREE.MeshStandardMaterial({map:t,roughness:.32});if(mode===1){m.emissive=lin("#ffb020");m.emissiveMap=t;m.emissiveIntensity=.55}if(mode===2)m.color.setScalar(.3);return m}
function makeDie(size){const m=new THREE.Mesh(GEO.box,DM);m.castShadow=true;const g=new THREE.Group();g.add(m);g.scale.setScalar(size);g.userData.size=size;return g}
function setFace(g,v){g.children[0].rotation.set(v===6?Math.PI:v===2?-Math.PI/2:v===5?Math.PI/2:0,0,v===3?Math.PI/2:v===4?-Math.PI/2:0);g.userData.v=v}
function makeCup(){const g=new THREE.Group(),w=part(GEO.cupW,MAT.leather,null,[0,.17,0]);w.castShadow=true;
 g.add(w,part(GEO.cupT,MAT.leather,null,[0,.34,0],[-Math.PI/2,0,0]),part(GEO.ring,MAT.brass,[.213,.213,.2],[0,.03,0],[Math.PI/2,0,0]),part(GEO.ring,MAT.brass,[.178,.178,.16],[0,.3,0],[Math.PI/2,0,0]));return g}
function glowSprite(h,sx,sy,op){const s=new THREE.Sprite(new THREE.SpriteMaterial({map:MAT.glowTex,color:lin(h),transparent:true,opacity:op,blending:THREE.AdditiveBlending,depthWrite:false,toneMapped:false,fog:false}));s.scale.set(sx,sy,1);return s}

function init3d(){
 const st=$("stage");
 R3=new THREE.WebGLRenderer({antialias:true});
 R3.setPixelRatio(Math.min(1.5,window.devicePixelRatio||1));R3.outputEncoding=THREE.sRGBEncoding;R3.toneMapping=THREE.ACESFilmicToneMapping;R3.toneMappingExposure=1.25;
 R3.shadowMap.enabled=true;R3.shadowMap.type=THREE.PCFSoftShadowMap;st.appendChild(R3.domElement);
 scene=new THREE.Scene();scene.background=lin("#070919");scene.fog=new THREE.Fog(0,8,19);scene.fog.color=lin("#070919");
 cam=new THREE.PerspectiveCamera(60,1,.1,40);clock=new THREE.Clock();tgt=new THREE.Vector3(0,.85,-.55);tgtT=tgt.clone();pv=new THREE.Vector3();
 // shared geometry and materials (made once, reused everywhere)
 GEO.box=new THREE.BoxGeometry(1,1,1);GEO.cyl=new THREE.CylinderGeometry(1,1,1,20);GEO.sph=new THREE.SphereGeometry(1,16,12);GEO.hemi=new THREE.SphereGeometry(1,20,10,0,Math.PI*2,0,Math.PI/2);
 GEO.cupW=new THREE.CylinderGeometry(.17,.215,.34,22,1,true);GEO.cupT=new THREE.CircleGeometry(.17,22);GEO.ring=new THREE.TorusGeometry(1,.06,6,24);
 GEO.body=new THREE.LatheGeometry([[0,0],[.33,0],[.37,.08],[.36,.23],[.31,.53],[.27,.73],[.26,.84],[.22,.93],[.12,.98],[0,.985]].map(p=>new THREE.Vector2(p[0],p[1])),26);
 GEO.head=new THREE.SphereGeometry(.27,26,18);GEO.eye=new THREE.SphereGeometry(.036,12,10);GEO.dot=new THREE.SphereGeometry(.012,8,6);GEO.smile=new THREE.TorusGeometry(.055,.011,6,14,Math.PI);
 GEO.hand=new THREE.SphereGeometry(.078,14,10);GEO.lens=new THREE.TorusGeometry(.062,.008,6,20);
 MAT.leather=std("#6b2318",.6,0,{side:THREE.DoubleSide});MAT.brass=std("#d8a84a",.3,.8);MAT.ink=std("#1b1420",.45);MAT.glove=std("#f7f3ea",.5);MAT.wood=std("#3b1d14",.7);MAT.cush=std("#7a2019",.85);
 MAT.white=bas("#ffffff");MAT.cheek=std("#ff8f86",.8,0,{transparent:true,opacity:.42});MAT.mouth=std("#4a1616",.6);
 MAT.glowTex=tex(64,64,(g,w)=>{const r=g.createRadialGradient(32,32,0,32,32,32);r.addColorStop(0,"rgba(255,255,255,1)");r.addColorStop(.35,"rgba(255,255,255,.35)");r.addColorStop(1,"rgba(255,255,255,0)");g.fillStyle=r;g.fillRect(0,0,w,w)});
 DM=[3,4,1,6,2,5].map(v=>faceMat(v,0));DMG=[3,4,1,6,2,5].map(v=>faceMat(v,1));DMD=[3,4,1,6,2,5].map(v=>faceMat(v,2));
 // lights: one shadow-casting lamp, two cheap fills
 scene.add(new THREE.HemisphereLight(lin("#8794ff"),lin("#3a1a10"),.6));
 spotL=new THREE.SpotLight(lin("#ffdcae"),2.7,11,.98,.6,1);spotL.position.set(0,3.5,0);spotL.castShadow=true;spotL.shadow.mapSize.set(1024,1024);spotL.shadow.bias=-.0006;spotL.shadow.camera.near=1;spotL.shadow.camera.far=6;scene.add(spotL,spotL.target);
 const fill=new THREE.PointLight(lin("#aab8ff"),.6,9);fill.position.set(0,2.4,3.8);scene.add(fill);
 const rim=new THREE.PointLight(lin("#ff6a5c"),1,11);rim.position.set(-2,3,-6);scene.add(rim);
 buildRoom();buildTable();
 // dice pool: 2 per seat, reused every round
 const rw=std("#8a5632",.55);rackG=new THREE.Group();rackG.position.set(0,TY+.145,1.27);rackG.rotation.x=1;const rp=part(GEO.box,rw,[.82,.02,.32],[0,-.01,0]),rl=part(GEO.box,rw,[.82,.05,.02],[0,.015,.152]);rackG.add(rp,rl);
 const rb=part(GEO.box,rw,[.74,.27,.03],[0,TY+.135,1.165]);rb.castShadow=true;scene.add(rackG,rb);rackParts=[rp,rl,rb];
 for(let i=0;i<4;i++){const a=[];for(let j=0;j<5;j++){const d=makeDie(i?.18:.25);d.visible=false;(i?scene:rackG).add(d);a.push(d)}dicePool.push(a)}
 youG=new THREE.Group();youCup=makeCup();youG.add(youCup);
 const resize=()=>{W=st.clientWidth||1;H=st.clientHeight||1;R3.setSize(W,H,false);cam.aspect=W/H;cam.fov=W<H?80:66;setSide(W<H?.72:.34);cam.setViewOffset(W,H,0,lift,W,H);cam.updateProjectionMatrix();measure()};
 new ResizeObserver(resize).observe(st);resize();
 st.addEventListener("pointermove",e=>{const r=st.getBoundingClientRect();look.tx=((e.clientX-r.left)/r.width-.5)*.7;look.ty=((e.clientY-r.top)/r.height-.5)*.25});
 st.addEventListener("pointerleave",()=>{look.tx=0;look.ty=0})}

function buildRoom(){
 const wallT=tex(256,512,(g,w,h)=>{const gr=g.createLinearGradient(0,0,0,h);gr.addColorStop(0,"#0a0e2c");gr.addColorStop(.78,"#1b2466");g.fillStyle=gr;g.fillRect(0,0,w,h);
   for(let x=0;x<w;x+=64){g.fillStyle="rgba(140,160,255,.055)";g.fillRect(x,0,32,h*.8)}
   g.fillStyle="rgba(255,190,120,.13)";for(let y=40;y<h*.76;y+=80)for(let x=16;x<w;x+=64){g.save();g.translate(x,y);g.rotate(Math.PI/4);g.fillRect(-5,-5,10,10);g.restore()}
   const y0=h*.8;g.fillStyle="#2a1610";g.fillRect(0,y0,w,h-y0);g.fillStyle="#3d2217";for(let x=8;x<w;x+=64)g.fillRect(x,y0+18,48,h-y0-30);g.fillStyle="#c79a4a";g.fillRect(0,y0-4,w,6)},[14,1]);
 const wall=new THREE.Mesh(new THREE.CylinderGeometry(7.5,7.5,5.6,40,1,true),new THREE.MeshLambertMaterial({map:wallT,side:THREE.BackSide}));wall.position.y=2.8;scene.add(wall);
 const floorT=tex(256,256,(g,w)=>{g.fillStyle="#2a160f";g.fillRect(0,0,w,w);for(let r=0;r<8;r++){let x=-((r*37)%90);while(x<w){const L=70+((r*53+x*7+900)%60),sh=((x*13+r*29+900)%16);g.fillStyle="rgb("+(62+sh)+","+(34+sh)+","+(20+sh)+")";g.fillRect(x,r*32,L-1.5,30.5);x+=L}}},[10,10]);
 const floor=new THREE.Mesh(new THREE.CircleGeometry(7.5,40),new THREE.MeshLambertMaterial({map:floorT}));floor.rotation.x=-Math.PI/2;floor.receiveShadow=true;scene.add(floor);
 const rugT=tex(256,256,(g,w)=>{const c=w/2,ring=(r,col)=>{g.fillStyle=col;g.beginPath();g.arc(c,c,r,0,7);g.fill()};ring(c,"#5e1913");ring(c*.93,"#8f8672");ring(c*.91,"#111a4c");ring(c*.62,"#182366");
   g.fillStyle="rgba(190,180,155,.42)";for(let i=0;i<32;i++){const a=i/32*Math.PI*2;g.save();g.translate(c+Math.cos(a)*c*.76,c+Math.sin(a)*c*.76);g.rotate(a);g.beginPath();g.moveTo(-8,-7);g.lineTo(8,0);g.lineTo(-8,7);g.fill();g.restore()}});
 const rug=new THREE.Mesh(new THREE.CircleGeometry(3.4,40),new THREE.MeshLambertMaterial({map:rugT}));rug.rotation.x=-Math.PI/2;rug.position.y=.006;rug.receiveShadow=true;scene.add(rug);
 const onWall=(a,y,r)=>{const g=new THREE.Group();g.position.set((r||7.3)*Math.sin(a),y,-(r||7.3)*Math.cos(a));g.rotation.y=-a;scene.add(g);return g};
 // neon sign
 const nc=document.createElement("canvas");nc.width=512;nc.height=160;const nt=new THREE.CanvasTexture(nc);nt.encoding=THREE.sRGBEncoding;
 const drawNeon=()=>{const g=nc.getContext("2d");g.clearRect(0,0,512,160);g.font='400 68px "Archivo Black",Archivo,Impact,sans-serif';g.textAlign="center";g.textBaseline="middle";g.lineJoin="round";
   for(let k=0;k<2;k++){g.shadowColor="#ff5a4e";g.shadowBlur=26;g.strokeStyle="#ff5a4e";g.lineWidth=7;g.strokeText("LIAR'S DICE",256,82,480)}g.shadowBlur=8;g.fillStyle="#ffe7e2";g.fillText("LIAR'S DICE",256,82,480);nt.needsUpdate=true};
 drawNeon();if(document.fonts&&document.fonts.load)document.fonts.load('68px "Archivo Black"').then(drawNeon).catch(()=>{});
 const neon=onWall(-.34,3.05);neon.add(new THREE.Mesh(new THREE.PlaneGeometry(3.1,.97),new THREE.MeshBasicMaterial({map:nt,transparent:true,toneMapped:false,fog:false})));const ng=glowSprite("#ff5a4e",4.6,2.2,.22);ng.position.z=-.05;neon.add(ng);
 // bottle shelf
 const shelf=onWall(.36,0),board=lam("#3d2217");shelf.add(part(GEO.box,board,[2.8,.05,.3],[0,2.46,.05]),part(GEO.box,board,[2.8,.05,.3],[0,3.06,.05]));
 const bg=new THREE.LatheGeometry([[0,0],[.07,0],[.07,.2],[.028,.29],[.028,.38],[0,.38]].map(p=>new THREE.Vector2(p[0],p[1])),10),im=new THREE.InstancedMesh(bg,new THREE.MeshLambertMaterial(),18),mx=new THREE.Matrix4(),cols=["#1f7a5c","#b5651d","#a02c22","#e7dcc6","#2b4bd8","#c79a4a"];
 for(let k=0;k<18;k++){mx.makeScale(1,.8+((k*37)%10)/25,1);mx.setPosition(-1.14+(k%9)*.285+Math.sin(k*7)*.03,k<9?2.485:3.085,.05);im.setMatrixAt(k,mx);im.setColorAt(k,lin(cols[(k*5)%6]))}
 im.instanceColor.needsUpdate=true;shelf.add(im);
 // wall lamps: glow only, no real lights
 for(const a of[-1.12,1.12,2.4,-2.4]){const s=onWall(a,2.7,7.25);s.add(part(GEO.sph,bas("#ffe2b0",{fog:false}),[.06,.09,.06]),glowSprite("#ffb060",1.9,1.9,.55))}
 // hanging lamp over the table
 const lamp=new THREE.Group();lamp.position.y=3.02;scene.add(lamp);
 const sg=new THREE.CylinderGeometry(.16,.64,.42,28,1,true);lamp.add(new THREE.Mesh(sg,lam("#ff5a4e")),part(sg,bas("#ffe6bd",{side:THREE.BackSide}),[.98,.98,.98]),part(GEO.sph,bas("#fff6dd"),[.09,.09,.09],[0,-.1,0]),part(GEO.cyl,bas("#0b0b12"),[.012,2.4,.012],[0,1.4,0]));
 const halo=glowSprite("#ffd9a0",1.7,1.7,.55);halo.position.y=-.12;lamp.add(halo);
 const cone=new THREE.Mesh(new THREE.ConeGeometry(2.1,2.25,28,1,true),new THREE.MeshBasicMaterial({color:lin("#ffd9a0"),transparent:true,opacity:.035,blending:THREE.AdditiveBlending,depthWrite:false,side:THREE.DoubleSide,toneMapped:false}));cone.position.y=-1.1;lamp.add(cone);
 // dust drifting through the lamp light
 const n=36,dp=new Float32Array(n*3);for(let i=0;i<n;i++){const a=Math.random()*6.28,r=Math.random()*1.7;dp[i*3]=Math.cos(a)*r;dp[i*3+1]=TY+.2+Math.random()*1.9;dp[i*3+2]=Math.sin(a)*r}
 const dg=new THREE.BufferGeometry();dg.setAttribute("position",new THREE.BufferAttribute(dp,3));dust=new THREE.Points(dg,new THREE.PointsMaterial({color:lin("#ffe2b8"),size:.022,transparent:true,opacity:.5,blending:THREE.AdditiveBlending,depthWrite:false}));scene.add(dust);
 // confetti (one reusable burst)
 const cn=110,cp=new Float32Array(cn*3),cc=new Float32Array(cn*3),cg=new THREE.BufferGeometry();cg.setAttribute("position",new THREE.BufferAttribute(cp,3));cg.setAttribute("color",new THREE.BufferAttribute(cc,3));
 confP=new THREE.Points(cg,new THREE.PointsMaterial({size:.075,vertexColors:true,transparent:true,depthWrite:false,toneMapped:false}));confP.visible=false;confP.frustumCulled=false;scene.add(confP);conf.v=new Float32Array(cn*3);conf.n=cn}

function buildTable(){
 const feltT=tex(512,512,(g,w)=>{const c=w/2;g.fillStyle="#1b3bd0";g.fillRect(0,0,w,w);const rg=g.createRadialGradient(c,c,w*.05,c,c,c);rg.addColorStop(0,"rgba(90,130,255,.4)");rg.addColorStop(.7,"rgba(30,60,200,0)");rg.addColorStop(1,"rgba(4,10,70,.6)");g.fillStyle=rg;g.fillRect(0,0,w,w);
   for(let i=0;i<7000;i++){g.fillStyle=Math.random()<.5?"rgba(255,255,255,.035)":"rgba(0,0,40,.07)";g.fillRect(Math.random()*w,Math.random()*w,2,2)}
   g.strokeStyle="rgba(255,255,255,.42)";g.lineWidth=3;g.beginPath();g.arc(c,c,c*.9,0,7);g.stroke();g.lineWidth=1.5;g.strokeStyle="rgba(255,255,255,.22)";g.beginPath();g.arc(c,c,c*.86,0,7);g.stroke();
   g.strokeStyle="rgba(255,255,255,.34)";g.lineWidth=2.5;g.beginPath();g.arc(c,c,c*.3,0,7);g.stroke();
   for(let i=0;i<24;i++){const a=i/24*Math.PI*2,r1=c*.3,r2=c*(i%6===0?.37:.33);g.beginPath();g.moveTo(c+Math.cos(a)*r1,c+Math.sin(a)*r1);g.lineTo(c+Math.cos(a)*r2,c+Math.sin(a)*r2);g.stroke()}});
 const top=new THREE.Mesh(new THREE.CylinderGeometry(1.75,1.75,.08,56),[std("#3a1c12",.6),new THREE.MeshStandardMaterial({map:feltT,roughness:.95}),std("#24110b",.8)]);top.position.y=TY-.04;top.receiveShadow=true;scene.add(top);
 const rail=part(new THREE.TorusGeometry(1.79,.085,12,64),std("#4a2414",.42),null,[0,TY+.005,0],[Math.PI/2,0,0]);rail.receiveShadow=true;scene.add(rail,part(new THREE.TorusGeometry(1.67,.011,6,72),MAT.brass,null,[0,TY+.003,0],[Math.PI/2,0,0]));
 scene.add(part(GEO.cyl,MAT.wood,[.2,TY-.1,.2],[0,(TY-.1)/2,0]),part(new THREE.CylinderGeometry(.68,.8,.08,28),MAT.wood,null,[0,.04,0]));
 // pool of light in front of whoever's turn it is
 turnArc=new THREE.Group();arcMat=new THREE.MeshBasicMaterial({color:lin("#ffe9c4"),transparent:true,opacity:0,blending:THREE.AdditiveBlending,depthWrite:false,toneMapped:false});
 const arc=new THREE.Mesh(new THREE.RingGeometry(.98,1.62,24,1,-.42,.84),arcMat);arc.rotation.x=-Math.PI/2;arc.position.y=TY+.004;turnArc.add(arc);turnArc.rotation.y=-SA[0];scene.add(turnArc);
 // the claim marker: a big die tilted toward you, with the quantity beside it
 bidG=new THREE.Group();const tilt=new THREE.Group();tilt.rotation.x=.95;bidDie=makeDie(.3);tilt.add(bidDie);bidG.add(tilt);bidG.visible=false;scene.add(bidG);
 bidGlow=new THREE.Mesh(new THREE.CircleGeometry(.5,28),new THREE.MeshBasicMaterial({map:MAT.glowTex,color:lin("#ffe9c4"),transparent:true,opacity:.3,blending:THREE.AdditiveBlending,depthWrite:false,toneMapped:false}));bidGlow.rotation.x=-Math.PI/2;bidGlow.position.set(.05,TY+.005,0);bidGlow.visible=false;scene.add(bidGlow);
 qc=document.createElement("canvas");qc.width=256;qc.height=128;qtx=new THREE.CanvasTexture(qc);qtx.encoding=THREE.sRGBEncoding;
 qtySp=new THREE.Sprite(new THREE.SpriteMaterial({map:qtx,transparent:true,depthTest:false,toneMapped:false}));qtySp.scale.set(.7,.35,1);qtySp.position.set(-.34,TY+.37,.05);qtySp.renderOrder=5;qtySp.visible=false;scene.add(qtySp)}
function drawQty(q){const g=qc.getContext("2d");g.clearRect(0,0,256,128);g.font='400 92px "Archivo Black",Archivo,Impact,sans-serif';g.textAlign="right";g.textBaseline="middle";g.lineJoin="round";g.lineWidth=12;g.strokeStyle="#0b1440";g.strokeText(q+" ×",248,68);g.fillStyle="#fbfaf7";g.fillText(q+" ×",248,68);qtx.needsUpdate=true}

// ---------- characters: wooden game pawns with painted faces and floating gloves ----------
const WOOD={cfr:"#ecc9a0",dqn:"#c08a5c",ppo:"#f1d9b5",nfsp:"#d7a274",ismcts:"#e4bd88",mc:"#a86c46"};
function makeAvatar(n){
 const robot=n==="math"||n==="random",g=new THREE.Group(),bodyG=new THREE.Group(),headG=new THREE.Group();
 const cBody=std(COL[n],robot?.36:.42,robot?.55:.05),cDark=std(shade(COL[n],.5),.55);
 const u={name:n,bodyG,headG,bodyM:cBody,base:cBody.color.clone(),top:2.2,roll0:0,smY:0,omy:1};
 // chair
 g.add(part(GEO.cyl,MAT.wood,[.4,.06,.4],[0,.49,-.02]),part(GEO.cyl,MAT.wood,[.06,.5,.06],[0,.25,-.02]),part(GEO.box,MAT.wood,[.8,1.25,.08],[0,1.12,-.42]),part(GEO.cyl,MAT.wood,[.4,.08,.4],[0,1.745,-.42],[Math.PI/2,0,0]),part(GEO.box,MAT.cush,[.62,.86,.04],[0,1.22,-.37]));
 bodyG.position.y=.52;const body=part(GEO.body,cBody);body.castShadow=true;bodyG.add(body);headG.position.y=.98;bodyG.add(headG);g.add(bodyG);
 let on,eyes,brows,smile,om,my=-.092,by0=.122;
 if(!robot){
  const HR=.27;on=(x,y,pad)=>[x,.27+y,.02+Math.sqrt(Math.max(0,HR*HR-x*x-y*y))+(pad||0)];
  const head=part(GEO.head,std(WOOD[n],.5),null,[0,.27,.02]);head.castShadow=true;headG.add(head);
  const hair=std(n==="mc"||n==="cfr"?"#d9d9df":"#1b1420",.7);
  if(n==="cfr"){u.top=2.16;const tie=std("#162058",.5),cone=new THREE.ConeGeometry(.055,.11,10);
   for(const sx of[-1,1]){headG.add(part(GEO.lens,MAT.brass,null,on(sx*.095,.035,.032)),part(GEO.sph,hair,[.09,.11,.1],[sx*.24,.31,-.03]));bodyG.add(part(cone,tie,null,[sx*.06,.9,.235],[0,0,sx*Math.PI/2]))}
   headG.add(part(GEO.box,MAT.brass,[.07,.008,.008],on(0,.04,.032)),part(GEO.sph,hair,[.2,.15,.14],[0,.24,-.17]));bodyG.add(part(GEO.sph,tie,[.028,.028,.028],[0,.9,.24]))}
  if(n==="ismcts"){u.top=2.5;my=-.122;const hat=new THREE.Group(),blk=std("#15131c",.45);hat.position.set(0,.5,0);hat.rotation.z=-.1;const crown=part(GEO.cyl,blk,[.195,.32,.195],[0,.17,0]);crown.castShadow=true;
   hat.add(part(GEO.cyl,blk,[.31,.022,.31]),crown,part(GEO.cyl,std("#ff5a4e",.5),[.2,.06,.2],[0,.045,0]));headG.add(hat,part(GEO.lens,MAT.brass,null,on(.095,.035,.032)));
   for(const sx of[-1,1])headG.add(part(GEO.sph,MAT.ink,[.062,.022,.02],on(sx*.05,-.062,.006),[0,0,sx*.28]))}
  if(n==="ppo"){u.top=2.26;by0=.1;const cap=std("#0a6e55",.6);headG.add(part(GEO.hemi,cap,[.252,.2,.252],[0,.39,.02]),part(new THREE.CylinderGeometry(.2,.2,.018,18,1,false,-Math.PI/2,Math.PI),cap,null,[0,.4,.2],[-.16,0,0]),part(GEO.sph,cap,[.03,.02,.03],[0,.59,.02]));
   bodyG.add(part(new THREE.TorusGeometry(.2,.062,8,20),cDark,null,[0,.9,0],[Math.PI/2,0,0]))}
  if(n==="dqn"){u.top=2.16;const lens=std("#1a0f24",.12,.5);headG.add(part(GEO.box,MAT.ink,[.05,.012,.012],on(0,.05,.03)),part(GEO.hemi,hair,[.283,.2,.292],[0,.34,0],[-.28,0,0]));
   for(const sx of[-1,1])headG.add(part(GEO.sph,lens,[.075,.052,.02],on(sx*.095,.035,.022)));
   for(const sx of[-1,1])headG.add(part(GEO.box,MAT.ink,[.012,.02,.2],[sx*.155,.305,.17]));
   bodyG.add(part(new THREE.CylinderGeometry(.26,.2,.13,18,1,true),std("#f7f3ea",.6,0,{side:THREE.DoubleSide}),null,[0,.93,0]))}
  if(n==="nfsp"){u.top=2.2;const sil=std("#d9dce6",.3,.6);headG.add(part(new THREE.TorusGeometry(.287,.022,6,20,Math.PI),sil,null,[0,.27,.02]),
    part(new THREE.SphereGeometry(.283,12,8,Math.PI/2,Math.PI,0,Math.PI*.3),cDark,null,[0,.27,.02],[-.2,0,0]),part(new THREE.SphereGeometry(.283,12,8,-Math.PI/2,Math.PI,0,Math.PI*.3),sil,null,[0,.27,.02],[-.2,0,0]));
   for(const sx of[-1,1])headG.add(part(GEO.cyl,sx<0?cDark:sil,[.085,.06,.085],[sx*.28,.27,.02],[0,0,Math.PI/2]))}
  if(n==="mc"){u.top=2.32;my=-.128;const hat=std("#23262f",.5);const dome=part(GEO.hemi,hat,[.235,.21,.235],[0,.47,.01]);dome.castShadow=true;headG.add(dome,part(GEO.cyl,hat,[.33,.022,.33],[0,.47,.01]),part(GEO.cyl,cBody,[.238,.045,.238],[0,.495,.01]));
   for(const sx of[-1,1])headG.add(part(GEO.sph,hair,[.078,.03,.025],on(sx*.062,-.066,.008),[0,0,-sx*.42]));
   for(let k=0;k<3;k++)bodyG.add(part(GEO.sph,MAT.brass,[.022,.022,.022],[0,.72-k*.14,.272+k*.024]))}
  eyes=[-1,1].map(sx=>{const e=part(GEO.eye,MAT.ink,[1,1.15,.55],on(sx*.095,.035,-.004));e.userData.sy=1.15;e.add(part(GEO.dot,MAT.white,null,[.012,.014,.034]));headG.add(e);return e});
  brows=[-1,1].map(sx=>{const b=part(GEO.box,hair,[.085,.02,.02],on(sx*.095,by0,.004));b.userData={sx,y0:b.position.y};headG.add(b);return b});
  smile=part(GEO.smile,MAT.ink,null,on(0,my,.006),[0,0,Math.PI]);om=part(GEO.sph,MAT.mouth,[.045,.036,.016],on(0,my-.02,-.002));u.omy=.036;
  for(const sx of[-1,1])headG.add(part(GEO.sph,MAT.cheek,[.04,.03,.008],on(sx*.165,-.05,0)));
 }else{
  on=(x,y,pad)=>[x,.25+y,.236+(pad||0)];const led=bas(n==="math"?"#7df9ff":"#ffd84a");
  const head=part(GEO.box,std(n==="math"?"#aeb6c8":"#e3e6ef",.35,.55),[.5,.42,.42],[0,.25,.02]);head.castShadow=true;headG.add(head,part(GEO.box,std("#0b1030",.25),[.42,.3,.02],[0,.25,.226]));
  for(const sx of[-1,1])headG.add(part(GEO.cyl,MAT.brass,[.05,.04,.05],[sx*.265,.25,.02],[0,0,Math.PI/2]));
  eyes=[-1,1].map(sx=>{const sy=n==="random"&&sx<0?.13:.085,e=part(GEO.box,led,[.075,sy,.01],on(sx*.1,.03));e.userData={sy,sx,y0:e.position.y};headG.add(e);return e});brows=eyes;
  smile=part(GEO.box,led,[.14,.018,.01],on(0,-.085));om=part(GEO.box,led,[.1,.06,.01],on(0,-.085));u.omy=.06;
  if(n==="math"){u.top=2.36;headG.add(part(GEO.cyl,MAT.ink,[.012,.18,.012],[0,.55,.02]));u.bulb=part(GEO.sph,bas("#ff5a4e"),[.035,.035,.035],[0,.66,.02]);headG.add(u.bulb);
   bodyG.add(part(GEO.box,std("#0b1030",.3),[.22,.14,.02],[0,.5,.315],[-.13,0,0]));["#7df9ff","#ffd84a","#ff5a4e"].forEach((c,k)=>bodyG.add(part(GEO.sph,bas(c),[.02,.02,.01],[-.06+k*.06,.5,.33])))}
  else{u.top=2.5;u.roll0=.09;const ht=tex(64,64,(g,w)=>{for(let k=0;k<8;k++){g.fillStyle=k%2?"#ff5a4e":"#ffd84a";g.fillRect(0,k*8,w,8)}});
   headG.add(part(new THREE.ConeGeometry(.13,.34,16),new THREE.MeshStandardMaterial({map:ht,roughness:.6}),null,[.06,.62,.02],[0,0,-.16]),part(GEO.sph,MAT.glove,[.04,.04,.04],[.088,.8,.02]));
   const qt=tex(64,64,(g,w)=>{g.fillStyle="#0b1030";g.fillRect(0,0,w,w);g.fillStyle="#ffd84a";g.font="700 52px Archivo,sans-serif";g.textAlign="center";g.textBaseline="middle";g.fillText("?",32,35)});
   bodyG.add(part(GEO.box,new THREE.MeshBasicMaterial({map:qt,toneMapped:false}),[.18,.18,.02],[0,.5,.315],[-.13,0,0]))}
 }
 om.visible=false;headG.add(smile,om);u.smY=smile.position.y;u.smx=smile.scale.x;
 const hm=robot?std("#565d70",.4,.7):MAT.glove;u.hL=part(GEO.hand,hm,[1,.72,1.15],[-.27,TY+.055,.6]);u.hR=part(GEO.hand,hm,[1,.72,1.15],[.27,TY+.055,.6]);u.hL.castShadow=u.hR.castShadow=true;g.add(u.hL,u.hR);
 u.cup=makeCup();u.cup.position.set(0,TY,.82);g.add(u.cup);
 u.eyes=eyes;u.brows=brows;u.smile=smile;u.om=om;g.userData=u;resetAvatar(u);return g}
function resetAvatar(u){u.mood="neutral";u.cheer=false;u.hold=true;u.hop=0;u.kick=0;u.gL=null;u.talkUntil=0;u.blinkAt=1+Math.random()*3;u.blinkEnd=0;u.cur={yaw:0,p:0,r:0,l:0,b:0,by:0};u.bodyM.color.copy(u.base)}
const MOOD={neutral:{b:0,by:0,p:.04,r:0,l:0,m:1},think:{b:-.1,by:.012,p:-.1,r:.16,l:.03,m:0},angry:{b:.5,by:-.012,p:.06,r:0,l:.17,m:-1},sad:{b:-.45,by:0,p:.36,r:-.07,l:-.05,m:-1},happy:{b:-.12,by:.022,p:-.14,r:0,l:.02,m:1.35},out:{b:-.35,by:0,p:.5,r:.16,l:-.13,m:-1}};
function updAvatar(s,i,t,dt,now){const u=s.u,out=!!G&&G.counts[i]===0,M=MOOD[out?"out":u.mood],c=u.cur,k=Math.min(1,dt*7);
 let yawT=0,px=0;const f=G?(G.focus!=null?G.focus:(G.phase==="play"?G.turn:-1)):-1;
 if(f>=0&&f!==i&&!out){const tx=f===0?cam.position.x:seats[f].g.position.x,tz=f===0?cam.position.z:seats[f].g.position.z,dx=tx-s.g.position.x,dz=tz-s.g.position.z,ph=s.g.rotation.y;
  yawT=Math.max(-.78,Math.min(.78,Math.atan2(dx*Math.cos(ph)-dz*Math.sin(ph),dx*Math.sin(ph)+dz*Math.cos(ph))))}else if(f===i)px=.14;
 c.yaw+=(yawT-c.yaw)*k;c.p+=(M.p+px-c.p)*k;c.r+=(M.r-c.r)*k;c.l+=(M.l+u.kick-c.l)*k;c.b+=(M.b-c.b)*k;c.by+=(M.by-c.by)*k;u.kick*=1-Math.min(1,dt*4);
 u.bodyG.rotation.set(c.l,c.yaw*.25,0);u.bodyG.position.y=.52+u.hop+(u.cheer?Math.abs(Math.sin(t*7))*.07:0);u.bodyG.scale.y=1+(out?0:Math.sin(t*1.7+i*2)*.006);
 u.headG.rotation.set(c.p+(u.mood==="think"?Math.sin(t*1.3)*.035:0),c.yaw*.75,c.r+u.roll0);
 if(t>u.blinkAt){u.blinkEnd=t+.13;u.blinkAt=t+2+Math.random()*4}
 const closed=out||t<u.blinkEnd;for(const e of u.eyes)e.scale.y=e.userData.sy*(closed?.12:1);
 for(const b of u.brows){b.rotation.z=b.userData.sx*c.b;b.position.y=b.userData.y0+c.by+(u.mood==="think"&&b.userData.sx>0?.014:0)}
 const talk=now<u.talkUntil;u.om.visible=talk||M.m===0;u.smile.visible=!u.om.visible;
 if(talk)u.om.scale.y=u.omy*(.35+.65*Math.abs(Math.sin(t*17)));else if(M.m===0)u.om.scale.y=u.omy*.6;
 else{const sc=Math.abs(M.m);u.smile.rotation.z=M.m>0?Math.PI:0;u.smile.scale.x=u.smx*sc;u.smile.position.y=u.smY-(M.m<0?.045:0)}
 if(u.bulb)u.bulb.visible=out?false:(t*1.4+i)%1<.6;
 // gloves: left one gestures, right one rests on the cup
 let lx=-.27,ly=TY+.055,lz=.6,rx=.27,ry=TY+.055,rz=.6;const cu=u.cup;
 if(out){lx=-.2;rx=.2;ly=ry=.58;lz=rz=.32}
 else if(u.cheer){lx=-.44;rx=.44;lz=rz=.2;ly=1.72+Math.sin(t*9)*.09;ry=1.72+Math.cos(t*9)*.09}
 else{if(u.hold&&cu.visible){rx=cu.position.x;ry=cu.position.y+.39*Math.cos(cu.rotation.x);rz=cu.position.z+.39*Math.sin(cu.rotation.x)}
  if(u.mood==="think")ly+=Math.max(0,Math.sin(t*11))*.035;
  if(u.gL&&now<u.gL.until){lx=u.gL.x;ly=u.gL.y;lz=u.gL.z}}
 const kh=Math.min(1,dt*11),L=u.hL.position,R=u.hR.position;L.x+=(lx-L.x)*kh;L.y+=(ly-L.y)*kh;L.z+=(lz-L.z)*kh;R.x+=(rx-R.x)*kh;R.y+=(ry-R.y)*kh;R.z+=(rz-R.z)*kh}
function gest(i,type,toward){const s=seats[i];if(!s||!s.u)return;const u=s.u,now=performance.now();
 if(type==="bid"){u.talkUntil=now+650;u.kick=.12;u.gL={x:-.12,y:TY+.2,z:.98,until:now+520}}
 if(type==="liar"){u.mood="angry";u.talkUntil=now+800;u.kick=.2;let x=0,z=1;if(toward>0){const p=seats[toward].g.position,dx=p.x-s.g.position.x,dz=p.z-s.g.position.z,ph=s.g.rotation.y,lx=dx*Math.cos(ph)-dz*Math.sin(ph),lz=dx*Math.sin(ph)+dz*Math.cos(ph),d=Math.hypot(lx,lz)||1;x=lx/d;z=lz/d}
  u.gL={x:x*.62,y:TY+.5,z:Math.max(.25,z*.62+.2),until:now+1700}}
 if(type==="hop")anim(480,k=>{u.hop=Math.sin(k*Math.PI)*.12},k=>k)}
function setMood(i,m){const s=seats[i];if(s&&s.u&&G&&G.counts[i]>0)s.u.mood=m}

// ---------- frame loop ----------
function frame(){
 const dt=Math.min(.05,clock.getDelta()),t=clock.elapsedTime,now=performance.now();
 if(!lowQ&&!window.LD_NOADAPT&&t>2.5){fpsN++;fpsT+=dt;if(fpsN===80){if(fpsT/80>.034){lowQ=true;R3.setPixelRatio(1);spotL.castShadow=false;R3.setSize(W,H,false)}fpsN=0;fpsT=0}}
 for(let i=anims.length-1;i>=0;i--){const a=anims[i],k=Math.min(1,(now-a.t0)/a.dur);a.f(a.e(k));if(k>=1){anims.splice(i,1);a.res()}}
 if(Math.abs(liftT-lift)>.5){lift+=(liftT-lift)*Math.min(1,dt*5);cam.setViewOffset(W,H,0,lift,W,H);cam.updateProjectionMatrix()}
 const port=cam.aspect<1;look.x+=(look.tx-look.x)*.06;look.y+=(look.ty-look.y)*.06;zoom+=(zoomT-zoom)*Math.min(1,dt*3);
 fit+=(fitT-fit)*Math.min(1,dt*2.5);cam.position.set(Math.sin(look.x)*.3+(reduce?0:Math.sin(t*.45)*.025),(port?2.55:2.05)-look.y*.4-fit*.1,(port?4.6:3.75)-zoom*.15-fit*(port?.35:.5));
 const fs=G&&G.phase==="play"?G.turn:-1;tgtT.set(0,port?.85:.8,-.5);if(fs>0){tgtT.x+=Math.cos(SA[fs])*SEAT_R*.16;tgtT.z+=Math.sin(SA[fs])*SEAT_R*.07}
 tgt.lerp(tgtT,Math.min(1,dt*2.5));cam.lookAt(tgt.x+look.x*1.2,tgt.y-look.y,tgt.z);
 for(let i=1;i<seats.length;i++)if(seats[i].u)updAvatar(seats[i],i,t,dt,now);
 // turn light
 if(fs>=0){let d=-SA[fs]-turnArc.rotation.y;d=((d+Math.PI)%(Math.PI*2)+Math.PI*2)%(Math.PI*2)-Math.PI;turnArc.rotation.y+=d*Math.min(1,dt*6)}
 arcMat.opacity+=((fs>=0?.2+Math.sin(t*4)*.07:0)-arcMat.opacity)*Math.min(1,dt*6);
 if(bidG.visible&&!bidG.userData.fly)bidG.position.y=TY+.37+Math.sin(t*2)*.018;
 if(!reduce){const p=dust.geometry.attributes.position;for(let i=0;i<p.count;i++){let y=p.getY(i)+dt*.035;if(y>TY+2.1)y=TY+.2;p.setY(i,y)}p.needsUpdate=true}
 if(conf.life>0){conf.life-=dt;const p=confP.geometry.attributes.position.array,v=conf.v;for(let i=0;i<conf.n*3;i+=3){v[i+1]-=dt*2.6;p[i]+=v[i]*dt;p[i+1]=Math.max(.02,p[i+1]+v[i+1]*dt);p[i+2]+=v[i+2]*dt}
  confP.geometry.attributes.position.needsUpdate=true;confP.material.opacity=Math.min(1,conf.life);if(conf.life<=0)confP.visible=false}
 R3.render(scene,cam);overlays()}
function measure(){if(!cam)return;const port=W<H;cam.position.set(0,port?2.55:2.05,port?4.6:3.75);cam.lookAt(0,port?.85:.8,-.5);cam.clearViewOffset();cam.updateProjectionMatrix();cam.updateMatrixWorld();cam.matrixWorldInverse.copy(cam.matrixWorld).invert();
 const ry=(-pv.set(0,TY-.03,1.5).project(cam).y*.5+.5)*H;cam.setViewOffset(W,H,0,lift,W,H);cam.updateProjectionMatrix();
 const dh=$("dock").offsetHeight;fitT=Math.max(0,Math.min(1,1-(dh-95)/(W<H?190:125)));liftT=Math.max(0,Math.min(H*.25,dh+26-(H-ry)))}
function diePos(i,j,n,d){const sz=d.userData.size,off=j-(n-1)/2,a=SA[i];if(i===0)d.position.set(off*sz*1.28,sz/2,.015);else{const sp=sz*1.16;d.position.set(Math.cos(a)*CUP_R-Math.sin(a)*off*sp,TY+sz/2,Math.sin(a)*CUP_R+Math.cos(a)*off*sp)}}
function posDice(i){const s=seats[i];if(!s||!i||!G)return;const n=G.counts[i];s.dice.forEach((d,j)=>{if(j<n)diePos(i,j,n,d)})}
function sizeTable(N){const ys=N<=2?.25:N===3?.22:.19,as=N<=2?.18:.16;dicePool.forEach((a,i)=>a.forEach(d=>{d.userData.size=i?as:ys}));
 const w=Math.max(.82,N*ys*1.28+.2),cx=w/2+.27;rackParts[0].scale.x=rackParts[1].scale.x=w;rackParts[2].scale.x=w-.08;youX=-cx;youZ=SEAT_R-Math.min(1.3,Math.sqrt(Math.max(.2,1.42*1.42-cx*cx)))}
function setSide(a){if(SA[3]===-a)return;SA[1]=Math.PI+a;SA[3]=-a;seats.forEach((s,i)=>{if(!s.g)return;const p=seatPos(i,SEAT_R);s.g.position.copy(p);s.g.rotation.y=Math.atan2(-p.x,-p.z);posDice(i)})}
function confetti(x,y,z){if(reduce)return;const p=confP.geometry.attributes.position.array,c=confP.geometry.attributes.color.array,cols=["#ff5a4e","#ffd84a","#fbfaf7","#7df9ff","#13b38a"].map(lin);
 for(let i=0;i<conf.n;i++){p[i*3]=x;p[i*3+1]=y;p[i*3+2]=z;const a=Math.random()*6.28,s=.6+Math.random()*1.6;conf.v[i*3]=Math.cos(a)*s;conf.v[i*3+1]=1.2+Math.random()*2.2;conf.v[i*3+2]=Math.sin(a)*s;const k=cols[i%5];c[i*3]=k.r;c[i*3+1]=k.g;c[i*3+2]=k.b}
 confP.geometry.attributes.color.needsUpdate=true;conf.life=2.8;confP.visible=true}
function place(el,x,y){const k=x.toFixed(1)+","+y.toFixed(1);if(el._k!==k){el._k=k;el.style.transform="translate("+x.toFixed(1)+"px,"+y.toFixed(1)+"px) translate(-50%,-100%)"}}
function overlays(){for(let i=0;i<seats.length;i++){const s=seats[i];if(i===0)pv.set(-.66,TY+.36,1.3);else pv.set(s.g.position.x,s.u.top,s.g.position.z);pv.project(cam);
  const hw=Math.max(78,(s.tw||0)+8),x=Math.max(hw,Math.min(W-hw,(pv.x*.5+.5)*W)),y=Math.max(i===2?(W<640?104:84):i?122:60,(-pv.y*.5+.5)*H);if(s.tag)place(s.tag,x,y);place(s.bub,x,y-(i?34:0))}}

// ---------- game flow ----------
let pick=store.get("pick",null);if(!Array.isArray(pick)||pick.length!==3||new Set(pick).size!==3||!pick.every(n=>NAMES[n]))pick=["ppo","ismcts","cfr"];
let qS=1,fS=1,qMore=0,busy=false,looping=false;
const roll=n=>Array.from({length:n},()=>1+Math.floor(Math.random()*6)).sort();
const who=i=>i===0?"You":NAMES[seats[i].name];
const active=()=>[0,1,2,3].filter(s=>G.counts[s]>0);
const nextSeat=s=>{const a=active();return a[(a.indexOf(s)+1)%a.length]};
const myTurn=()=>!!G&&G.phase==="play"&&G.turn===0&&!G.wait;
const obs=s=>{const tot=sum(G.counts);return[G.hands[s],G.counts[s],tot-G.counts[s],active().length-1,G.last,G.prev]};
function log(h,cls){const li=document.createElement("li");if(cls)li.className=cls;li.innerHTML=h;$("log4").prepend(li)}
function setMsg(h){$("msg4").innerHTML=h;if(R3)measure()}
function bubble(i,html,cls){const b=seats[i].bub;b.className="bubble "+(cls||"");b.innerHTML='<span class="bb">'+(cls==="think"?"<i></i><i></i><i></i>":html)+"</span>";b.hidden=false}
function verdict(big,small,cls){const v=$("verdict");v.className="verdict "+(cls||"");v.innerHTML="<b>"+big+"</b><span>"+small+"</span>";v.hidden=false;measure()}

function seatTable(){
 for(const s of seats)if(s.g)scene.remove(s.g);$("overlay").innerHTML="";dicePool.forEach(a=>a.forEach(d=>{d.visible=false}));bidG.visible=bidGlow.visible=qtySp.visible=false;
 seats=[0,1,2,3].map(i=>{const s={i,dice:dicePool[i]};
  if(i===0){s.name="you";s.g=youG;s.cup=youCup;s.u=null}else{const n=pick[i-1];s.name=n;s.g=AV[n]||(AV[n]=makeAvatar(n));s.u=s.g.userData;s.cup=s.u.cup;resetAvatar(s.u)}
  s.cup.visible=true;s.cup.position.set(i?0:youX,TY,i?.82:youZ);s.cup.rotation.set(0,0,0);const p=seatPos(i,SEAT_R);s.g.position.copy(p);s.g.rotation.y=Math.atan2(-p.x,-p.z);scene.add(s.g);
  s.bub=document.createElement("div");s.bub.className="bubble";s.bub.hidden=true;
  if(i>0){s.tag=document.createElement("div");s.tag.className="tag";s.tag.innerHTML='<i style="background:'+COL[s.name]+'"></i><b>'+esc(NAMES[s.name])+'</b><span class="pips"></span>';$("overlay").appendChild(s.tag)}
  $("overlay").appendChild(s.bub);return s})}
function newGame(){sizeTable(DICE);G={start:DICE,taught:false,counts:[DICE,DICE,DICE,DICE],starter:Math.floor(Math.random()*4),phase:"idle",round:0,hist:[],last:-1,prev:-1,lastBy:-1,turn:-1,focus:null,wait:false,hands:[[],[],[],[]]};
 $("log4").innerHTML="";seats.forEach(s=>{if(s.u)resetAvatar(s.u)});conf.life=0;confP.visible=false;newRound()}
function placeDice(i){const s=seats[i],n=G.counts[i],a=SA[i];
 s.dice.forEach((d,j)=>{d.visible=false;d.children[0].material=DM;d.scale.setScalar(d.userData.size);if(j>=n)return;setFace(d,G.hands[i][j]);diePos(i,j,n,d);
  d.rotation.y=i===0?(Math.random()-.5)*.16:-a+Math.PI/2+(Math.random()-.5)*.4})}
async function newRound(){const g=G;G.phase="shake";G.round++;G.hands=G.counts.map(roll);G.last=G.prev=-1;G.hist=[];G.lastBy=-1;G.turn=G.starter;G.focus=null;G.wait=false;
 $("verdict").hidden=true;zoomT=0;bidG.visible=bidGlow.visible=qtySp.visible=false;
 seats.forEach((s,i)=>{s.bub.hidden=true;const alive=G.counts[i]>0;s.cup.visible=alive;s.cup.position.set(i?0:youX,TY,i?.82:youZ);s.cup.rotation.set(0,0,0);placeDice(i);if(s.u){s.u.mood="neutral";s.u.hold=alive;s.u.gL=null}});
 log("Round "+G.round+" · "+sum(G.counts)+" dice in play","hd");setMsg("New round. Everyone shakes their cup…");render();SFX.rattle();
 seats.forEach((s,i)=>{if(G.counts[i]>0){const c=s.cup,z0=c.position.z,x0=c.position.x;anim(750,k=>{const a=Math.sin(k*Math.PI);c.position.x=x0+Math.sin(k*42+i)*.035*a;c.position.y=TY+Math.abs(Math.sin(k*27))*.05*a;c.position.z=z0;c.rotation.z=Math.sin(k*35)*.12*a},k=>k)}});
 await sleep(850);if(G!==g)return;
 if(G.counts[0]>0){SFX.cup();const c=youCup;anim(460,k=>{const a=Math.sin(k*Math.PI);c.rotation.z=-a*.85;c.position.y=TY+a*.16;c.position.x=youX+a*.14},k=>k);
  seats[0].dice.forEach((d,j)=>{if(j<G.counts[0]){d.visible=true;const sz=d.userData.size;anim(320+j*90,k=>d.scale.setScalar(sz*Math.max(.01,k)),k=>1-Math.pow(1-k,2))}});await sleep(480);if(G!==g)return}
 G.phase="play";loop()}
function decide(s){const tot=sum(G.counts),o=obs(s),acts=legal(G.last,tot),order=[];let x=s;do{order.push(x);x=nextSeat(x)}while(x!==s);
 let a;try{a=policyOf(seats[s].name,G.start)(o,acts,{order,total:tot,hist:G.hist,counts:G.counts})}catch(e){a=mathBot(o,acts)}return acts.includes(a)?a:mathBot(o,acts)}
async function loop(){if(looping)return;looping=true;const g=G;
 try{while(G===g&&G.phase==="play"&&G.turn!==0){const s=G.turn;setMood(s,"think");bubble(s,"","think");setMsg("<b>"+esc(who(s))+"</b> is thinking…");render();
   await sleep(950+Math.random()*450);if(G!==g||G.phase!=="play")return;const a=decide(s);setMood(s,"neutral");
   if(a===CALL){await reveal(s);return}await doBid(s,a)}
  if(G===g&&G.phase==="play"&&G.turn===0){SFX.turn();promptMe();render()}}finally{looping=false}}
function flyBid(s,a){const p=seatPos(s,s?1.05:.95),ex=.2,ey=TY+.37;drawQty(bq(a));setFace(bidDie,bf(a));bidG.visible=true;bidGlow.visible=true;qtySp.visible=false;bidG.userData.fly=true;
 anim(460,k=>{bidG.position.set(p.x+(ex-p.x)*k,ey+Math.sin(k*Math.PI)*.38,p.z*(1-k));bidG.rotation.y=(1-k)*5;bidG.scale.setScalar(.45+.55*k)}).then(()=>{bidG.userData.fly=false;qtySp.visible=bidG.visible;qtySp.scale.set(.7,.35,1)})}
async function doBid(s,a){const g=G;G.prev=G.last;G.last=a;G.lastBy=s;G.hist.push([s,a]);G.wait=true;
 seats.forEach((x,i)=>{if(i!==s&&!x.bub.hidden)x.bub.classList.add("past")});bubble(s,bq(a)+" × "+pip(bf(a)));
 log("<span>"+esc(who(s))+" "+(s===0?"claim":"claims")+"</span><b>"+fw(bq(a),bf(a))+"</b>");gest(s,"bid");SFX.bid();flyBid(s,a);G.turn=nextSeat(s);G.focus=s;
 setMsg("<b>"+esc(who(s))+"</b> "+(s===0?"claim":"claims")+" at least "+fw(bq(a),bf(a))+".");render();
 await sleep(650);if(G!==g)return;G.focus=null;G.wait=false;render()}
function hop(d){const y0=d.position.y;anim(330,k=>{d.position.y=y0+Math.sin(k*Math.PI)*.17},k=>k)}
async function reveal(s){const g=G,b=G.lastBy,q=bq(G.last),f=bf(G.last);G.phase="reveal";G.focus=b;G.wait=false;zoomT=1;
 seats.forEach((x,i)=>{if(i!==s&&!x.bub.hidden)x.bub.classList.add("past")});bubble(s,"Liar!","liar");gest(s,"liar",b);SFX.liar();
 log("<span>"+esc(who(s))+" "+(s===0?"call":"calls")+" Liar! on "+(b===0?"you":esc(who(b)))+"</span>","hd");
 setMsg("<b>"+esc(who(s))+"</b> "+(s===0?"call":"calls")+" Liar! on "+(b===0?"your":esc(who(b))+"'s")+" claim of at least "+fw(q,f)+". Cups up…");
 verdict("Counting "+FW[f]+"…","0 found · the claim needs "+q);render();await sleep(1100);if(G!==g)return;
 let n=0;
 for(const i of active()){const x=seats[i];
  if(i>0){SFX.cup();const c=x.cup;anim(420,k=>{c.position.y=TY+.26*k;c.position.z=.82-.36*k;c.rotation.x=-.95*k});x.dice.forEach((d,j)=>{if(j<G.counts[i]){d.visible=true;const sz=d.userData.size;anim(260+j*40,k=>d.scale.setScalar(sz*(.3+.7*k)))}})}
  await sleep(320);if(G!==g)return;
  for(let j=0;j<G.counts[i];j++){const d=x.dice[j];if(d.userData.v===f){n++;d.children[0].material=DMG;hop(d);SFX.tick(n);verdict("Counting "+FW[f]+"…",n+" found · the claim needs "+q);await sleep(340);if(G!==g)return}}}
 seats.forEach((x,i)=>x.dice.forEach((d,j)=>{if(j<G.counts[i]&&d.userData.v!==f)d.children[0].material=DMD}));
 await sleep(380);if(G!==g)return;
 const tru=n>=q,loser=tru?s:b,winner=tru?b:s;
 verdict(tru?"True!":"False!",(tru?fw(n,f)+" on the table, enough for the claim of "+q+". ":(n?"Only "+fw(n,f):"No "+FW[f])+" on the table, the claim needed "+q+". ")+
  (tru?(s===0?"Your call was wrong: you lose a die.":esc(who(s))+" called wrong and loses a die."):(b===0?"Your bluff is caught: you lose a die.":esc(who(b))+" is caught and loses a die.")),tru?"t":"f");
 log("<span>"+(n?fw(n,f):"no "+FW[f])+" showing, "+(tru?"claim true":"claim false")+"</span><b>"+(loser===0?"you":esc(who(loser)))+" −1</b>");
 G.focus=loser;setMood(loser,"sad");setMood(winner,"happy");gest(winner,"hop");if(loser===0)SFX.lose();else if(winner===0)SFX.good();else SFX.bid();
 const ld=seats[loser].dice.slice(0,G.counts[loser]),dd=ld.find(d=>d.userData.v!==f)||ld[0],y0=dd.position.y,sz=dd.userData.size;
 anim(800,k=>{dd.position.y=y0+k*.9;dd.rotation.y+=.25;dd.scale.setScalar(sz*Math.max(.01,1-k))},k=>k*k).then(()=>{dd.visible=false});
 G.counts[loser]--;G.starter=loser;
 if(G.counts[loser]===0){log("<span>"+esc(who(loser))+" "+(loser===0?"are":"is")+" out of dice</span>","hd");if(loser>0){const u=seats[loser].u;u.hold=false;u.bodyM.color.copy(u.base).lerp(lin("#6d7184"),.7);seats[loser].cup.visible=false}SFX.out()}
 while(G.counts[G.starter]===0)G.starter=(G.starter+1)%4;
 await sleep(900);if(G!==g)return;
 const alive=active();
 if(alive.length===1){G.phase="over";const w=alive[0];zoomT=0;G.focus=w;const rec=store.get("rec",[0,0]);rec[w===0?0:1]++;store.set("rec",rec);
  seats.forEach((x,i)=>{if(x.u&&i!==w&&G.counts[i]>0)x.u.mood="sad"});
  if(w>0){const u=seats[w].u,p=seats[w].g.position;u.mood="happy";u.cheer=true;confetti(p.x*.7,2.3,p.z*.7);setMsg("<b>"+esc(who(w))+"</b> wins the table. Play again to get even.")}
  else{confetti(0,2.2,.9);SFX.win();setMsg("<b>You win the table!</b> All three AIs are out of dice.")}
  log("<span>"+esc(who(w))+" "+(w===0?"win":"wins")+" the table</span>","hd");render();return}
 G.phase="between";
 if(G.counts[0]===0){setMsg("You're out of dice. Watching the rest of the table…");render();await sleep(2400);if(G===g&&G.phase==="between")newRound();return}
 setMsg(G.starter===loser?"<b>"+esc(who(loser))+"</b> "+(loser===0?"lose a die and open":"loses a die and opens")+" the next round.":"<b>"+esc(who(loser))+"</b> "+(loser===0?"are":"is")+" out. "+(G.starter===0?"You open":esc(who(G.starter))+" opens")+" the next round.");render()}

// ---------- coach + HUD ----------
function bestRaise(){const tot=sum(G.counts),hid=tot-G.counts[0],h=G.hands[0],rs=legal(G.last,tot).filter(a=>a!==CALL).map(a=>[a,ptrue(bq(a),bf(a),h,hid)]);
 if(!rs.length)return null;const ok=rs.filter(r=>r[1]>=.55);return ok.length?ok[ok.length-1]:rs.reduce((b,r)=>r[1]>b[1]+1e-9?r:b)}
function promptMe(){const tot=sum(G.counts),hid=tot-G.counts[0],h=G.hands[0],sug=bestRaise();
 qMore=0;const first=legal(G.last,tot).find(a=>a!==CALL),pre=COACH&&sug?sug[0]:first;if(pre!==undefined){qS=bq(pre);fS=bf(pre)}
 if(!COACH){setMsg(G.last<0?"<b>Your turn.</b> Open with any claim.":"<b>Your turn.</b> Raise the claim or call Liar!");return}
 const c=cnt6(h),parts=[1,2,3,4,5,6].filter(x=>c[x-1]).sort((a,b)=>c[b-1]-c[a-1]||b-a).map(x=>c[x-1]===1?"a "+F1[x]:c[x-1]+" "+FW[x]),hold=h.length===2&&h[0]===h[1]?"a pair of "+FW[h[0]]:parts.length>1?parts.slice(0,-1).join(", ")+" and "+parts[parts.length-1]:parts[0];
 const teach=G.taught?"":"A claim counts <em>all "+tot+" dice on the table</em>, not just your "+h.length+". ";G.taught=true;
 if(G.last<0){setMsg("<b>Your turn to open.</b> "+teach+"You hold "+hold+". "+(teach?"Each hidden die shows any given number 1 time in 6, so <em>at least ":"<em>At least ")+fw(bq(sug[0]),bf(sug[0]))+"</em> is "+pct(sug[1])+" likely"+(teach?". It's selected below.":" and selected below."));return}
 const q=bq(G.last),f=bf(G.last),pl=ptrue(q,f,h,hid),mine=h.filter(d=>d===f).length,need=q-mine,wb=esc(who(G.lastBy));
 const lead="<b>Your turn.</b> "+teach+wb+" claims at least "+fw(q,f)+". You hold "+(mine?fw(mine,f):"no "+FW[f])+", so "+(need<=0?"the claim is already true":need+" more must be among the "+hid+" hidden dice: <em>"+pct(pl)+" likely</em>")+". ";
 if(!sug)setMsg(lead+"No higher claim is possible, so call <em>Liar!</em>");
 else if(1-pl>sug[1])setMsg(lead+"Calling <em>Liar!</em> wins "+pct(1-pl)+" of the time, better than your safest raise ("+pct(sug[1])+").");
 else setMsg(lead+(need<=0?"Don't call. ":"Calling would only win "+pct(1-pl)+" of the time. ")+"Better to raise: <em>at least "+fw(bq(sug[0]),bf(sug[0]))+"</em> is "+pct(sug[1])+" likely and selected below.")}
function render(){if(!G)return;const tot=sum(G.counts),mine=myTurn(),acts=legal(G.last,tot),hid=tot-G.counts[0],ok=(q,f)=>acts.includes((q-1)*6+f-1);
 $("claim").innerHTML=G.last<0?"<small>"+tot+" dice in play</small><b>No claim yet</b>":"<small>"+esc(who(G.lastBy))+" "+(G.lastBy===0?"claim":"claims")+" at least</small><b>"+bq(G.last)+" × "+pip(bf(G.last))+"<span>"+FW[bf(G.last)]+"</span></b>";
 $("mine").innerHTML=G.counts[0]?'<span class="lbl">Your dice</span>'+G.hands[0].map(pip).join("")+'<span class="lbl">+</span><span class="hidn">'+"<s></s>".repeat(hid)+'</span><span class="lbl">'+hid+" hidden</span>":'<span class="lbl">You\'re out</span>';
 $("ofall").textContent="Of all "+tot+" dice, ";
 $("hud").classList.toggle("myturn",mine);$("build").hidden=!mine;
 if(mine){if(!ok(qS,fS)){const k=acts.find(a=>a!==CALL);if(k!==undefined){qS=bq(k);fS=bf(k)}}
  const qmin=acts.length&&acts[0]!==CALL?bq(acts[0]):tot+1,qhi=Math.min(tot,Math.max(qmin+7+qMore,qS));let qh="";for(let q=qmin;q<=qhi;q++)qh+='<button data-q="'+q+'" aria-pressed="'+(q===qS)+'">'+q+"</button>";
  if(qhi<tot)qh+='<button data-more="1" aria-label="Show higher numbers">…</button>';$("qty4").innerHTML=qh||"<span class='lbl'>No higher claim is possible</span>";
  $("face4").innerHTML=[1,2,3,4,5,6].map(f=>'<button class="fc" data-f="'+f+'" aria-label="'+FW[f]+'" aria-pressed="'+(f===fS)+'" '+(ok(qS,f)?"":"disabled")+">"+pip(f)+"</button>").join("");
  const can=ok(qS,fS),p=ptrue(qS,fS,G.hands[0],hid);$("bid4").disabled=!can;$("bid4").innerHTML=can?"Claim at least "+fw(qS,fS):"Pick a higher claim";
  $("liar4").disabled=G.last<0;$("liar4").textContent=G.last<0?"Liar! (nothing to call yet)":"Call Liar! on "+who(G.lastBy);
  $("odds").hidden=!COACH||!can;$("oddsbar").style.width=(p*100).toFixed(0)+"%";$("oddstxt").textContent=pct(p)+" likely, counting all "+tot+" dice · "+(p>=.75?"safe":p>=.5?"solid":p>=.25?"risky":"a bluff")}
 $("next4").hidden=!(G.phase==="between"&&G.counts[0]>0);$("again4").hidden=G.phase!=="over";
 seats.forEach((s,i)=>{if(!s.tag)return;s.tag.classList.toggle("on",G.phase==="play"&&G.turn===i);s.tag.classList.toggle("out",G.counts[i]===0);s.tag.lastChild.innerHTML="<s></s>".repeat(G.counts[i]);s.tw=s.tag.offsetWidth/2});measure()}

// ---------- lobby, menu, wiring ----------
function lobby(){const WIN=WINS[DICE]||{},pos=["Left","Across","Right"],avail=Object.keys(NAMES).sort((a,b)=>(WIN[b]||0)-(WIN[a]||0));
 $("diceSeg").querySelectorAll("button").forEach(b=>{b.setAttribute("aria-pressed",+b.dataset.d===DICE);b.onclick=()=>{DICE=+b.dataset.d;store.set("dice",DICE);lobby()}});
 $("seatpick").innerHTML=pos.map((p,i)=>'<label><span>'+p+'</span><select data-i="'+i+'">'+avail.map(n=>'<option value="'+n+'" '+(n===pick[i]?"selected":"")+">"+esc(NAMES[n])+"</option>").join("")+"</select><b>"+(WIN[pick[i]]!=null?pct(WIN[pick[i]])+" wins":"")+"</b><small>"+esc(STYLE[pick[i]])+"</small></label>").join("");
 $("seatpick").querySelectorAll("select").forEach(s=>s.onchange=()=>{const i=+s.dataset.i,j=pick.indexOf(s.value);if(j>=0&&j!==i)pick[j]=pick[i];pick[i]=s.value;store.set("pick",pick);if(R3&&(!G||G.phase==="over"))seatTable();lobby()});
 const rec=store.get("rec",[0,0]);$("lobbyNote").textContent="Win rates are from four-player games between the bots with "+DICE+" dice each (25% is average)."+(DICE>2?" With more than 2 dice the AIs play from a shortlist of sensible claims and the odds of each.":"")+(rec[0]+rec[1]?" Your record: "+rec[0]+" won, "+rec[1]+" lost.":"");
 $("coachChk").checked=COACH;$("lobby").hidden=false;menuText()}
function menuText(){$("coachBtn").textContent="Coach tips: "+(COACH?"on":"off");$("sndBtn").textContent="Sound: "+(MUTE?"off":"on");$("spdBtn").textContent="Speed: "+(SPD===2?"fast":"normal");$("logBtn").textContent=$("logp").hidden?"Show table log":"Hide table log"}
function setCoach(v){COACH=v;store.set("coach",v);$("coachChk").checked=v;menuText();if(myTurn()){promptMe();render()}}
$("coachChk").onchange=e=>setCoach(e.target.checked);$("coachBtn").onclick=()=>setCoach(!COACH);
$("sndBtn").onclick=()=>{MUTE=!MUTE;store.set("mute",MUTE);if(!MUTE){audioOn();SFX.turn()}menuText()};
$("spdBtn").onclick=()=>{SPD=SPD===2?1:2;store.set("spd",SPD);menuText()};
$("logBtn").onclick=()=>{$("logp").hidden=!$("logp").hidden;menuText()};
$("helpBtn").onclick=()=>{$("help").hidden=false};$("helpClose").onclick=()=>{$("help").hidden=true};
$("lobbyBtn").onclick=()=>{$("menu").open=false;if(G)G=Object.assign({},G,{phase:"over"});$("verdict").hidden=true;$("build").hidden=true;$("next4").hidden=$("again4").hidden=true;setMsg("Pick your opponents to start.");lobby()};
$("dealBtn").onclick=async()=>{audioOn();if(!R3)return;const b=$("dealBtn");b.disabled=true;b.textContent="Waking up the AIs…";const ok=await Promise.all(pick.map(n=>loadAgent(n,DICE)));b.disabled=false;b.textContent="Deal me in";

 $("lobby").hidden=true;seatTable();newGame();const bad=pick.filter((n,i)=>!ok[i]);if(bad.length)log("<span>Couldn't load "+bad.map(n=>esc(NAMES[n])).join(", ")+"; playing by the raw odds instead.</span>")};
$("bid4").onclick=async()=>{if(busy||!myTurn())return;audioOn();busy=true;try{await doBid(0,(qS-1)*6+fS-1)}finally{busy=false}loop()};
$("liar4").onclick=async()=>{if(busy||!myTurn()||G.last<0)return;audioOn();busy=true;try{await reveal(0)}finally{busy=false}};
$("next4").onclick=()=>{if(G&&G.phase==="between")newRound()};$("again4").onclick=()=>{if(G&&G.phase==="over")newGame()};
$("hud").addEventListener("click",e=>{const q=e.target.closest("[data-q]"),f=e.target.closest("[data-f]");if(!myTurn())return;if(e.target.closest("[data-more]")){qMore+=8;render();return}
 if(q){qS=+q.dataset.q;const tot=sum(G.counts),acts=legal(G.last,tot);if(!acts.includes((qS-1)*6+fS-1)){const k=[1,2,3,4,5,6].find(x=>acts.includes((qS-1)*6+x-1));if(k)fS=k}render()}
 if(f&&!f.disabled){fS=+f.dataset.f;render()}});
document.addEventListener("click",e=>{const m=$("menu");if(m.open&&!m.contains(e.target))m.open=false});

const view=$("view-table");
window.LDTable={pause(){running=false;if(R3)R3.setAnimationLoop(null)},resume(){if(R3&&!running&&!document.hidden&&!view.hidden){running=true;clock.getDelta();R3.setAnimationLoop(frame)}}};
document.addEventListener("visibilitychange",()=>{if(document.hidden)window.LDTable.pause();else window.LDTable.resume()});
window.LDTest={get G(){return G},seats:()=>seats,low:()=>lowQ};
try{init3d();seatTable();G=null;window.LDTable.resume();setMsg("Pick your opponents to start.");lobby()}
catch(e){setMsg("3D couldn't start on this device ("+esc(e.message)+").");$("lobby").hidden=true}
})();
