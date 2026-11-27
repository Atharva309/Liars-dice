// One-on-one round games solved exactly.
// A round: player 0 holds n0 dice, player 1 holds n1, `starter` makes the first claim. Payoff = 1 for winning the round (the other player loses a die).
// Because "the other player loses a die" is always the better outcome, perfect play in a match is perfect play in each round, so rounds can be solved separately.
'use strict';
const {CALL,legal,hands}=require('./lib.js');

function makeGame(n0,n1,starter){
 const N=n0+n1,NB=6*N,HS=[hands(n0),hands(n1)],Hn=[HS[0].length,HS[1].length],Hmax=Math.max(Hn[0],Hn[1]),nd=[n0,n1];
 const prior=HS.map(L=>Float64Array.from(L,x=>x.p)),cf=HS.map(L=>{const a=new Uint8Array(L.length*6);L.forEach((x,i)=>{for(let f=0;f<6;f++)a[i*6+f]=x.c[f]});return a});
 // storage: one slot per (bid sequence, hand of the player to act, action). Sequences are subsets of bids = bitmasks; those with top bid k are contiguous.
 const baseK=new Float64Array(NB+1);let size=NB*Hmax;for(let k=0;k<NB;k++){baseK[k]=size;size+=Math.pow(2,k)*(NB-k)*Hmax}
 const nA=k=>NB-1-k+(k>=0?1:0),off=(mask,k)=>k<0?0:baseK[k]+(mask-(1<<k))*(NB-k)*Hmax;
 return{n0,n1,nd,starter,N,NB,HS,Hn,Hmax,prior,cf,size,nA,off}}

// tails of reach grouped by how many of face f a hand holds: T[t] = total reach of hands holding at least t
function tails(G,pl,reach,f,T){const n=G.nd[pl],c=G.cf[pl],H=G.Hn[pl];for(let t=0;t<=n+1;t++)T[t]=0;for(let h=0;h<H;h++)T[c[h*6+f]]+=reach[h];for(let t=n-1;t>=0;t--)T[t]+=T[t+1];return T}
const tl=(T,n,need)=>need<=0?T[0]:need>n?0:T[need];

// CFR+ (regret matching+, linear averaging). Returns the strategy sums S; the average strategy is S normalised per (sequence, hand).
function solve(G,iters,onCheck,every=200){
 const {NB,Hn,Hmax,cf,nd}=G,R=new Float32Array(G.size),S=new Float32Array(G.size),D=NB+2;
 const mk=()=>Array.from({length:D},()=>new Float64Array(Hmax)),r=[mk(),mk()],v=[mk(),mk()],tv=[new Float64Array(Hmax),new Float64Array(Hmax)];
 const sig=Array.from({length:D},()=>new Float64Array(NB*Hmax)),va=Array.from({length:D},()=>new Float64Array(NB*Hmax)),T0=new Float64Array(8),T1=new Float64Array(8);let W=1;
 function rec(mask,k,d){const p=(G.starter+d)&1,q=1-p,Hp=Hn[p],Hq=Hn[q],nb=NB-1-k,A=nb+(k>=0?1:0),o0=G.off(mask,k),rp=r[p][d],rq=r[q][d],vp=v[p][d],vq=v[q][d],sg=sig[d],vA=va[d];
  for(let h=0;h<Hp;h++){const o=o0+h*A;let s=0;for(let a=0;a<A;a++)s+=R[o+a];if(s>0)for(let a=0;a<A;a++)sg[h*A+a]=R[o+a]/s;else{const u=1/A;for(let a=0;a<A;a++)sg[h*A+a]=u}}
  vp.fill(0,0,Hp);vq.fill(0,0,Hq);
  for(let a=0;a<A;a++){const cp=r[p][d+1];for(let h=0;h<Hp;h++)cp[h]=rp[h]*sg[h*A+a];let cvp,cvq;
   if(a<nb){const cq=r[q][d+1];for(let h=0;h<Hq;h++)cq[h]=rq[h];const j=k+1+a;rec(mask|(1<<j),j,d+1);cvp=v[p][d+1];cvq=v[q][d+1]}
   else{ // p calls the claim k made by q: the claim is true -> q wins the round
    const f=k%6,qq=((k/6)|0)+1;tails(G,q,rq,f,T0);tails(G,p,cp,f,T1);cvp=tv[p];cvq=tv[q];
    for(let h=0;h<Hp;h++)cvp[h]=T0[0]-tl(T0,nd[q],qq-cf[p][h*6+f]);for(let h=0;h<Hq;h++)cvq[h]=tl(T1,nd[p],qq-cf[q][h*6+f])}
   for(let h=0;h<Hp;h++){const x=cvp[h];vA[a*Hmax+h]=x;vp[h]+=sg[h*A+a]*x}for(let h=0;h<Hq;h++)vq[h]+=cvq[h]}
  for(let h=0;h<Hp;h++){const o=o0+h*A,x=vp[h],rw=rp[h]*W;for(let a=0;a<A;a++){const t=R[o+a]+vA[a*Hmax+h]-x;R[o+a]=t>0?t:0;S[o+a]+=rw*sg[h*A+a]}}}
 for(let it=1;it<=iters;it++){W=it;r[0][0].set(G.prior[0]);r[1][0].set(G.prior[1]);rec(0,-1,0);if(onCheck&&(it%every===0||it===iters)&&onCheck(it,S)===false)break}
 return S}

// average strategy of the player to act at a node, for one hand: fills out[0..A)
function stratRow(G,S,mask,k,h,out){const A=G.nA(k),o=G.off(mask,k)+h*A;let s=0;for(let a=0;a<A;a++)s+=S[o+a];if(s>0)for(let a=0;a<A;a++)out[a]=S[o+a]/s;else for(let a=0;a<A;a++)out[a]=1/A;return out}

// Best reply of `seat` to the stored strategy S of the other player. Returns the best reply's chance of winning the round.
// If Q is given, it is filled with action values for `seat`: Q[slot] = chance of winning the round after that move when the best reply follows (and the opponent keeps to S).
function bestReply(G,S,seat,Q){
 const {NB,Hn,Hmax,cf,nd}=G,opp=1-seat,Hb=Hn[seat],Ho=Hn[opp],D=NB+2,ro=Array.from({length:D},()=>new Float64Array(Hmax)),vb=Array.from({length:D},()=>new Float64Array(Hmax)),T=new Float64Array(8),row=new Float64Array(NB+1),tmp=new Float64Array(Hmax);
 function rec(mask,k,d){const p=(G.starter+d)&1,nb=NB-1-k,A=nb+(k>=0?1:0),o0=G.off(mask,k),r=ro[d],out=vb[d];
  if(p===seat){let tot=0;for(let h=0;h<Ho;h++)tot+=r[h];out.fill(-1,0,Hb);
   for(let a=0;a<A;a++){let cv;if(a<nb){ro[d+1].set(r);const j=k+1+a;rec(mask|(1<<j),j,d+1);cv=vb[d+1]}
    else{const f=k%6,qq=((k/6)|0)+1;tails(G,opp,r,f,T);cv=tmp;for(let h=0;h<Hb;h++)cv[h]=T[0]-tl(T,nd[opp],qq-cf[seat][h*6+f])}
    for(let h=0;h<Hb;h++){if(cv[h]>out[h])out[h]=cv[h];if(Q)Q[o0+h*A+a]=tot>0?cv[h]/tot:0}}}
  else{out.fill(0,0,Hb);const st=new Float64Array(Ho*A);for(let h=0;h<Ho;h++){stratRow(G,S,mask,k,h,row);for(let a=0;a<A;a++)st[h*A+a]=row[a]}
   for(let a=0;a<A;a++){const c=ro[d+1];let m=0;for(let h=0;h<Ho;h++){c[h]=r[h]*st[h*A+a];m+=c[h]}if(m<=0)continue;let cv;
    if(a<nb){const j=k+1+a;rec(mask|(1<<j),j,d+1);cv=vb[d+1]}
    else{const f=k%6,qq=((k/6)|0)+1;tails(G,opp,c,f,T);cv=tmp;for(let h=0;h<Hb;h++)cv[h]=tl(T,nd[opp],qq-cf[seat][h*6+f])}
    for(let h=0;h<Hb;h++)out[h]+=cv[h]}}}
 ro[0].set(G.prior[opp]);rec(0,-1,0);let val=0;for(let h=0;h<Hb;h++)val+=G.prior[seat][h]*vb[0][h];return val}

// how far S is from perfect play: what the two best replies win together, minus 1 (0 = perfect)
const gap=(G,S)=>bestReply(G,S,0)+bestReply(G,S,1)-1;

// bid sequence of a node (bitmask) as the history the bots expect: [[seat, bid], ...]
function histOf(G,mask){const h=[];let i=0;for(let j=0;j<G.NB;j++)if(mask&(1<<j)){h.push([(G.starter+i)&1,j]);i++}return h}
// a bot's move probabilities at a node for hand h, as a row over the node's actions (bids above k, then call)
function botRow(G,dist,seat,mask,k,hist,h,out){const A=G.nA(k),nb=G.NB-1-k,last=k,prev=hist.length>1?hist[hist.length-2][1]:-1,o=[G.HS[seat][h].h,G.nd[seat],G.nd[1-seat],1,last,prev];
 const d=dist(o,legal(last,G.N),{order:[seat,1-seat],total:G.N,hist,counts:G.nd});for(let a=0;a<A;a++)out[a]=0;for(const[m,p]of d){if(m===CALL)out[A-1]+=p;else{if(m<=k||m>=G.NB)throw new Error('illegal move '+m);out[m-k-1]+=p}}return out}

module.exports={makeGame,solve,stratRow,bestReply,gap,tails,tl,histOf,botRow};
