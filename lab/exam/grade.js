// Paper A: grade a bot against the answer key (the solved game).
// For every situation the bot can face against the perfect player, compare its move with the best move.
'use strict';
const fs=require('fs'),path=require('path'),{CALL,legal,distFor}=require('./lib.js'),D=require('./duel.js');
const TOL=.005,BLUNDER=.05,TIER=.005,EPS=1e-12;   // a move counts as correct if it gives up at most half a point of round-win chance; a blunder gives up 5 points or more

function loadKey(a,b,s){const G=D.makeGame(a,b,s),file=path.join(__dirname,'out',`nash_${a}_${b}_${s}.f32`),S=new Float32Array(G.size),fd=fs.openSync(file,'r'),u8=new Uint8Array(S.buffer);
 let pos=0;while(pos<u8.length){const n=fs.readSync(fd,u8,pos,Math.min(1<<28,u8.length-pos),pos);if(!n)throw new Error('short read');pos+=n}fs.closeSync(fd);
 const Q=new Float32Array(G.size),v0=D.bestReply(G,S,0,Q),v1=D.bestReply(G,S,1,Q);return{G,S,Q,value:[(v0+1-v1)/2,(v1+1-v0)/2],br:[v0,v1],gap:v0+v1-1}}   // value[s]: seat s's chance of winning the round under perfect play; br[s]: what a best reply to the key wins (equal up to the gap)

const blank=()=>({n:0,ok:0,loss:0,bl:0});
function grade(K,seat,dist,mathD,want={}){
 const {G,S,Q}=K,{NB,Hn,Hmax,cf,nd}=G,opp=1-seat,Hb=Hn[seat],Ho=Hn[opp],Dp=NB+2,mk=H=>Array.from({length:Dp},()=>new Float64Array(H));
 const ro=mk(Ho),rbot=mk(Hb),rstd=mk(Hb),vb=mk(Hb),T=new Float64Array(8),tmp=new Float64Array(Hb),prow=new Float64Array(NB+1),mrow=new Float64Array(NB+1),nrow=new Float64Array(NB+1),hist=[];
 const R={std:blank(),act:blank(),tier:{sanity:blank(),easy:blank(),hard:blank()},kind:{open:blank(),reply:blank()},calledProven:0,nProven:0,missedImpossible:0,nImpossible:0,
  err:{badCall:0,missedCall:0,badRaise:0,badOpen:0},callRate:{n:0,bot:0,key:0},top:[],nodes:0,evals:0};
 function add(z,w,ok,loss,bl){z.n+=w;z.ok+=w*ok;z.loss+=w*loss;z.bl+=w*bl}
 function rec(mask,k,d){const p=(G.starter+d)&1,nb=NB-1-k,A=nb+(k>=0?1:0),o0=G.off(mask,k),out=vb[d];R.nodes++;out.fill(0);
  if(p===seat){const r=ro[d],rb=rbot[d],rs=rstd[d];let tot=0;for(let h=0;h<Ho;h++)tot+=r[h];
   const pol=new Float64Array(Hb*A),nash=new Float64Array(Hb*A),f=k>=0?k%6:0,qq=k>=0?((k/6)|0)+1:0;
   for(let h=0;h<Hb;h++){const needB=rb[h]>EPS,needS=rs[h]>EPS;if(!needB&&!needS)continue;
    D.botRow(G,dist,seat,mask,k,hist,h,prow);D.stratRow(G,S,mask,k,h,nrow);R.evals++;for(let a=0;a<A;a++){pol[h*A+a]=prow[a];nash[h*A+a]=nrow[a]}
    let qmax=-1,abest=0;for(let a=0;a<A;a++){const q=Q[o0+h*A+a];if(q>qmax){qmax=q;abest=a}}
    let ok=0,loss=0,bl=0;for(let a=0;a<A;a++){const pa=prow[a];if(pa<=0)continue;const l=Math.max(0,qmax-Q[o0+h*A+a]);loss+=pa*l;if(l<=TOL)ok+=pa;if(l>=BLUNDER)bl+=pa}
    const ws=G.prior[seat][h]*rs[h]*tot,wa=G.prior[seat][h]*rb[h]*tot;
    if(needS){add(R.std,ws,ok,loss,bl);add(k<0?R.kind.open:R.kind.reply,ws,ok,loss,bl);
     let tier;const c=cf[seat][h*6+f];
     if(k>=0&&c>=qq){tier='sanity';R.nProven+=ws;R.calledProven+=ws*prow[A-1]}
     else if(k>=0&&qq-c>nd[opp]){tier='sanity';R.nImpossible+=ws;R.missedImpossible+=ws*(1-prow[A-1])}
     else{D.botRow(G,mathD,seat,mask,k,hist,h,mrow);let ml=0;for(let a=0;a<A;a++)if(mrow[a]>0)ml+=mrow[a]*Math.max(0,qmax-Q[o0+h*A+a]);tier=ml<=TIER?'easy':'hard'}
     add(R.tier[tier],ws,ok,loss,bl);
     if(k>=0){R.callRate.n+=ws;R.callRate.bot+=ws*prow[A-1];R.callRate.key+=ws*nrow[A-1]}}
    if(needB&&wa>0){add(R.act,wa,ok,loss,bl);
     if(k<0)R.err.badOpen+=wa*loss;else{const lc=Math.max(0,qmax-Q[o0+h*A+A-1]);R.err.badCall+=wa*prow[A-1]*lc;let lr=0;for(let a=0;a<nb;a++)if(prow[a]>0)lr+=prow[a]*Math.max(0,qmax-Q[o0+h*A+a]);if(lc<=TOL)R.err.missedCall+=wa*lr;else R.err.badRaise+=wa*lr}
     if(want.top)for(let a=0;a<A;a++){const pa=prow[a],l=qmax-Q[o0+h*A+a];if(pa>.2&&l>=BLUNDER){const sc=wa*pa*l;if(R.top.length<want.top||sc>R.top[R.top.length-1].sc){R.top.push({sc,hand:G.HS[seat][h].h,bids:hist.map(x=>x[1]),move:a<nb?k+1+a:CALL,best:abest<nb?k+1+abest:CALL,p:pa,lost:l,q:Q[o0+h*A+a],qbest:qmax});R.top.sort((x,y)=>y.sc-x.sc);if(R.top.length>want.top)R.top.pop()}}}}}
   for(let a=0;a<A;a++){let anyB=false,anyS=false;const cb=rbot[d+1],cs=rstd[d+1];for(let h=0;h<Hb;h++){cb[h]=rb[h]*pol[h*A+a];cs[h]=rs[h]*nash[h*A+a];if(cb[h]>EPS)anyB=true;if(cs[h]>EPS)anyS=true}
    let cv;if(a<nb){if(!anyB&&!anyS)continue;ro[d+1].set(r);const j=k+1+a;hist.push([seat,j]);rec(mask|(1<<j),j,d+1);hist.pop();cv=vb[d+1]}
    else{if(!anyB)continue;D.tails(G,opp,r,f,T);cv=tmp;for(let h=0;h<Hb;h++)cv[h]=T[0]-D.tl(T,nd[opp],qq-cf[seat][h*6+f])}
    for(let h=0;h<Hb;h++)if(rb[h]>EPS)out[h]+=pol[h*A+a]*cv[h]}}
  else{const r=ro[d],st=new Float64Array(Ho*A);for(let h=0;h<Ho;h++){if(r[h]<=0)continue;D.stratRow(G,S,mask,k,h,nrow);for(let a=0;a<A;a++)st[h*A+a]=nrow[a]}
   for(let a=0;a<A;a++){const c=ro[d+1];let m=0;for(let h=0;h<Ho;h++){c[h]=r[h]*st[h*A+a];m+=c[h]}if(m<=EPS)continue;let cv;
    if(a<nb){rbot[d+1].set(rbot[d]);rstd[d+1].set(rstd[d]);const j=k+1+a;hist.push([opp,j]);rec(mask|(1<<j),j,d+1);hist.pop();cv=vb[d+1]}
    else{const f=k%6,qq=((k/6)|0)+1;D.tails(G,opp,c,f,T);cv=tmp;for(let h=0;h<Hb;h++)cv[h]=D.tl(T,nd[opp],qq-cf[seat][h*6+f])}
    for(let h=0;h<Hb;h++)out[h]+=cv[h]}}}
 ro[0].set(G.prior[opp]);rbot[0].fill(1);rstd[0].fill(1);rec(0,-1,0);let win=0;for(let h=0;h<Hb;h++)win+=G.prior[seat][h]*vb[0][h];R.win=win;R.key=K.value[seat];return R}
module.exports={loadKey,grade,TOL,BLUNDER,TIER};
