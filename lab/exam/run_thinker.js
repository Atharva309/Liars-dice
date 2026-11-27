// Paper A for the Thinker. It is a randomised search, so its move probabilities cannot be written down exactly; it is graded on sampled rounds instead.
// node run_thinker.js <n0> <n1> <starter> <rounds>   (adds "…:ismcts@seat" entries, marked sampled, to out/key_<n0>_<n1>_<s>.json)
const fs=require('fs'),{CALL,legal,playFor}=require('./lib.js'),D=require('./duel.js'),{loadKey,TOL,BLUNDER,TIER}=require('./grade.js'),{cachedDist}=require('./students.js');
const [a,b,s,M]=process.argv.slice(2,6).map(Number),K=loadKey(a,b,s),{G,S,Q}=K,file=`${__dirname}/out/key_${a}_${b}_${s}.json`,out=JSON.parse(fs.readFileSync(file)),mathD=cachedDist('math'),row=new Float64Array(G.NB+1),mrow=new Float64Array(G.NB+1);
const blank=()=>({n:0,ok:0,loss:0,bl:0}),add=(z,ok,l)=>{z.n++;if(l<=TOL)z.ok++;z.loss+=l;if(l>=BLUNDER)z.bl++};
const pickHand=p=>{let r=Math.random();const L=G.prior[p];for(let i=0;i<L.length;i++){r-=L[i];if(r<=0)return i}return L.length-1},pickRow=A=>{let r=Math.random();for(let i=0;i<A;i++){r-=row[i];if(r<=0)return i}return A-1};
function ask(play,seat,h,k,hist){const mv=play([G.HS[seat][h].h,G.nd[seat],G.nd[1-seat],1,k,hist.length>1?hist[hist.length-2][1]:-1],legal(k,G.N),{order:[seat,1-seat],total:G.N,hist:hist.slice(),counts:G.nd});return mv===CALL?G.nA(k)-1:mv-k-1}
function sampleGrade(play,seat){const R={std:blank(),act:blank(),tier:{sanity:blank(),easy:blank(),hard:blank()},kind:{open:blank(),reply:blank()},calledProven:0,nProven:0,missedImpossible:0,nImpossible:0,err:{badCall:0,missedCall:0,badRaise:0,badOpen:0},callRate:{n:0,bot:0,key:0},top:[],sampled:M};let wins=0;
 for(let mode=0;mode<2;mode++)for(let g=0;g<M;g++){const hd=[pickHand(0),pickHand(1)],hist=[];let mask=0,k=-1,d=0,win=null;
  for(;;){const p=(G.starter+d)&1,A=G.nA(k),nb=G.NB-1-k,o0=G.off(mask,k)+hd[p]*A;let act;
   if(p===seat){let qmax=-1;for(let x=0;x<A;x++)if(Q[o0+x]>qmax)qmax=Q[o0+x];const ba=ask(play,seat,hd[p],k,hist),l=Math.max(0,qmax-Q[o0+ba]);
    if(mode===0){act=ba;add(R.act,0,l);if(k<0)R.err.badOpen+=l;else{const lc=Math.max(0,qmax-Q[o0+A-1]);if(ba===A-1)R.err.badCall+=l;else if(lc<=TOL)R.err.missedCall+=l;else R.err.badRaise+=l}}
    else{D.stratRow(G,S,mask,k,hd[p],row);act=pickRow(A);add(R.std,0,l);add(k<0?R.kind.open:R.kind.reply,0,l);const f=k>=0?k%6:0,qq=k>=0?((k/6)|0)+1:0,c=G.cf[seat][hd[p]*6+f];let tier;
     if(k>=0&&c>=qq){tier='sanity';R.nProven++;if(ba===A-1)R.calledProven++}else if(k>=0&&qq-c>G.nd[1-seat]){tier='sanity';R.nImpossible++;if(ba!==A-1)R.missedImpossible++}
     else{D.botRow(G,mathD,seat,mask,k,hist,hd[p],mrow);let ml=0;for(let x=0;x<A;x++)if(mrow[x]>0)ml+=mrow[x]*Math.max(0,qmax-Q[o0+x]);tier=ml<=TIER?'easy':'hard'}
     add(R.tier[tier],0,l);if(k>=0){R.callRate.n++;if(ba===A-1)R.callRate.bot++;R.callRate.key+=row[A-1]}}}
   else{D.stratRow(G,S,mask,k,hd[p],row);act=pickRow(A)}
   if(act===A-1&&k>=0){const f=k%6,qq=((k/6)|0)+1,truth=G.cf[0][hd[0]*6+f]+G.cf[1][hd[1]*6+f]>=qq;win=truth?1-p:p;break}
   const j=k+1+act;hist.push([p,j]);mask|=1<<j;k=j;d++}
  if(mode===0&&win===seat)wins++}
 R.win=wins/M;R.key=K.value[seat];return R}
const ids=['new:ismcts','old:ismcts'].concat(Math.max(a,b)<=2?['orig:ismcts']:[]);
for(const id of ids)for(const seat of [0,1]){const t0=Date.now(),R=sampleGrade(playFor(id),seat);R.ms=Date.now()-t0;out.res[id+'@'+seat]=R;
 console.error(id.padEnd(11),'seat',seat,`wins round ${(R.win*100).toFixed(1)}% (perfect ${(K.value[seat]*100).toFixed(1)}%)  paper: correct ${(100*R.std.ok/R.std.n).toFixed(1)}% lost/decision ${(100*R.std.loss/R.std.n).toFixed(2)} | own games: correct ${(100*R.act.ok/R.act.n).toFixed(1)}% lost ${(100*R.act.loss/R.act.n).toFixed(2)}  ${R.ms}ms`)}
fs.writeFileSync(file,JSON.stringify(out));
