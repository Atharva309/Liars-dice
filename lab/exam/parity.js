// Check: the exact move distributions used by the exam match what the shipped (sampling) policies actually do.
const {CALL,legal,distFor,playFor,LEARN,hands}=require('./lib.js');
const rnd=n=>Math.floor(Math.random()*n),roll=n=>Array.from({length:n},()=>1+rnd(6)).sort();
function randomState(k,maxd){const cnt=Array.from({length:k},()=>1+rnd(maxd)),total=cnt.reduce((a,b)=>a+b,0),hist=[];let last=-1,prev=-1,seat=rnd(k);const steps=rnd(5);
 for(let i=0;i<steps;i++){const hi=Math.min(total*6-1,last+1+rnd(4));if(hi<=last||hi>=total*6-1)break;hist.push([seat,hi]);prev=last;last=hi;seat=(seat+1)%k}
 const order=[];for(let i=0;i<k;i++)order.push((seat+i)%k);return{o:[roll(cnt[seat]),cnt[seat],total-cnt[seat],k-1,last,prev],acts:legal(last,total),ctx:{order,total,hist,counts:cnt}}}
let worst=0,checked=0;
for(const g of ['new','mid','old','orig'])for(const n of LEARN.concat(['math','random'])){const id=g+':'+n,D=distFor(id),P=playFor(id);let mx=0;
 for(let t=0;t<60;t++){const s=g==='orig'?randomState(2+rnd(3),2):randomState(2+rnd(3),g==='old'?5:3),d=D(s.o,s.acts,s.ctx),tot=d.reduce((a,p)=>a+p[1],0);
  if(Math.abs(tot-1)>1e-9)throw new Error('not normalised '+id);for(const p of d)if(!s.acts.includes(p[0]))throw new Error('illegal in dist '+id+' '+p[0]);
  const M=2500,f=new Map();for(let i=0;i<M;i++){const a=P(s.o,s.acts,s.ctx);f.set(a,(f.get(a)||0)+1)}
  const keys=new Set([...f.keys(),...d.map(p=>p[0])]);for(const a of keys){const pe=(f.get(a)||0)/M,pd=(d.find(p=>p[0]===a)||[0,0])[1],se=Math.sqrt(Math.max(pd*(1-pd),1e-4)/M),z=Math.abs(pe-pd)/se;if(pd===0&&pe>0)throw new Error('sampled move missing from dist '+id);if(z>mx)mx=z}checked++}
 if(mx>worst)worst=mx;console.log(id.padEnd(12),'max z',mx.toFixed(2))}
console.log('states checked',checked,'worst z',worst.toFixed(2),worst<5?'OK':'CHECK');
