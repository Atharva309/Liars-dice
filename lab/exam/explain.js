// Show the hunter's plan against a bot in plain terms: node explain.js <id> <hunterDice,botDice> [hunterStarts=1]
const {CALL,legal,distFor,hands}=require('./lib.js'),H=require('./hunter.js');
const id=process.argv[2],start=process.argv[3].split(',').map(Number),hs=process.argv[4]!=='0',bot=H.makeBot(id,distFor(id)),D=distFor(id),name=k=>k===CALL?'Liar!':`${((k/6)|0)+1}×${k%6+1}`;
const policies=new Map(),memo=new Map(),r=H.hunterMatch(start,hs?0:1,bot,memo,{eps:1e-9,policies}),st=memo.get(start.join(',')+'|'+(hs?0:1)),pol=policies.get(start.join(',')+'|'+(hs?0:1));
const p=(st.value-st.pay[0])/(st.pay[1]-st.pay[0]);console.log(`${id}: hunter (${start[0]} dice${hs?', starts':''}) vs bot (${start[1]} dice${hs?'':', starts'}): hunter wins the round ${(p*100).toFixed(1)}%, the match ${(st.value*100).toFixed(1)}%`);
const HH=hands(start[0]),HB=hands(start[1]),N=start[0]+start[1];
// walk the most likely lines
function botReply(hist){const m=new Map(),last=hist[hist.length-1][1],prev=hist.length>1?hist[hist.length-2][1]:-1;for(const h of HB){const d=D([h.h,start[1],start[0],1,last,prev],legal(last,N),{order:[1,0],total:N,hist,counts:start});for(const[a,q]of d)m.set(a,(m.get(a)||0)+q*h.p)}return[...m.entries()].sort((a,b)=>b[1]-a[1])}
if(hs){const open=new Map();HH.forEach((h,i)=>{const a=pol.get('')[i];open.set(a,(open.get(a)||0)+h.p)});
 for(const[a,q]of[...open.entries()].sort((x,y)=>y[1]-x[1]).slice(0,4)){const rep=botReply([[0,a]]);console.log(`  hunter opens ${name(a)} (${(q*100).toFixed(0)}% of its hands) -> bot: ${rep.slice(0,4).map(([b,x])=>`${name(b)} ${(x*100).toFixed(0)}%`).join(', ')}`);
  for(const[b,x]of rep.slice(0,2)){if(b===CALL)continue;const nx=pol.get(a+','+b);if(!nx)continue;const m=new Map();HH.forEach((h,i)=>{if(pol.get('')[i]===a)m.set(nx[i],(m.get(nx[i])||0)+h.p)});console.log(`      after bot says ${name(b)}: hunter ${[...m.entries()].sort((u,v)=>v[1]-u[1]).slice(0,3).map(([c,y])=>`${name(c)} ${(100*y/q).toFixed(0)}%`).join(', ')}`)}}}
else{const m=new Map();for(const h of HB){const d=D([h.h,start[1],start[0],1,-1,-1],legal(-1,N),{order:[1,0],total:N,hist:[],counts:start});for(const[a,q]of d)m.set(a,(m.get(a)||0)+q*h.p)}
 for(const[a,q]of[...m.entries()].sort((x,y)=>y[1]-x[1]).slice(0,4)){const nx=pol.get(''+a),mm=new Map();if(!nx)continue;HH.forEach((h,i)=>mm.set(nx[i],(mm.get(nx[i])||0)+h.p));console.log(`  bot opens ${name(a)} (${(q*100).toFixed(0)}%) -> hunter: ${[...mm.entries()].sort((u,v)=>v[1]-u[1]).slice(0,4).map(([c,y])=>`${name(c)} ${(100*y).toFixed(0)}%`).join(', ')}`)}}
