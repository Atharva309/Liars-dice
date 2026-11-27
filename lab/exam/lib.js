// Exam library: loads every generation of bots with the exact JavaScript that shipped, and adds functions that return
// each bot's full move distribution (instead of one sampled move) so results can be computed exactly.
'use strict';
const fs=require('fs'),path=require('path'),ROOT=path.join(__dirname,'..'),rd=f=>JSON.parse(fs.readFileSync(path.join(ROOT,f)));
function logic(file,extra,ret){const src=fs.readFileSync(path.join(ROOT,file),'utf8'),a=src.indexOf('//<logic>'),b=src.indexOf('//</logic>');return new Function(src.slice(a,b)+';'+extra+';return '+ret)()}
const RANDD=`const randD=(o,acts)=>{const b=acts.filter(a=>a!==CALL);if(!b.length)return[[CALL,1]];if(!acts.includes(CALL))return b.map(a=>[a,1/b.length]);return[[CALL,1/3]].concat(b.map(a=>[a,2/3/b.length]))};
const norm=w=>{let s=0;for(const p of w)s+=p[1];return s>0?w.filter(p=>p[1]>0).map(p=>[p[0],p[1]/s]):null};`;
// distribution versions of policyN (generation 2)
const DIST_NEW=RANDD+`
function distN(n){
 if(n==="math")return(o,acts)=>[[mathBot(o,acts),1]];if(n==="random")return randD;
 return(o,acts,ctx)=>{const ag=BOT[n];if(!ag||!MODEL)throw new Error("missing bot "+n);const x=viewN(o,ctx),P=prepN(x),A=P.acts;if(A.length===1)return[[A[0],1]];
  if(!ag.net){const v=tviewN(x,P.T,A),row=ag.t[v.key];if(!row)return[[mathBot(o,acts),1]];const bs=[...v.bm.keys()].sort((a,b)=>a-b);
   if(ag.kind==="mcq"){let b=bs[0];for(const k of bs)if(row[k]>row[b])b=k;return[[v.bm.get(b),1]]}
   return norm(bs.map(k=>[v.bm.get(k),Math.max(row[k],0)]))||[[mathBot(o,acts),1]]}
  const F=featsN(x,P.T,A),z=A.map((_,i)=>fwd(ag.net,F.subarray(i*NFN,i*NFN+NFN))[0]);
  if(ag.kind==="q"){let b=0;for(let i=1;i<z.length;i++)if(z[i]>z[b])b=i;return[[A[b],1]]}
  const m=Math.max.apply(null,z),T=ag.temp||1;return norm(A.map((a,i)=>[a,Math.exp((z[i]-m)/T)]))}}`;
// distribution versions of policy2 (original 2-dice bots) and policy5 (first any-dice bots)
const DIST_OLD=RANDD+`
function dist2(n){
 if(n==="math")return(o,acts)=>[[mathBot(o,acts),1]];if(n==="random")return randD;
 const ag=AG[n];if(!ag)throw new Error("missing bot "+n);const to1=a=>a===CALL?48:a,from1=a=>a===48?CALL:a;
 if(ag.kind==="det")return(o,acts)=>{const a=from1(ag.policy[tkey(o)]);return[[a!==undefined&&acts.includes(a)?a:mathBot(o,acts),1]]};
 if(ag.kind==="table")return(o,acts)=>{const p=(ag.policy[tkey(o)]||[]).map(x=>[from1(x[0]),x[1]]).filter(x=>acts.includes(x[0]));return p.length?(norm(p)||[[p[0][0],1]]):[[mathBot(o,acts),1]]};
 return(o,acts)=>{const out=fwd(ag.net,encode(o)),v=a=>out[to1(a)];if(ag.kind==="greedy"){let b=acts[0];for(const a of acts)if(v(a)>v(b))b=a;return[[b,1]]}
  let m=-1e9;for(const a of acts)if(v(a)>m)m=v(a);return norm(acts.map(a=>[a,Math.exp(v(a)-m)]))}}
function dist5(n){
 if(n==="math")return(o,acts)=>[[mathBot(o,acts),1]];if(n==="random")return randD;
 const ag=AG5[n];if(!ag)throw new Error("missing bot "+n);
 if(ag.net)return(o)=>{const c=cnt6(o[0]),A=cands5(c,o[1],o[2],o[4]);if(A.length===1)return[[A[0],1]];const F=feats5(c,o[1],o[2],o[3],o[4],o[5],A),z=A.map((_,i)=>fwd(ag.net,F.subarray(i*15,i*15+15))[0]);
  if(ag.kind==="q"){let b=0;for(let i=1;i<z.length;i++)if(z[i]>z[b])b=i;return[[A[b],1]]}const m=Math.max.apply(null,z);return norm(A.map((a,i)=>[a,Math.exp(z[i]-m)]))};
 return(o,acts)=>{const c=cnt6(o[0]),bm=bmap5(c,o[2],o[4],cands5(c,o[1],o[2],o[4])),row=ag.t[skey5(c,o[1],o[2],o[3],o[4])];if(!row)return[[mathBot(o,acts),1]];const bs=[...bm.keys()].sort((a,b)=>a-b);
  if(ag.kind==="mcq"){let b=bs[0];for(const x of bs)if(row[x]>row[b])b=x;return[[bm.get(b),1]]}return norm(bs.map(b=>[bm.get(b),Math.max(row[b],0)]))||[[mathBot(o,acts),1]]}}`;
const RET_NEW='{legal,CALL,bq,bf,policyN,distN,BOT,decodeNet,setModel:m=>{MODEL=m}}',RET_OLD='{legal,CALL,policy2,policy5,dist2,dist5,AG,AG5,decodeNet}';
const NEW=logic('site/table.js',DIST_NEW,RET_NEW),MID=logic('site/table.js',DIST_NEW,RET_NEW),OLD=logic('old_table.js',DIST_OLD,RET_OLD);
const LEARN=['ppo','dqn','mc','nfsp','cfr'],dec=(L,d)=>d.layers?{kind:d.kind,temp:d.temp,net:L.decodeNet(d.layers)}:d;
NEW.setModel(rd('site/bots/model.json'));NEW.BOT.lite=dec(NEW,rd('site/bots/lite.json'));MID.setModel(rd('exam/bots_p1/model.json'));
for(const n of LEARN){NEW.BOT[n]=dec(NEW,rd('site/bots/'+n+'.json'));MID.BOT[n]=dec(MID,rd('exam/bots_p1/'+n+'.json'));OLD.AG5[n]=dec(OLD,rd('site/ai5/'+n+'.json'));OLD.AG[n]=dec(OLD,rd('site/ai/'+n+'.json'))}
// student ids: "new:ppo" (current), "mid:ppo" (halfway through the retrain), "old:ppo" (first any-dice version), "orig:ppo" (original 2-dice bots), "math", "random"
const CALL=NEW.CALL,legal=NEW.legal;
function distFor(id){const[g,n]=id.includes(':')?id.split(':'):['new',id];return g==='new'?NEW.distN(n):g==='mid'?MID.distN(n):g==='old'?OLD.dist5(n):OLD.dist2(n)}
function playFor(id){const[g,n]=id.includes(':')?id.split(':'):['new',id];return g==='new'?NEW.policyN(n):g==='mid'?MID.policyN(n):g==='old'?OLD.policy5(n):OLD.policy2(n)}
const NAMES={ppo:'Gut Player',dqn:'Exploiter',mc:'Old-Schooler',nfsp:'Hybrid',cfr:'Game Theorist',ismcts:'Thinker',math:'Math Bot',random:'Random Bot'};
// all hands of n dice as sorted arrays, with probability and face counts
const HCACHE={};function hands(n){if(HCACHE[n])return HCACHE[n];const out=[],fact=[1,1,2,6,24,120],rec=(start,cur)=>{if(cur.length===n){const c=[0,0,0,0,0,0];for(const d of cur)c[d-1]++;let p=fact[n];for(const v of c)p/=fact[v];out.push({h:cur.slice(),c,p:p/Math.pow(6,n)});return}for(let d=start;d<=6;d++){cur.push(d);rec(d,cur);cur.pop()}};rec(1,[]);return HCACHE[n]=out}
module.exports={NEW,MID,OLD,LEARN,CALL,legal,distFor,playFor,NAMES,hands,rd,ROOT};
