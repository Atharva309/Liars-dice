// Paper B: the perfect hunter. For one bot, compute the best possible counter-strategy against that bot's actual policy (an exact best reply),
// at a table of k players where every other seat is a copy of the bot. Nothing can beat the bot by more than this.
'use strict';
const {CALL,legal,hands}=require('./lib.js');
// what a bot's policy can depend on, per generation (used to reuse policy evaluations across bid histories that the bot cannot tell apart)
function sigFor(id){const g=id.includes(':')?id.split(':')[0]:'new',n=id.includes(':')?id.split(':')[1]:id,jump=o=>o[4]>=0&&o[5]>=0?Math.max(0,Math.min(3,((o[4]/6)|0)-((o[5]/6)|0))):0;
 if(n==='math'||n==='random')return o=>o[1]+'|'+o[2]+'|'+o[4];
 if(g==='old')return o=>o[1]+'|'+o[2]+'|'+o[3]+'|'+o[4]+'|'+jump(o);
 if(g==='orig')return o=>o[1]+'|'+o[2]+'|'+o[3]+'|'+o[4]+'|'+o[5];
 return(o,ctx)=>{let s=o[1]+'|'+o[4]+'|'+jump(o)+'|'+Math.min(ctx.hist.length,8);for(let i=1;i<ctx.order.length;i++){const st=ctx.order[i],fb=[0,0,0,0,0,0];for(const h of ctx.hist)if(h[0]===st)fb[h[1]%6]++;s+='|'+ctx.counts[st]+':';for(let f=0;f<6;f++)s+=Math.min(fb[f],3)}return s}}
function makeBot(id,dist){return{id,dist,sig:sigFor(id),cache:new Map(),evals:0}}

// one round. nd: dice per seat (turn order), hs: hunter's seat, pay[seat]: hunter's value if that seat loses a die. Returns {value, err, nodes}; err bounds what pruning could have hidden.
function hunterRound(nd,hs,starter,bot,pay,opt={}){
 const k=nd.length,N=nd.reduce((a,b)=>a+b,0),NB=6*N,eps=opt.eps===undefined?1e-10:opt.eps,budget=opt.budget||Infinity,HS=nd.map(n=>hands(n)),Hn=HS.map(L=>L.length),Hh=Hn[hs];
 const reach=[],vh=[],lvl=d=>{while(reach.length<=d){reach.push(nd.map((n,j)=>new Float64Array(Hn[j])));vh.push(new Float64Array(Hh))}};
 const hist=[],cnts=nd.slice(),bots=[];for(let j=0;j<k;j++)if(j!==hs)bots.push(j);let nodes=0;const tmp=new Float64Array(Hh),nbd=N-nd[hs],Dc=new Float64Array(nbd+2),De=new Float64Array(nbd+2);
 // value for every hunter hand when `caller` calls the claim `last` made by `bidder`, given the bots' reach vectors R[j]
 function terminal(last,caller,bidder,R,out){const f=last%6,qq=((last/6)|0)+1;let len=1;Dc.fill(0);Dc[0]=1;
  for(const j of bots){const r=R[j],L=HS[j],n=nd[j];De.fill(0,0,len+n);for(let h=0;h<r.length;h++){const x=r[h];if(x===0)continue;const c=L[h].c[f];for(let i=0;i<len;i++)De[i+c]+=Dc[i]*x}len+=n;for(let i=0;i<len;i++)Dc[i]=De[i]}
  for(let i=len-2;i>=0;i--)Dc[i]+=Dc[i+1];const M=Dc[0],pc=pay[caller],pb=pay[bidder],L=HS[hs];
  for(let h=0;h<Hh;h++){const need=qq-L[h].c[f],t=need<=0?M:need>=len?0:Dc[need];out[h]=t*pc+(M-t)*pb}}
 function rows(seat,last,prev,order){const o=[null,nd[seat],N-nd[seat],k-1,last,prev],ctx={order,total:N,hist,counts:cnts},key=bot.sig(o,ctx);let r=bot.cache.get(key);
  if(!r){r=new Array(Hn[seat]);bot.cache.set(key,r)}return{r,o,ctx}}
 function rec(d,last,prev){if(++nodes>budget||((nodes&8191)===0&&opt.deadline&&Date.now()>opt.deadline))throw new Error('budget');lvl(d+1);const seat=(starter+d)%k,out=vh[d],R=reach[d],C=reach[d+1];
  if(seat===hs){let err=0;const rc=opt.record,best=rc?new Int16Array(Hh).fill(CALL):null;if(last>=0)terminal(last,hs,(hs-1+k)%k,R,out);else out.fill(-Infinity);
   for(let j=last+1;j<NB;j++){for(const b of bots)C[b].set(R[b]);hist.push([hs,j]);const e=rec(d+1,j,last);hist.pop();const cv=vh[d+1];for(let h=0;h<Hh;h++)if(cv[h]>out[h]){out[h]=cv[h];if(rc)best[h]=j}if(e>err)err=e}
   if(rc)rc.set(hist.map(x=>x[1]).join(','),best);return err}
  const order=[];for(let i=0;i<k;i++)order.push((seat+i)%k);const{r:rw,o,ctx}=rows(seat,last,prev,order),acts=legal(last,N),rs=R[seat],H=Hn[seat],by=new Map();let others=1;
  for(const b of bots)if(b!==seat){let m=0;const x=R[b];for(let h=0;h<x.length;h++)m+=x[h];others*=m}
  for(let h=0;h<H;h++){const x=rs[h];if(x<=0)continue;let row=rw[h];if(!row){o[0]=HS[seat][h].h;row=rw[h]=bot.dist(o,acts,ctx);bot.evals++}
   for(const[a,p]of row){if(p<=0)continue;let v=by.get(a);if(!v){v=new Float64Array(H);by.set(a,v)}v[h]=x*p}}
  out.fill(0);let err=0;const keep=C[seat];
  for(const[a,v]of by){let m=0;for(let h=0;h<H;h++)m+=v[h];const M=m*others;if(M<eps){err+=M;continue}
   if(a===CALL){const sv=R[seat];R[seat]=v;terminal(last,seat,(seat-1+k)%k,R,tmp);R[seat]=sv;for(let h=0;h<Hh;h++)out[h]+=tmp[h]}
   else{for(const b of bots)if(b!==seat)C[b].set(R[b]);C[seat]=v;hist.push([seat,a]);const e=rec(d+1,a,last);hist.pop();C[seat]=keep;const cv=vh[d+1];for(let h=0;h<Hh;h++)out[h]+=cv[h];err+=e}}
  return err}
 lvl(0);for(const b of bots)reach[0][b].set(HS[b].map(x=>x.p));const err=rec(0,-1,-1);let value=0;for(let h=0;h<Hh;h++)value+=HS[hs][h].p*vh[0][h];return{value,err,nodes}}

// whole match by working back from the endgames. counts: dice per table seat, hunter = seat 0. Returns the hunter's chance of winning the match.
function hunterMatch(counts,starter,bot,memo,opt={}){
 if(counts[0]===0)return{value:0,err:0};const act=[];for(let s=0;s<counts.length;s++)if(counts[s]>0)act.push(s);if(act.length===1)return{value:1,err:0};
 const key=counts.join(',')+'|'+starter;if(memo.has(key))return memo.get(key);
 const pay=[],perr=[];for(const s of act){const c=counts.slice();c[s]--;let st=s;while(c[st]===0)st=(st+1)%c.length;const r=hunterMatch(c,st,bot,memo,opt);pay.push(r.value);perr.push(r.err)}
 let o2=opt;if(opt.policies){o2=Object.assign({},opt,{record:new Map()});opt.policies.set(key,o2.record)}
 const t0=Date.now(),R=hunterRound(act.map(s=>counts[s]),0,act.indexOf(starter),bot,pay,o2),res={value:R.value,err:R.err+Math.max.apply(null,perr),nodes:R.nodes,ms:Date.now()-t0,pay};
 if(opt.log)opt.log(key,res);memo.set(key,res);return res}
module.exports={makeBot,hunterRound,hunterMatch,sigFor};
