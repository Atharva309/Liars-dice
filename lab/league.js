// Tournaments run with the exact JS the page ships (site/table.js). The previous generation is loaded from old_table.js for comparisons.
// node league.js league <dice> <games> | mixed <dice> <games> | vsold <dice> <games per bot> | duel <dice> <games per pairing> | time <dice>
const fs=require('fs'),dir=__dirname,rd=f=>JSON.parse(fs.readFileSync(dir+'/'+f));
function logic(file,ret){const src=fs.readFileSync(dir+'/'+file,'utf8'),a=src.indexOf('//<logic>'),b=src.indexOf('//</logic>');return new Function(src.slice(a,b)+';return '+ret)()}
const NEW=logic('site/table.js','{legal,CALL,bq,bf,ptrue,policyN,BOT,decodeNet,setModel:m=>{MODEL=m},setStages:v=>{STAGES=v}}');
const OLD=logic('old_table.js','{policyOf,AG,AG5,decodeNet}');
const LEARN=['cfr','dqn','ppo','nfsp','mc'],AIS=LEARN.concat(['ismcts']),ALL=AIS.concat(['math','random']);
NEW.setModel(rd('site/bots/model.json'));
const dec=(L,d)=>d.layers?{kind:d.kind,temp:d.temp,net:L.decodeNet(d.layers)}:d;
NEW.BOT.lite=dec(NEW,rd('site/bots/lite.json'));if(process.env.STAGES)NEW.setStages(JSON.parse(process.env.STAGES));
for(const n of LEARN){NEW.BOT[n]=dec(NEW,rd('site/bots/'+n+'.json'));OLD.AG5[n]=dec(OLD,rd('site/ai5/'+n+'.json'));OLD.AG[n]=dec(OLD,rd('site/ai/'+n+'.json'))}
const [mode,S,G]=[process.argv[2],+process.argv[3],+process.argv[4]],cache={};
const bot=id=>cache[id]||(cache[id]=id.startsWith('new:')?NEW.policyN(id.slice(4)):OLD.policyOf(id.slice(4),S));
const roll=n=>Array.from({length:n},()=>1+Math.floor(Math.random()*6)).sort(),rnd=n=>Math.floor(Math.random()*n);
const st={},stat=id=>st[id]||(st[id]={bids:0,bluffs:0,calls:0,chances:0,right:0,called:0,caught:0,moves:0,ms:0});
function play(ids,start){const n=ids.length,cnt=Array(n).fill(start);let starter=rnd(n);
 while(cnt.filter(v=>v>0).length>1){const act=[];for(let s=0;s<n;s++)if(cnt[s]>0)act.push(s);const hands=cnt.map(roll),total=cnt.reduce((p,q)=>p+q,0);let last=-1,prev=-1,seat=starter,bidder=-1,hist=[],loser,lastBluff=false;
  for(;;){const o=[hands[seat],cnt[seat],total-cnt[seat],act.length-1,last,prev],acts=NEW.legal(last,total),i0=act.indexOf(seat),order=act.slice(i0).concat(act.slice(0,i0)),z=stat(ids[seat]);
   const t0=process.hrtime.bigint(),mv=bot(ids[seat])(o,acts,{order,total,hist,counts:cnt});z.ms+=Number(process.hrtime.bigint()-t0)/1e6;z.moves++;if(!acts.includes(mv))throw new Error('illegal move by '+ids[seat]+': '+mv);
   if(last>=0)z.chances++;
   if(mv===NEW.CALL){const f=NEW.bf(last);let c=0;for(const s of act)c+=hands[s].filter(d=>d===f).length;loser=c>=NEW.bq(last)?seat:bidder;z.calls++;if(loser!==seat)z.right++;
    if(lastBluff){const b=stat(ids[bidder]);b.called++;if(loser===bidder)b.caught++}break}
   lastBluff=NEW.ptrue(NEW.bq(mv),NEW.bf(mv),hands[seat],total-cnt[seat])<.5;z.bids++;if(lastBluff)z.bluffs++;
   hist.push([seat,mv]);prev=last;last=mv;bidder=seat;seat=act[(i0+1)%act.length]}
  cnt[loser]--;starter=loser;while(cnt[starter]===0)starter=(starter+1)%n}
 return cnt.findIndex(v=>v>0)}
function field(pool,games,seats){const played={},won={};pool.forEach(n=>{played[n]=0;won[n]=0});
 for(let g=0;g<games;g++){const p=pool.slice(),ids=[];for(let i=0;i<seats;i++)ids.push(p.splice(rnd(p.length),1)[0]);const w=play(ids,S);ids.forEach(n=>played[n]++);won[ids[w]]++}
 const win={};pool.forEach(n=>win[n]=+(won[n]/played[n]).toFixed(3));return{win,played}}
const t0=Date.now();let out;
if(mode==='league'){out=field(ALL.map(n=>'new:'+n),G,4);out.style={};for(const id in st){const z=st[id];out.style[id]={bluff:+(z.bluffs/z.bids).toFixed(3),call:+(z.calls/z.chances).toFixed(3),right:+(z.right/Math.max(1,z.calls)).toFixed(3),caught:+(z.caught/Math.max(1,z.called)).toFixed(3)}}}
else if(mode==='mixed')out=field(AIS.map(n=>'new:'+n).concat(AIS.map(n=>'old:'+n)),G,4);
else if(mode==='vsold'){out={win:{}};for(const n of AIS){let w=0;for(let g=0;g<G;g++){const ids=['old:'+n,'old:'+n,'old:'+n],me=rnd(4);ids.splice(me,0,'new:'+n);if(play(ids,S)===me)w++}out.win[n]=+(w/G).toFixed(3)}}
else if(mode==='duel'){out={res:{}};const ids=ALL.map(n=>'new:'+n);for(let i=0;i<ids.length;i++)for(let j=i+1;j<ids.length;j++){let w=0;for(let g=0;g<G;g++){if(g%2===0){if(play([ids[i],ids[j]],S)===0)w++}else if(play([ids[j],ids[i]],S)===1)w++}out.res[ALL[i]+'|'+ALL[j]]=[w,G]}}
else if(mode==='time'){field(ALL.map(n=>'new:'+n),G,4);out={ms:{}};for(const id in st)out.ms[id]=+(st[id].ms/st[id].moves).toFixed(3)}
out.games=G;out.sec=Math.round((Date.now()-t0)/1000);fs.writeFileSync(dir+`/out6/${mode}_${S}.json`,JSON.stringify(out));console.log(mode,S+' dice:',JSON.stringify(out.win||out.res||out.ms),'|',out.sec+'s');
