// Collect every exam result into out/exam.json and print a readable summary.
const fs=require('fs'),{NAMES,LEARN}=require('./lib.js'),D=require('./duel.js');
const rd=f=>fs.existsSync(`${__dirname}/out/${f}`)?JSON.parse(fs.readFileSync(`${__dirname}/out/${f}`)):null;
const K={'1,1,0':rd('key_1_1_0.json'),'2,1,0':rd('key_2_1_0.json'),'2,1,1':rd('key_2_1_1.json'),'2,2,0':rd('key_2_2_0.json')};
const HU={};for(const m of ['duel1','duel2','duel3','table1','table2','table3'])HU[m]=rd(`hunter_${m}.json`);
// tighter re-runs replace a looser result for the same student
for(const f of ['hunter_table2r.json','hunter_table2s.json']){const HR=rd(f);if(HR&&HU.table2)for(const id in HR.res){const a=HU.table2.res[id],b=HR.res[id];if(b&&!b.fail&&(!a||a.fail||b.err<a.err))HU.table2.res[id]=b}}
const GENS=['orig','old','mid','new'],ids=[];for(const g of GENS)for(const n of LEARN.concat(['ismcts']))ids.push(g+':'+n);ids.push('math','random');
const sum=(zs,f)=>zs.reduce((t,z)=>t+f(z),0),pct=x=>x===null||x===undefined||isNaN(x)?null:+(100*x).toFixed(2);
// papers in each family: [key, seat]; and the six round states of a duel that starts with up to 2 dice against 1
const FAM={one:[['1,1,0',0],['1,1,0',1]],three:[['2,1,0',0],['2,1,0',1],['2,1,1',0],['2,1,1',1]],two:[['2,2,0',0],['2,2,0',1]]};
function paper(id,key,seat){const k=K[key];return k&&k.res[id+'@'+seat]||null}
function fam(id,list){const P=list.map(([k,s])=>paper(id,k,s));if(P.some(p=>!p))return null;const agg=sel=>{const zs=P.map(sel),n=sum(zs,z=>z.n);return{n,acc:pct(sum(zs,z=>z.ok)/n),lost:pct(sum(zs,z=>z.loss)/n),blunder:pct(sum(zs,z=>z.bl)/n)}};
 const e=k=>sum(P,p=>p.err[k]),et=e('badCall')+e('missedCall')+e('badRaise')+e('badOpen');
 return{sampled:P[0].sampled||0,paper:agg(p=>p.std),own:agg(p=>p.act),sanity:agg(p=>p.tier.sanity),easy:agg(p=>p.tier.easy),hard:agg(p=>p.tier.hard),open:agg(p=>p.kind.open),reply:agg(p=>p.kind.reply),
  calledProven:pct(sum(P,p=>p.calledProven)/sum(P,p=>p.nProven)),missedImpossible:pct(sum(P,p=>p.missedImpossible)/Math.max(1e-12,sum(P,p=>p.nImpossible))),
  callRate:pct(sum(P,p=>p.callRate.bot)/sum(P,p=>p.callRate.n)),keyCallRate:pct(sum(P,p=>p.callRate.key)/sum(P,p=>p.callRate.n)),
  where:et>0?{badCall:pct(e('badCall')/et),missedCall:pct(e('missedCall')/et),badRaise:pct(e('badRaise')/et),badOpen:pct(e('badOpen')/et)}:null}}
// chance of winning a whole duel against the perfect player, from round-win chances (w = function(key,seat) -> chance the student wins that round)
function matches(w){const p11b=w('1,1,0',0),p11o=w('1,1,0',1),p21b=w('2,1,0',0),p21o=w('2,1,1',0),p12b=w('2,1,1',1),p12o=w('2,1,0',1);if([p11b,p11o,p21b,p21o,p12b,p12o].some(x=>x===null))return null;
 const M11=(p11b+p11o)/2,M21=s=>{const p=s?p21b:p21o;return p+(1-p)*p11b},M12=s=>{const p=s?p12b:p12o;return p*p11o},three=(M21(1)+M21(0)+M12(1)+M12(0))/4,r={one:M11,three};
 const p22b=w('2,2,0',0),p22o=w('2,2,0',1);if(p22b!==null&&p22o!==null){const M22=s=>{const p=s?p22b:p22o;return p*M21(0)+(1-p)*M12(1)};r.two=(M22(1)+M22(0))/2}return r}
const exam={students:{},names:NAMES,keys:{}};
for(const k in K)if(K[k])exam.keys[k]={gap:K[k].gap,value:K[k].value};
const perfect=matches((k,s)=>K[k]?K[k].value[s]:null);exam.perfectMatch=perfect;
for(const id of ids){const m=matches((k,s)=>{const p=paper(id,k,s);return p?p.win:null}),S={key:{},hunter:{}};let any=false;
 for(const f in FAM){const r=fam(id,FAM[f]);if(r){S.key[f]=r;any=true}}
 const all=fam(id,FAM.one.concat(FAM.three));if(all)S.key.all=all;if(m)S.match={one:pct(m.one),three:pct(m.three),two:m.two===undefined?null:pct(m.two)};
 for(const mo in HU){const h=HU[mo]&&HU[mo].res[id];if(h&&!h.fail){S.hunter[mo]={v:pct(h.value),err:pct(h.err),sec:Math.round(h.sec)};any=true}}
 if(any)exam.students[id]=S}
// size of the answer key: every (bid sequence, hand) where a player must decide
let situations=0;for(const k of Object.keys(K)){if(!K[k])continue;const[a,b]=k.split(',').map(Number),G=D.makeGame(a,b,0),half=Math.pow(2,G.NB)/2;situations+=half*(G.Hn[0]+G.Hn[1])}exam.situations=Math.round(situations);
fs.writeFileSync(`${__dirname}/out/exam.json`,JSON.stringify(exam));
// ---- printable summary
const pad=(s,n)=>String(s===null||s===undefined?'–':s).padStart(n),nm=id=>id.includes(':')?id.split(':')[1]:id;
console.log('answer key covers',exam.situations.toLocaleString(),'situations; perfect player duel values',JSON.stringify(perfect));
console.log('\nANSWER KEY (endgames with up to 3 dice on the table)   acc = % of decisions within half a point of the best move, lost = points of round-win chance given up per decision');
console.log('student'.padEnd(12)+' acc    lost  blunder | sanity  easy   hard | duel 1v1  duel 3 dice | calls proven  misses impossible | call rate (key)');
for(const id of ids){const S=exam.students[id];if(!S||!S.key.all)continue;const a=S.key.all;console.log((id+(a.sampled?'*':'')).padEnd(12)+pad(a.paper.acc,5)+pad(a.paper.lost,7)+pad(a.paper.blunder,8)+' |'+pad(a.sanity.acc,6)+pad(a.easy.acc,7)+pad(a.hard.acc,7)+' |'+pad(S.match&&S.match.one,8)+pad(S.match&&S.match.three,12)+'   |'+pad(a.calledProven,8)+pad(a.missedImpossible,14)+'        |'+pad(a.callRate,6)+' ('+a.keyCallRate+')')}
const sh=exam.students['math'].key.all;console.log('share of decisions by tier: sanity',(100*sh.sanity.n/sh.paper.n).toFixed(1),'easy',(100*sh.easy.n/sh.paper.n).toFixed(1),'hard',(100*sh.hard.n/sh.paper.n).toFixed(1),' | opening',(100*sh.open.n/sh.paper.n).toFixed(1),'reply',(100*sh.reply.n/sh.paper.n).toFixed(1));
console.log('\nBY STAGE (first any-dice version -> halfway -> current), learners:');
for(const n of LEARN.concat(['ismcts'])){const g=(st,f)=>{const S=exam.students[st+':'+n];try{const v=f(S);return v===undefined||v===null?'–':v}catch(e){return'–'}},row=f=>['orig','old','mid','new'].map(st=>pad(g(st,f),7)).join(' ');
 console.log(NAMES[n].padEnd(14),'accuracy',row(S=>S.key.all.paper.acc),'| hard',row(S=>S.key.all.hard.acc),'| lost/dec',row(S=>S.key.all.paper.lost),'| vs perfect 3-dice',row(S=>S.match.three));
 console.log(''.padEnd(14),'hunter duel2',row(S=>S.hunter.duel2.v),'| table1',row(S=>S.hunter.table1.v),'| table2',row(S=>S.hunter.table2.v+(S.hunter.table2.err>.05?'+'+S.hunter.table2.err:'')),'| duel3',row(S=>S.hunter.duel3.v))}
console.log('\nPERFECT HUNTER: % of matches the best possible counter-strategy wins (duels: 50 = unbeatable; tables: 25 = fair share)');
const modes=Object.keys(HU).filter(m=>HU[m]);console.log('student'.padEnd(12)+modes.map(m=>pad(m,14)).join(''));
for(const id of ids){const S=exam.students[id];if(!S||!Object.keys(S.hunter).length)continue;console.log(id.padEnd(12)+modes.map(m=>{const h=S.hunter[m];return pad(h?h.v+(h.err>.05?'+'+h.err:''):'–',14)}).join(''))}
