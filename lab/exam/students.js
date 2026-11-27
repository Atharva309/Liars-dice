// the students sitting the exam, and cached exact policies for them
'use strict';
const {distFor,LEARN}=require('./lib.js'),{sigFor}=require('./hunter.js');
const STAGES=[['orig','Original 2-dice bots'],['old','First any-dice version'],['mid','Halfway through the retrain'],['new','Current']];
function students(maxDice){const out=[];for(const[g]of STAGES){if(g==='orig'&&maxDice>2)continue;for(const n of LEARN)out.push(g+':'+n)}out.push('math','random');return out}
function cachedDist(id){const dist=distFor(id),sig=sigFor(id),M=new Map();const f=(o,acts,ctx)=>{const key=o[0].join('')+'/'+sig(o,ctx);let r=M.get(key);if(!r){r=dist(o,acts,ctx);M.set(key,r);f.evals++}return r};f.evals=0;return f}
module.exports={STAGES,students,cachedDist};
