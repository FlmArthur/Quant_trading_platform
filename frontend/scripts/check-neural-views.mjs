import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
import {createRequire} from 'node:module';
import {build} from 'esbuild';
import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
const require=createRequire(import.meta.url);
const bundle=await build({entryPoints:[fileURLToPath(new URL('../src/NeuralStages.tsx',import.meta.url))],bundle:true,platform:'node',format:'cjs',write:false,external:['react']});
const module={exports:{}};
new Function('require','module','exports',bundle.outputFiles[0].text)(require,module,module.exports);
const View=module.exports.default;
const s={checkpoint_id:'direction-1',as_of:'2024-01-02T10:00',sample_id:50,objective:'direction',active_branches:'hybrid',
 input:Array.from({length:4},()=>[1,2,3,4,5]),trend:[[1,.1],[2,.2],[3,.3],[4,.4]],
 conv1:[[1,2,3,4],[5,6,7,8]],conv2:[[2,3,4,5],[6,7,8,9]],conv_kernel:[[[1],[1],[1],[1],[1]],[[1],[1],[1],[1],[1]]],conv_bias:[0,0],
 conv2_kernel:[[[1],[1]],[[1],[1]]],conv2_bias:[0,0],
 hidden:Array.from({length:4},()=>[.1,.2,.3,.4]),cell:Array.from({length:4},()=>[.3,.4,.5,.6]),
 gates:Array.from({length:4},()=>Array.from({length:4},()=>[.1,.2,.3,.4])),
 cnn_projection:[1,2,3,4],lstm_projection:[.1,.2,.3,.4],fusion:[1.1,2.2,3.3,4.4],output_weight:[.1,.2,.3,.4],output_bias:.1,output:1,prediction:.73};
const props={snapshot:s,step:0,setStep:()=>{},channel:0,setChannel:()=>{},onMask:()=>{},onReset:()=>{}};
const keys=['convolution','memory','fusion'];
const views=keys.map((key,i)=>{
 const html=renderToStaticMarkup(React.createElement(View,{...props,stage:i+3}));
 assert.ok(html.includes(`data-testid="${key}-view"`));
 for(const other of keys.filter(v=>v!==key))assert.ok(!html.includes(`data-testid="${other}-view"`));
 return html;
});
assert.ok(views[0].includes('观察卷积层')&&views[0].includes('卷积位置'));
assert.ok(views[1].includes('LSTM时间步 t4')&&views[1].includes('遗忘门'));
assert.ok(views[2].includes('融合坐标')&&views[2].includes('全部 4 维贡献之和'));
assert.equal(renderToStaticMarkup(React.createElement(View,{...props,snapshot:null,stage:3})).includes('等待训练检查点'),true);
console.log('三种步骤显示独立视图，专属控件与空状态检查通过。');
